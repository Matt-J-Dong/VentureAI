import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
import argparse
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    args = parser.parse_args()
    local_rank = args.local_rank

    # Initialize the process group
    dist.init_process_group(backend='nccl')

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Load the model and tokenizer
    model_name = "gpt2"  # Replace with "tiiuae/falcon-7b-instruct" if desired

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

    # Load the model in default precision (float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
    model.to(device)

    # Wrap the model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Read the data from the CSV file
    data = pd.read_csv('../data_generation/combined_results.csv')

    # Load only the first 30% of the dataset for testing purposes
    data = data.head(int(len(data) * 0.3))

    # Define a custom Dataset class
    class PromptResponseDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length=512):
            self.dataframe = dataframe.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            prompt = str(self.dataframe.iloc[idx]['prompt'])
            response = str(self.dataframe.iloc[idx]['response'])
            # Combine prompt and response
            input_text = f"{prompt}{tokenizer.eos_token}{response}{tokenizer.eos_token}"
            # Tokenize the input text
            encoding = tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    # Create the dataset and DistributedSampler
    dataset = PromptResponseDataset(data, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Initialize the GradScaler for mixed precision
    scaler = GradScaler()

    # Add this list to store the loss values
    losses = []

    # Training loop with mixed precision and loss logging
    epochs = 1
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(dataloader, desc=f"Rank {local_rank} Training", leave=False) if local_rank == 0 else dataloader
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast('cuda', dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping to prevent overflow
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Log the loss
            if local_rank == 0:
                losses.append(loss.item())
                progress_bar.set_postfix({'loss': loss.item()})

            if (batch_idx + 1) % 10 == 0 and local_rank == 0:
                print(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item()}")

        if local_rank == 0:
            print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

    if local_rank == 0:
        print("Training completed.")
        output_dir = "./trained_model"
        model.module.save_pretrained(output_dir)  # Use model.module when saving
        tokenizer.save_pretrained(output_dir)

        # Save losses for visualization
        with open("losses.txt", "w") as f:
            f.write("\n".join(map(str, losses)))

    # Plot the training loss curve
    if local_rank == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Batch Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve.png")
        plt.show()

    # Clean up the distributed environment
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
