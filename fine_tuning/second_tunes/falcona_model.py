import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

# =========================
# 1. Setup Logging
# =========================
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# =========================
# 2. Define Device and Initialize DDP
# =========================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can choose any free port
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    destroy_process_group()

# =========================
# 3. Load the Model and Tokenizer
# =========================
def load_model(tokenizer, rank):
    model_name = "tiiuae/falcon-7b"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for compatibility with GradScaler
    )

    # Resize token embeddings to account for the new pad token
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Move the model to the appropriate GPU
    model.to(rank)

    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    return ddp_model

# =========================
# 4. Prepare the Dataset
# =========================
class PromptResponseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        prompt = str(self.dataframe.iloc[idx]['prompt'])
        response = str(self.dataframe.iloc[idx]['response'])
        # Combine prompt and response with eos_token
        input_text = f"{prompt}{self.tokenizer.eos_token}{response}{self.tokenizer.eos_token}"
        # Tokenize the input text
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze()

        # Create labels: mask the prompt tokens
        labels = input_ids.clone()
        # Calculate the number of tokens in the prompt
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        # Mask the prompt tokens and first eos_token by setting them to -100
        labels[:prompt_length + 1] = -100  # +1 for eos_token

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def prepare_dataloaders(tokenizer):
    data_path = '../data_generation/combined_results.csv'
    data = pd.read_csv(data_path)

    # Remove duplicates to prevent the model from learning repetitive patterns
    data = data.drop_duplicates(subset=['prompt', 'response'])

    # Limit the dataset size for testing purposes
    data = data.head(100)

    # Split into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    train_dataset = PromptResponseDataset(train_data, tokenizer)
    val_dataset = PromptResponseDataset(val_data, tokenizer)

    return train_dataset, val_dataset

# =========================
# 5. Training Function
# =========================
def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    # Add a unique pad token if it doesn't already exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

    train_dataset, val_dataset = prepare_dataloaders(tokenizer)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4  # Adjust based on your CPU cores
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=4  # Adjust based on your CPU cores
    )

    model = load_model(tokenizer, rank)

    optimizer = AdamW(model.parameters(), lr=3e-5)  # Lowered learning rate

    epochs = 3  # Increased number of epochs
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,        # Number of warmup steps
        num_training_steps=total_steps
    )

    scaler = GradScaler()  # Correct initialization without arguments

    best_val_loss = float('inf')
    patience = 2
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        logging.info(f"Epoch {epoch+1}/{epochs}")
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss = loss.mean()  # Aggregate loss to a scalar

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': loss.item()})
            logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {loss.item()}")

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss.item()}")

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")
        logging.info(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

        # =========================
        # 7. Validation Step
        # =========================
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss = loss.mean()  # Aggregate loss to a scalar
                    total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss}")
        logging.info(f"Validation Loss after epoch {epoch+1}: {avg_val_loss}")

        # =========================
        # 8. Early Stopping Check
        # =========================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            output_dir = "./trained_falcon_7b_a"
            if world_size > 1 and rank == 0:
                torch.distributed.barrier()  # Ensure all processes reach this point
                model.module.save_pretrained(output_dir)  # Save only on rank 0
                tokenizer.save_pretrained(output_dir)
                print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss}")
                logging.info(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss}")
            elif world_size == 1:
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss}")
                logging.info(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                logging.info("Early stopping triggered.")
                break

    print("Training completed.")
    logging.info("Training completed.")

    cleanup()

# =========================
# 6. Main Function to Launch DDP
# =========================
def main():
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available for training.")

    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
