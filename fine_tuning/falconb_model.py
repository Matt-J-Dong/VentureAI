# falconb_model.py

import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

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
# 2. Prepare the Dataset
# =========================
class PromptResponseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # Batch size already reduced
        shuffle=True,  # Shuffle handled by Accelerator
        pin_memory=True,
        num_workers=4  # Adjust based on your CPU cores
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    return train_dataloader, val_dataloader

# =========================
# 3. Setup the Optimizer and Scheduler
# =========================
def setup_optimizer_scheduler(model, train_dataloader, epochs=3):
    optimizer = AdamW(model.parameters(), lr=3e-5)  # Lowered learning rate
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,        # Number of warmup steps
        num_training_steps=total_steps
    )
    return optimizer, scheduler

# =========================
# 4. Training Loop with Mixed Precision and Logging
# =========================
def train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, tokenizer, accelerator, epochs=3, patience=2):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        logging.info(f"Epoch {epoch+1}/{epochs}")
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            # No need to manually move tensors to device; Accelerator handles it
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss = loss.mean()  # Aggregate loss to a scalar

            # Backpropagation with Accelerator
            accelerator.backward(loss)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': loss.item()})
            logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {loss.item()}")

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss.item()}")

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")
        logging.info(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

        # =========================
        # 5. Validation Step
        # =========================
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                with autocast('cuda'):
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
        # 6. Early Stopping Check
        # =========================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            output_dir = "./trained_falcon_7b_b"
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir)
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

# =========================
# 5. Main Function to Launch Accelerate
# =========================
def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    # Add a unique pad token if it doesn't already exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

    print(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Prepare dataloaders
    train_dataloader, val_dataloader = prepare_dataloaders(tokenizer)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        torch_dtype=torch.float16,  # Use float16 to reduce memory usage
    )
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_scheduler(model, train_dataloader, epochs=3)

    # Prepare with accelerator.prepare()
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # Train
    train_model(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        tokenizer,
        accelerator,
        epochs=3,
        patience=2
    )

if __name__ == "__main__":
    main()
