import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
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
# 2. Clear CUDA Cache
# =========================
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================
# 3. Load the Model and Tokenizer
# =========================
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a unique pad token if it doesn't already exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Define a new pad token

tokenizer.pad_token = '[PAD]'  # Set pad_token to the new unique token

print(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# Load the model in float32 precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,  # Use float32
    offload_folder="offload",    # Optional: specify folder for CPU offloading
    offload_state_dict=True       # Optional: offload state dict to CPU
)

# Resize token embeddings to account for the new pad token
model.resize_token_embeddings(len(tokenizer))

# =========================
# 4. Prepare the Dataset
# =========================
data_path = '../data_generation/combined_results.csv'
data = pd.read_csv(data_path)

# Remove duplicates to prevent the model from learning repetitive patterns
data = data.drop_duplicates(subset=['prompt', 'response'])

# Limit the dataset size for testing purposes
data = data.head(100)

# Split into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# Define a custom Dataset class with label masking
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

# Create the training and validation datasets
train_dataset = PromptResponseDataset(train_data, tokenizer)
val_dataset = PromptResponseDataset(val_data, tokenizer)

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =========================
# 5. Setup the Optimizer and Scheduler
# =========================
optimizer = AdamW(model.parameters(), lr=3e-5)  # Lowered learning rate

epochs = 3  # Increased number of epochs
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,        # Number of warmup steps
    num_training_steps=total_steps
)

# Initialize GradScaler for mixed precision training with float16
scaler = GradScaler(device)

# =========================
# 6. Training Loop with Mixed Precision and Logging
# =========================
best_val_loss = float('inf')
patience = 2
epochs_no_improve = 0

model.train()
for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}/{epochs}")
    print(f"Epoch {epoch+1}/{epochs}")
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        with autocast(device):  # Use default dtype (float16) for autocast
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

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
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            with autocast(device):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss}")
    logging.info(f"Validation Loss after epoch {epoch+1}: {avg_val_loss}")
    model.train()

    # =========================
    # 8. Early Stopping Check
    # =========================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the best model
        output_dir = "./trained_falcon_7b_a"
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

# =========================
# 9. Save the Trained Model and Tokenizer
# =========================
# (Already saved within the early stopping condition)
