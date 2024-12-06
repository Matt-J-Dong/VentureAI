import torch
# from torch.cuda.amp import
from torch import GradScaler, autocast #GradScaler and autocast moved here to avoid deprecation
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

# Clear CUDA cache
torch.cuda.empty_cache()
#scaler = GradScaler('cuda')  # Updated to avoid deprecation

# Load the Falcon-7B model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Load the model with device_map to distribute across GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    #torch_dtype=torch.float16,       # Added fp16 to increase speed
    torch_dtype=torch.bfloat16,  # Use bfloat16
    offload_folder="offload",        # Optional: specify folder for CPU offloading
    offload_state_dict=True          # Optional: offload state dict to CPU
)

# Read the data from train.csv
data = pd.read_csv('../data_generation/combined_results.csv')

# Load only the first 1% of the dataset for testing purposes
data = data.head(int(len(data) * 0.01))

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
        labels = input_ids.clone()  # Labels for causal language modeling
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Create the dataset and dataloader
dataset = PromptResponseDataset(data, tokenizer)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #batch size of 16 is too big! Crazy. A100 size
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #batch size of 16 is too big! Crazy. Coping V100

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop with mixed precision
epochs = 1
model.train()
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(model.device)  # Move to model's device
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        with autocast('cuda', dtype=torch.bfloat16):  # Ensure autocast uses fp16
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': loss.item()})

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item()}")


    print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

print("Training completed.")

output_dir = "./trained_falcon_7b_8_instruct"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
