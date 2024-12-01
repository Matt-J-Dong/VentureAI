import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from dotenv import load_dotenv
from typing import Dict

# Enable cuDNN benchmarking for optimized GPU performance
torch.backends.cudnn.benchmark = True

# Load environment variables from .env file
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set in the .env file.")

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "tiiuae/falcon-7b"  # Updated to Falcon-7B for quicker testing

# Load tokenizer without device mapping
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HUGGING_FACE_TOKEN,
    trust_remote_code=False  # Set to False as per your requirement
)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Load model without device_map and without LoRA or quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for faster computations
    trust_remote_code=False      # Set to False as per your requirement
)

# Move model to the primary GPU (Trainer will handle distribution)
model.to(device)
print(f"Successfully loaded the model '{model_name}'.")

# Load your dataset
try:
    dataset = load_dataset('csv', data_files={'train': 'train.csv'})
    print("Successfully loaded the dataset.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise e

# Preprocessing function
def preprocess_function(examples: Dict) -> Dict:
    inputs = examples['prompt']
    outputs = examples['response']
    # Combine inputs and outputs
    texts = [inp + "\n" + out for inp, out in zip(inputs, outputs)]
    tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=1024)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

# Apply preprocessing
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)
print("Preprocessing completed.")

# Optionally, split into training and validation sets
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Define training arguments with updated eval_strategy
training_args = TrainingArguments(
    output_dir="./falcon-7b-finetuned",
    per_device_train_batch_size=2,      # Adjust based on your GPU memory
    gradient_accumulation_steps=4,      # To simulate a larger batch size
    num_train_epochs=3,                 # Adjust based on your needs
    learning_rate=5e-5,                 # A common starting point
    fp16=True,                          # Enable mixed precision
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",              # Updated from evaluation_strategy
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to=["none"],                 # Disable reporting to WandB or other services
    optim="adamw_torch",                # Use optimized optimizer
    dataloader_num_workers=4,           # Utilize data loading parallelism
)

# Define data collator (optional, as Trainer can handle it)
def data_collator(data: Dict) -> Dict:
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
        'labels': torch.stack([torch.tensor(f['labels']) for f in data]),
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Start training
try:
    trainer.train()
    print("Training completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")
    raise e

# Save the fine-tuned model and tokenizer
try:
    trainer.save_model("./falcon-7b-finetuned")
    tokenizer.save_pretrained("./falcon-7b-finetuned")
    print("Model and tokenizer saved successfully.")
except Exception as e:
    print(f"Error saving the model/tokenizer: {e}")
    raise e
