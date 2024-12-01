import os
import json
import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from typing import Optional, Dict

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

model_name = "tiiuae/falcon-40b"

# Define BitsAndBytesConfig for 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=HUGGING_FACE_TOKEN,
    trust_remote_code=False  # Use native tokenizer
)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Load model with BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,  # Optimize CPU memory usage
    torch_dtype=torch.float16,  # Use float16 for faster computations
    trust_remote_code=False  # Use native model implementation
)

# Load your dataset
dataset = load_dataset('csv', data_files={'train': 'train.csv'})

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['prompt']
    outputs = examples['response']
    # Combine inputs and outputs
    texts = [inp + "\n" + out for inp, out in zip(inputs, outputs)]
    tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=1024)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing to save memory (optional)
# model.gradient_checkpointing_enable()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./falcon-40b-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,  # Enable mixed precision
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to=["none"],
    optim="adamw_torch",  # Use optimized optimizer
)

# Define data collator
def data_collator(data):
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
        'labels': torch.stack([torch.tensor(f['labels']) for f in data]),
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./falcon-40b-finetuned")
tokenizer.save_pretrained("./falcon-40b-finetuned")
