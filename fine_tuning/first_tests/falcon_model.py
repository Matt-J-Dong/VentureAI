import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from typing import Optional, Dict

#Our own code
#from tripadvisor import get_all_details 

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set in the .env file.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "tiiuae/falcon-40b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True,  # Enable 8-bit quantization
)

# Load your dataset
#dataset = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'validation.csv'})
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
tokenized_datasets = dataset.map(preprocess_function, batched=True)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["dense_h_to_4h", "dense_4h_to_h"],  # Target Falcon's feed-forward layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

#print(model)
#print(dataset)

training_args = TrainingArguments(
    output_dir="./falcon-40b-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,  # Enable mixed precision
    logging_steps=10,
    save_steps=500,
    #evaluation_strategy="steps",
    #eval_steps=500,
    save_total_limit=2,
    report_to=["none"],
)

def data_collator(data):
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
        'labels': torch.stack([torch.tensor(f['labels']) for f in data]),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    #eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./falcon-40b-finetuned")
tokenizer.save_pretrained("./falcon-40b-finetuned")
