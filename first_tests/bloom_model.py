import wandb
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import accelerate
import os
from dotenv import load_dotenv

# Set the environment variable for the Hugging Face token
load_dotenv()
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify the model name
model_name = "bigscience/bloom"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

# Create the text-generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,  # Pass the model name directly
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency if supported
    device_map="auto",           # Automatically map the model to available devices (handles multi-GPU)
) 

prompt = (
    "I would like to travel from New York City to 's-Hertogenbosch for 7 days. Give me a trip plan that focuses on museums."
)
# Generate sequences
sequences = pipeline(
    prompt,
    max_length=300,
    do_sample=True,
    top_k=10,
    num_return_sequences=2,
    eos_token_id=tokenizer.eos_token_id,
)

# Print the generated sequences
for idx, seq in enumerate(sequences, 1):
    print(f"Result {idx}: {seq['generated_text']}\n")

