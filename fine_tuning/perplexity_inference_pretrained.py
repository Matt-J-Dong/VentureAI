from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import torch
from transformers import BitsAndBytesConfig  # Import only if using BitsAndBytes for 8-bit inference
from peft import PeftModel  # Import PeftModel for loading LoRA adapters
import json

# Load environment variables from .env file
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Ensure that the Hugging Face token is set (optional, based on model access)
if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set in the .env file.")

# Determine the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the directory where the trained model and tokenizer are saved
model_dir = "./trained_falcon7b_7"  # Updated to match the training script's output directory

# Check if the model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Trained model directory not found at {model_dir}")

print("Loading tokenizer...")
# Load the tokenizer from the trained model directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print("Loading base model...")
# Load the base model without 8-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # Changed from bfloat16 to float16 for broader GPU compatibility
    device_map="auto"           # Automatically maps the model to available devices
)

print("Loading LoRA adapters...")
# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, model_dir)

print("Model is loaded ")

with open("perplexity_data_all.json", "r") as file:
    data = json.load(file)

total_loss = 0
for item in data:
    reference_text = item["reference"]
    inputs = tokenizer(reference_text, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]
    outputs = model(**inputs)
    total_loss += outputs.loss.item()

average_loss = total_loss / len(data)
print("average loss calculated")
fine_tuned_perplexity = torch.exp(torch.tensor(average_loss)).item()
print(f"Fine-Tuned Model Perplexity: {fine_tuned_perplexity:.2f}")


# # Example usage of the inference script
# if __name__ == "__main__":
#     print("\nModel loaded. Ready for inference.\n")

#     with open("perplexity_data_output.json", "r") as file:
#         input_data = json.load(file)

#     output_data = []

#     #while True:
#         # prompt = input("Enter a prompt (or type 'exit' to quit): ")
#     for i, item in enumerate(input_data):
#         prompt = item["prompt"]
#         reference = item["reference"]

#         if prompt.lower() == 'exit':
#             break

#         # Generate the response based on the input prompt
#         response = generate_response(prompt, max_length=2000)

#         output_data.append({
#             "prompt": prompt,
#             "reference": reference,  # Include reference if available
#             "generated": response
#         })
#         #print(f"\nResponse:\n{response}\n")
#         print(f"Generated output for prompt {i}")
    
#     output_json_file = "perplexity_data_output.json"  # Replace with the desired output file name
#     with open(output_json_file, "w") as file:
#         json.dump(output_data, file, indent=4)

#     print(f"Responses successfully generated and saved to {output_json_file}.")
