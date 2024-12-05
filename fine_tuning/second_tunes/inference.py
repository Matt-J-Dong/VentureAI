from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import torch

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set in the .env file.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model and tokenizer
model_dir = "./trained_falcon_7b_test_data"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")

# Set model to evaluation mode
model.eval()

# Define a function for inference
def generate_response(prompt, max_length=100):
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,  # Ensure proper padding
            eos_token_id=tokenizer.eos_token_id,  # Stop generation at end-of-sequence token
        )
    
    # Decode the generated response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    print("\nModel loaded. Ready for inference.\n")

    while True:
        prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        
        response = generate_response(prompt, max_length=2000)
        print(f"\nResponse:\n{response}\n")
