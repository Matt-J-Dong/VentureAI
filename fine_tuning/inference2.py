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

# Define a function for inference with enhanced generation parameters
def generate_response(prompt, max_length=2000, temperature=0.7, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3):
    # Tokenize the prompt with attention mask
    encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    # Generate response with adjusted parameters
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,               # Increased max_length for longer responses
            temperature=temperature,             # Controls randomness
            top_p=top_p,                         # Nucleus sampling
            repetition_penalty=repetition_penalty,  # Penalizes repetition
            no_repeat_ngram_size=no_repeat_ngram_size,  # Prevents repeating n-grams
            pad_token_id=tokenizer.pad_token_id,    # Use the new pad token
            eos_token_id=tokenizer.eos_token_id,    # Stop generation at eos token
            do_sample=True,                        # Enable sampling for variability
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

