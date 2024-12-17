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

# Set the model to evaluation mode to disable dropout and other training-specific layers
model.eval()

# Define a function for generating responses with enhanced parameters
def generate_response(prompt, max_length=2000, temperature=0.2, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=6):
    """
    Generates a response from the model based on the input prompt.

    Args:
        prompt (str): The input text prompt.
        max_length (int): Maximum length of the generated response.
        temperature (float): Sampling temperature; higher values mean more random outputs.
        top_p (float): Nucleus sampling probability.
        repetition_penalty (float): Penalizes repeated tokens to reduce redundancy.
        no_repeat_ngram_size (int): Prevents the model from repeating n-grams of this size.

    Returns:
        str: The generated response text.
    """
    # Tokenize the input prompt
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # Ensure that input does not exceed model's max context length
    )
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    # Generate the response using the model's generate method with specified parameters
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,                     # Increased max_length for longer responses
            temperature=temperature,                   # Controls randomness
            top_p=top_p,                               # Nucleus sampling
            repetition_penalty=repetition_penalty,     # Penalizes repetition
            no_repeat_ngram_size=no_repeat_ngram_size, # Prevents repeating n-grams
            pad_token_id=tokenizer.pad_token_id,       # Use the pad token
            eos_token_id=tokenizer.eos_token_id,       # Stop generation at EOS token
            do_sample=True,                            # Enable sampling for variability
            early_stopping=True                        # Stop generation when EOS is reached
        )

    # Decode the generated tokens back into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# def dummy_generate_response(prompt, max_length=2000):
#     # Dummy response generator for testing; replace with actual model inference
#     return f"Generated response for: {prompt[:50]}..."

# Example usage of the inference script
if __name__ == "__main__":
    print("\nModel loaded. Ready for inference.\n")

    with open("self-bleu_data_input.json", "r") as file:
        input_data = json.load(file)

    output_data = []

    #while True:
        # prompt = input("Enter a prompt (or type 'exit' to quit): ")
    for i, item in enumerate(input_data):
        prompt = item["prompt"]
    
        if prompt.lower() == 'exit':
            break

        # Generate the response based on the input prompt
        responses = []
        for j in range(10):
            response = generate_response(prompt, max_length=2000)
            responses.append(response)
            print(f"Generated response for prompt {i + 1}'s {j + 1} output")

        output_data.append({
            "prompt": prompt,
            "generated": responses
        })
        #print(f"\nResponse:\n{response}\n")
        print(f"Generated final output for prompt {i + 1}")
        if i == 3:
            break
        i += 3
    
    output_json_file = "self-bleu_data_output.json"  # Replace with the desired output file name
    with open(output_json_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Responses successfully generated and saved to {output_json_file}.")
