from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import torch
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

print("Loading tokenizer...")
# Load the tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", use_auth_token=HUGGING_FACE_TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading pretrained model...")
# Load the pretrained Falcon7b model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b",
    torch_dtype=torch.float16,  # Use half precision for faster inference
    device_map="auto",         # Automatically map the model to available devices
    use_auth_token=HUGGING_FACE_TOKEN
)

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
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            early_stopping=True
        )

    # Decode the generated tokens back into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Example usage of the inference script
if __name__ == "__main__":
    print("\nModel loaded. Ready for inference.\n")

    # Load input data from JSON file
    with open("perplexity_data_input.json", "r") as file:
        input_data = json.load(file)

    output_data = []

    for i, item in enumerate(input_data):
        prompt = item["prompt"]
        reference = item["reference"]

        # Generate the response based on the input prompt
        response = generate_response(prompt, max_length=2000)

        output_data.append({
            "prompt": prompt,
            "reference": reference,  # Include reference if available
            "generated": response
        })
        print(f"Generated output for prompt {i}")
    
    # Save the output data to a JSON file
    output_json_file = "perplexity_data_output_pretrained.json"
    with open(output_json_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Responses successfully generated and saved to {output_json_file}.")
