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

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./falcon-40b-finetuned")
model = AutoModelForCausalLM.from_pretrained("./falcon-40b-finetuned", device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prepare input text
input_text = "Provide a 4-day itinerary for 3 friends traveling from Chicago to New Orleans from July 1st to July 4th, 2022, with a budget of $4,000.\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

# Generate output
outputs = model.generate(input_ids, max_length=400, num_return_sequences=2, do_sample=True, temperature=0.4)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)


