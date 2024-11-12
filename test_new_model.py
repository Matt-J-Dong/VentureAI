import requests

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xl"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
	
# Specify max_length to increase output length
output = query({
    "inputs": "What is the answer to life, the universe, and everything?",
    "parameters": {
        "max_length": 300  # Set this to a higher number for longer output
    }
})

print(output)
