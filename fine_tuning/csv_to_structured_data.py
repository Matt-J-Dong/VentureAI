import openai
import sys

#from google.colab import userdata

openai.api_key = ''

# Function to use OpenAI's new API for solving a math problem
def generate_solution_with_openai(prompt, model="gpt-4o-mini"):
    # Prepare the chat-based input prompt for the newer API format
    messages = [{"role": "user", "content": f"{prompt}"}]

    # Make an API call to OpenAI's model using the new chat completion method
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )

        solution = response.choices[0].message['content'].strip()
        return solution

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

import csv

def extract_prompts_and_responses(csv_file_path):
    csv.field_size_limit(sys.maxsize)
    # Dictionary to store prompts as keys and responses as values
    prompts_responses = {}

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # Read the CSV file
        reader = csv.DictReader(file)

        # Iterate through each row in the CSV
        for row in reader:
            # Extract the prompt and response
            prompt = row['prompt']
            response = row['response']
            
            # Add them to the dictionary
            prompts_responses[prompt] = response

    return prompts_responses

# Generate a solution with minimal prompting
csv_file_path = 'fine_tuning/train.csv'

# Extract the prompts and responses
prompts_responses_dict = extract_prompts_and_responses(csv_file_path)

# Print the result
input = ""

for prompt, data in prompts_responses_dict.items():
    input = "I am providing you with a Query that will ask you to plan a trip for me with instructions. I will also provide you with data regarding the location I want to travel to. Only use that data provided to answer the prompt; Do not come up with anything new.\n"
    input += f"{data}\n"
    input += f"{prompt}\n"
    #print(input)
    break


solution = generate_solution_with_openai(input)

# Print the results
#print("Problem:", problem)
print("Solution:", solution)