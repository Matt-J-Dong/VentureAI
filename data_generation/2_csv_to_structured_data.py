import openai
import sys
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to use OpenAI's new API for solving a problem
def generate_solution_with_openai(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": f"{prompt}"}]

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

# Function to extract prompts and responses from a CSV file
def extract_prompts_and_responses(csv_file_path):
    csv.field_size_limit(sys.maxsize)
    prompts_responses = []

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate through each row in the CSV
        for row in reader:
            prompt = row['prompt']
            response = row['response']
            prompts_responses.append((prompt, response))

    return prompts_responses

# Function to save the results to a new CSV file
def save_results_to_csv(results, output_file_path):
    # Open the CSV file for writing
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["prompt", "response"])
        writer.writeheader()

        # Write each result row to the CSV
        for row in results:
            writer.writerow(row)

# Main function to process the CSV and generate results
def main():
    input_csv_file = './train.csv'
    output_csv_file = './results.csv'

    # Extract prompts and responses from the input CSV
    prompts_responses = extract_prompts_and_responses(input_csv_file)

    # Determine the number of rows to process (half of the dataset)
    half_length = len(prompts_responses) // 2
    prompts_responses = prompts_responses[half_length:]

    # List to store results
    results = []

    # Process each data point in the first half of the CSV
    for idx, (prompt, data) in enumerate(prompts_responses, start=1):
        print(f"Processing row {idx}...")  # Debugging line

        # Construct the input for OpenAI
        input_prompt = (
            "I am providing you with a Query that will ask you to plan a trip for me with instructions. "
            "I will also provide you with data regarding the location I want to travel to. "
            "Only use that data provided to answer the prompt; do not come up with anything new.\n"
        )
        input_prompt += f"{data}\n"
        input_prompt += f"{prompt}\n"

        # Generate a solution using OpenAI
        response = generate_solution_with_openai(input_prompt)

        # Append the result to the list
        results.append({"prompt": prompt, "response": response})

    # Save the results to the output CSV
    save_results_to_csv(results, output_csv_file)
    print(f"Results saved to '{output_csv_file}'.")

# Run the main function
if __name__ == "__main__":
    main()
