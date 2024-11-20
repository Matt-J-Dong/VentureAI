import os
import json
import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline
from typing import Optional, Dict
from tripadvisor import get_all_details

# ===========================
# Configuration and Setup
# ===========================

# Load environment variables from .env file
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set in the .env file.")

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================
# Load and Parse JSON Data
# ===========================

def load_city_data(file_path: str) -> Dict:
    """
    Load JSON data from the specified file path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict: Parsed JSON data as a dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            city_data = json.load(file)
        print(f"Successfully loaded city data from {file_path}.")
        return city_data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"The file {file_path} is not a valid JSON file.")

# Path to the cleaned_wikivoyage.json file
CITY_DATA_FILE = "cleaned_wikivoyage.json"

# Load the city data
city_data = load_city_data(CITY_DATA_FILE)

# ===========================
# Define Utility Functions
# ===========================

def identify_cities_in_input(user_input: str, city_data: Dict) -> list:
    """
    Identify all cities mentioned in the user input that are present in the city_data.

    Args:
        user_input (str): The user's input string.
        city_data (Dict): The city data dictionary.

    Returns:
        list: A list of matched city dictionaries with extracted information.
    """
    matched_cities = []
    pages = city_data.get("mediawiki", {}).get("page", [])

    print(f"Total pages to process: {len(pages)}")  # Debugging statement

    for page in pages:
        name = page.get("title", "")
        alias = page.get("redirect", "")

        # Debugging: Print the types and values of name and alias
        #print(f"Processing page: name={name} (type: {type(name)}), alias={alias} (type: {type(alias)})")

        # Ensure name and alias are strings
        if not isinstance(name, str):
            print(f"Skipping page with non-string name: {name}")
            continue  # Skip this page if name is not a string
        if not isinstance(alias, str):
            print(f"Alias is not a string for city: {name}. Setting alias to empty string.")
            alias = ""

        # Only proceed if the city name or alias is in the user input
        name_pattern = rf"\b{re.escape(name)}\b"
        alias_pattern = rf"\b{re.escape(alias)}\b" if alias else ''

        if re.search(name_pattern, user_input, re.IGNORECASE) or \
           (alias and re.search(alias_pattern, user_input, re.IGNORECASE)):
            # Simple extraction from revision.text (this can be improved)
            revision_text = page.get("revision", {}).get("text", "")
            country = ""
            province = ""
            description = ""

            # Attempt to extract country and province from the description
            # This is a simplistic approach and may need more sophisticated parsing
            match = re.search(r"'''[^']+''' is in ([^,]+), ([^\.]+)\.", revision_text)
            if match:
                province = match.group(1).strip()
                country = match.group(2).strip()
                print(f"Extracted province: {province}, country: {country}")  # Debugging statement
            else:
                print(f"No match found for country and province extraction in city: {name}")  # Debugging statement

            # Extract description as the first sentence
            if revision_text:
                description = revision_text

            city_info = {
                "name": name,
                "alias": alias,
                "country": country,
                "province": province,
                "description": description
            }

            matched_cities.append(city_info)
            print(f"Matched City: {city_info['name']}")  # Debugging statement
        else:
            continue
            #print(f"No match for city: {name}")  # Debugging statement

    if matched_cities:
        print(f"Identified cities in input: {[city['name'] for city in matched_cities]}")
    else:
        print("No cities identified in the user input.")

    return matched_cities

def generate_prompt_with_city_context(user_input: str, matched_cities: list) -> str:
    """
    Generate a prompt by combining city context with the user's input.

    Args:
        user_input (str): The user's input string.
        matched_cities (list): A list of matched city dictionaries.

    Returns:
        str: The generated prompt.
    """
    contexts = []
    for city in matched_cities:
        context_parts = [f"{city['name']}, commonly known as {city.get('alias', city['name'])},"]
        if city.get('country'):
            context_parts.append(f"is a city in {city['country']}")
        if city.get('province'):
            context_parts.append(f"and the capital of {city['province']}.")
        if city.get('description'):
            context_parts.append(f"{city['description']}.")
        
        context = " ".join(context_parts)
        contexts.append(context)
        print(f"Constructed Context for {city['name']}: {context}")  # Debugging statement

    combined_context = " ".join(contexts)

    if combined_context:
        prompt = "###INFORMATION:\n" + combined_context + "\n\n###USER INPUT:\n" + user_input
    else:
        prompt = user_input

    print("Generated Prompt:\n", prompt, "-END OF GENERATED PROMPT-\n")  # Debugging statement
    return prompt

# ===========================
# Setup Hugging Face Pipeline
# ===========================

def setup_text_generation_pipeline(model_name: str, token: str):
    """
    Set up the Hugging Face text generation pipeline.

    Args:
        model_name (str): The name of the Hugging Face model.
        token (str): Hugging Face API token.

    Returns:
        pipeline: The Hugging Face text generation pipeline.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print(f"Successfully loaded model '{model_name}'.")
        return text_gen_pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to set up the text generation pipeline: {e}")

# Model configuration
MODEL_NAME = "mistralai/Mistral-Nemo-Base-2407"

# Initialize the text generation pipeline
text_generation_pipeline = setup_text_generation_pipeline(MODEL_NAME, HUGGING_FACE_TOKEN)

# ===========================
# Generate Itinerary
# ===========================

def generate_itinerary(user_input: str, city_data: Dict, pipeline, max_length: int = 300, 
                      do_sample: bool = True, top_k: int = 10, num_return_sequences: int = 2) -> list:
    """
    Generate a travel itinerary based on user input.

    Args:
        user_input (str): The user's travel request.
        city_data (Dict): The city data dictionary.
        pipeline: The Hugging Face text generation pipeline.
        max_length (int): Maximum length of the generated sequence.
        do_sample (bool): Whether to use sampling; use greedy decoding otherwise.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        num_return_sequences (int): The number of independently computed returned sequences.

    Returns:
        list: A list of generated itineraries.
    """
    # Identify cities in the user input
    matched_cities = identify_cities_in_input(user_input, city_data)
    
    # Generate the prompt with city context
    prompt = generate_prompt_with_city_context(user_input, matched_cities)
    
    print("\nGenerated Prompt:\n", prompt, "\n")
    
    # Generate sequences based on the prompt
    try:
        sequences = pipeline(
            prompt,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=pipeline.tokenizer.eos_token_id,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate text: {e}")
    
    # Extract generated texts
    itineraries = [seq['generated_text'] for seq in sequences]
    return itineraries

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    import json

    # TODO: we will use this set later as a way to automate this. for now it is manual
    with open('city_names.json', 'r') as json_file:
        city_names = set(json.load(json_file))  

    city_to = "Paris"
    locations_info = get_all_details(city_to, {city_to: {}})
    
    # Example user query
    user_input_example = f"I would like to travel from New York City to {city_to} for 7 days. Give me a trip plan that focuses on restaurants. The information below is about {city_to}; use it for your response\n"
    user_input_example = user_input_example + str(locations_info)    
    

    # Generate itineraries
    itineraries = generate_itinerary(
        user_input=user_input_example,
        city_data=city_data,
        pipeline=text_generation_pipeline,
        max_length=1000,  # Adjusted for more detailed itineraries
        do_sample=True,
        top_k=50,        # Adjusted for more diversity
        num_return_sequences=4  # Adjusted to generate 2 itineraries
    )

    # Print the generated itineraries
    for idx, itinerary in enumerate(itineraries, 1):
        print(f"--- Itinerary {idx} ---\n{itinerary}\n--- End of Itinerary {idx}---\n")
