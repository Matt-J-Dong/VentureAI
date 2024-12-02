import json
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
import unicodedata

def is_existing_city(city_name):
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        location = geolocator.geocode(city_name, timeout=10)
        return location is not None
    except GeopyError as e:
        print(f"Error checking city '{city_name}': {e}")
        return False


def normalize_and_clean_text(text):
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

# File path to your JSON file
input_file = "city_names.json"
output_file = "city_names_cleaned.json"

def clean_json_file(input_file, output_file):
    # Read data from the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if data is a list, otherwise handle appropriately
    if not isinstance(data, list):
        raise ValueError("The JSON file does not contain a list at the top level.")

    # Decode escaped characters and remove duplicates
    cleaned_data = list(set(item.encode('utf-8').decode('unicode_escape') for item in data))

    cleaned_data = list(
        set(
            normalize_and_clean_text(item.replace("Category:", "").strip())
            for item in cleaned_data
            if not item.startswith("File:")
            if not item.startswith("Wikivoyage:")
            if not item.startswith("Template:")
            if not item.startswith("Draft:")
            if not item.startswith("Help:")
            if not item.startswith("MediaWiki:")
            if not item.startswith("Mission:")
            if not item.startswith("Module:")
            if not item.startswith("Wts:")
            if not item.startswith("Shared:")
        )
    )
    cleaned_data = list(set(
        item.split("/")[0].strip() for item in cleaned_data if not re.search(r'\d', item)
    ))
    cleaned_data = [re.sub(r'\(.*?\)', '', item).strip() for item in cleaned_data]
    cleaned_data = list(set(item.replace("Greater", "").strip() for item in cleaned_data))
    cleaned_data = list(set(item.replace("Eastern", "").strip() for item in cleaned_data))
    cleaned_data = list(set(item.replace("Western", "").strip() for item in cleaned_data))
    cleaned_data = list(set(item.replace("Southern", "").strip() for item in cleaned_data))
    cleaned_data = list(set(item.replace("Northern", "").strip() for item in cleaned_data))
    cleaned_data = list(set(item.replace("Upper", "").strip() for item in cleaned_data))
    cleaned_data = list(set(item.replace("Lower", "").strip() for item in cleaned_data))

    cleaned_data = [item for item in cleaned_data if not item.isupper()]
    cleaned_data = [item for item in cleaned_data if item]

    cleaned_data = list(set(cleaned_data))
    
    
    cleaned_data.sort()
    # Write the cleaned and verified data back to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"Cleaned and verified data has been written to {output_file}")

# Clean the JSON file
clean_json_file(input_file, output_file)