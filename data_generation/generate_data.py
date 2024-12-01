import json
import pandas as pd
from typing import List, Dict
import os
import csv

def load_titles_list(file_path: str) -> List[str]:
    """
    Load titles from a text file, one title per line.

    Args:
        file_path (str): Path to 'titles_list.txt'.

    Returns:
        List[str]: List of titles.
    """
    titles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                title = line.strip()
                if title:
                    titles.append(title)
        print(f"Loaded {len(titles)} titles from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
    return titles

def load_cleaned_wikivoyage(file_path: str) -> Dict[str, str]:
    """
    Load 'cleaned_wikivoyage.json' and create a mapping from title to 'revision.text'.

    Args:
        file_path (str): Path to 'cleaned_wikivoyage.json'.

    Returns:
        Dict[str, str]: Mapping from title to 'revision.text'.
    """
    mapping = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Successfully loaded JSON data from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return mapping
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from '{file_path}'. {e}")
        return mapping

    # Navigate to 'mediawiki' -> 'page' list
    pages = data.get("mediawiki", {}).get("page", [])
    print(f"Total pages found in JSON: {len(pages)}")
    for idx, page in enumerate(pages, start=1):
        title = page.get("title")
        if not title:
            print(f"Page {idx}: Missing 'title'. Skipping.")
            continue
        # Check if the page has 'redirect' field; skip if present
        if "redirect" in page:
            continue
        revision = page.get("revision", {})
        text = revision.get("text", "")
        if isinstance(text, dict):
            # Extract '#text' from dict
            text = text.get("#text", "")
        elif not isinstance(text, str):
            # If text is neither dict nor str, skip
            print(f"Page {idx}: 'revision.text' is neither dict nor str. Skipping.")
            continue
        if not text:
            print(f"Page {idx}: 'revision.text' is empty. Skipping.")
            continue
        mapping[title] = text
    print(f"Total valid title to text mappings: {len(mapping)}")
    return mapping

def create_prompts_responses(titles: List[str], mapping: Dict[str, str]) -> List[Dict[str, str]]:
    """
    For each title, create a prompt and response.

    Args:
        titles (List[str]): List of titles from 'titles_list.txt'.
        mapping (Dict[str, str]): Mapping from title to 'revision.text'.

    Returns:
        List[Dict[str, str]]: List of dictionaries with 'title', 'prompt', 'response'.
    """
    data = []
    missing_titles = []
    for idx, title in enumerate(titles, start=1):
        text = mapping.get(title)
        if not text:
            print(f"Title {idx}: '{title}' not found in JSON. Skipping.")
            missing_titles.append(title)
            continue
        prompt = (
            f"I would like to travel from New York City to {title} for 7 days. "
            f"Give me a trip plan that focuses on restaurants. "
            f"The information below is about {title}; use it for your response.\n"
        )
        response = text.strip()
        data.append({
            "title": title,
            "prompt": prompt,
            "response": response
        })
    print(f"\nTotal prompts and responses created: {len(data)}")
    if missing_titles:
        print(f"Total titles missing in JSON: {len(missing_titles)}")
    return data

def save_to_csv(data: List[Dict[str, str]], output_file: str):
    """
    Save the prompts and responses to a CSV file, ensuring proper handling of quotes and line breaks.

    Args:
        data (List[Dict[str, str]]): List of dictionaries with 'title', 'prompt', 'response'.
        output_file (str): Path to output CSV file.
    """
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)

    # Optional: Sanitize response texts by replacing double quotes with escaped quotes
    df['response'] = df['response'].str.replace('"', '""')

    try:
        df.to_csv(
            output_file,
            index=False,
            encoding='utf-8-sig',         # Use UTF-8 with BOM for better compatibility
            quoting=csv.QUOTE_ALL,        # Enclose all fields in quotes
            #line_terminator='\n',         # Use LF as line terminator
            escapechar='\\'               # Escape character for special cases
        )
        print(f"Data successfully saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    """
    Main function to execute the prompt and response creation.
    """
    # Define file paths
    titles_list_path = "titles_list.txt"          # Path to 'titles_list.txt'
    cleaned_wikivoyage_path = "cleaned_wikivoyage.json"  # Path to 'cleaned_wikivoyage.json'
    output_csv_path = "prompts_responses.csv"     # Desired output CSV file name

    # Check if titles_list.txt exists
    if not os.path.exists(titles_list_path):
        print(f"Error: '{titles_list_path}' does not exist.")
        return
    # Check if cleaned_wikivoyage.json exists
    if not os.path.exists(cleaned_wikivoyage_path):
        print(f"Error: '{cleaned_wikivoyage_path}' does not exist.")
        return

    # Load titles
    titles = load_titles_list(titles_list_path)
    if not titles:
        print("No titles loaded. Exiting.")
        return

    # Load JSON mapping
    mapping = load_cleaned_wikivoyage(cleaned_wikivoyage_path)
    if not mapping:
        print("No mappings loaded. Exiting.")
        return

    # Create prompts and responses
    prompts_responses = create_prompts_responses(titles, mapping)

    # Save to CSV
    save_to_csv(prompts_responses, output_csv_path)

    # Optional: Validate the CSV by loading it back into Pandas
    try:
        df_loaded = pd.read_csv(output_csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"CSV successfully loaded back into Pandas. Shape: {df_loaded.shape}")
        print(df_loaded)
    except Exception as e:
        print(f"Error loading CSV into Pandas: {e}")

if __name__ == "__main__":
    main()

