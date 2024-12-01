import json
from typing import List, Dict

def load_titles_from_json(file_path: str) -> List[str]:
    """
    Load all titles from a JSON file with the specified structure,
    excluding titles that start with certain unwanted prefixes
    and titles that have a 'redirect' section.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        List[str]: A list of all titles extracted from the JSON, excluding unwanted ones.
    """
    # Define prefixes to exclude
    unwanted_prefixes = ["Category:", "MediaWiki:", "File:", "Template:", "Module:", "Wikivoyage:", "Help:", "Wts:"]

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Successfully loaded JSON data from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
        return []

    # Navigate to the 'mediawiki' -> 'page' list
    pages = data.get("mediawiki", {}).get("page", [])
    if not pages:
        print("Warning: No 'page' entries found in the JSON data.")
        return []

    titles = []
    for idx, page in enumerate(pages, start=1):
        title = page.get("title")
        if title:
            # Check if the title starts with any unwanted prefix
            if any(title.startswith(prefix) for prefix in unwanted_prefixes):
                #print(f"Page {idx}: '{title}' starts with an unwanted prefix. Skipped.")
                continue  # Skip this title

            # Check if the page has a 'redirect' section
            if "redirect" in page:
                #print(f"Page {idx}: '{title}' has a 'redirect' field. Skipped.")
                continue  # Skip titles with a 'redirect' section

            # Append the valid title to the list
            titles.append(title)
            # Optional: Uncomment the following line to see each added title
            # print(f"Page {idx}: '{title}' added to titles list.")
        else:
            # Optional: Uncomment the following line to see skipped pages without a title
            # print(f"Page {idx}: Missing 'title' field. Skipped.")
            continue

    print(f"\nTotal titles extracted (excluding unwanted prefixes and redirects): {len(titles)}")
    return titles

def main():
    """
    Main function to execute the title extraction.
    """
    # Specify the path to your JSON file
    json_file_path = "cleaned_wikivoyage.json"

    # Extract titles
    titles = load_titles_from_json(json_file_path)

    # Save titles to a text file
    if titles:
        try:
            with open("titles_list.txt", "w", encoding="utf-8") as outfile:
                for title in titles:
                    outfile.write(f"{title}\n")
            print("\nAll titles have been saved to 'titles_list.txt'.")
        except Exception as e:
            print(f"Error: Failed to write to 'titles_list.txt'. {e}")
    else:
        print("No titles were extracted.")

if __name__ == "__main__":
    main()