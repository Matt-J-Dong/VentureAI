import json

# Load JSON data
with open('cleaned_wikivoyage.json') as f:
    data = json.load(f)

city_names = set()

# Extract city names (titles) from the JSON
for page in data['mediawiki']['page']:
    title = page.get('title')
    if title:  # Ensure title is not None
        city_names.add(title)

with open('city_names.json', 'w') as json_file:
    json.dump(list(city_names), json_file)


# # Create a dictionary to map titles to their text
# title_to_text = {}
# for page in data['mediawiki']['page']:
#     title = page.get('title')
#     text = page.get('revision', {}).get('text', {}).get('#text', '')
#     title_to_text[title] = text

# # Open a file to write the output
# with open('city_descriptions.txt', 'w') as output_file:
#     for page in data['mediawiki']['page']:
#         title = page.get('title')
#         redirect = page.get('redirect', {}).get('@title')
#         text = page.get('revision', {}).get('text', {}).get('#text', '')

#         # Write title
#         output_file.write(f"Title: {title}\n")

#         # If it's a redirect, fetch the text of the target title
#         if redirect:
#             redirect_text = title_to_text.get(redirect, "Text not available")
#             output_file.write(f"Text from {redirect}: {redirect_text}\n")
#         else:
#             output_file.write(f"Text: {text}\n")

#         # Add a newline between entries for readability
#         output_file.write("\n")
