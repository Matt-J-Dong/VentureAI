import json

# Load the JSON data from the file with UTF-8 encoding
with open("./wikivoyage.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Find and print the text associated with the title "Adirondacks"
for page in data.get("mediawiki", {}).get("page", []):
    if page.get("title") == "Adirondacks":
        text = page.get("revision", {}).get("text", "")
        print("Text associated with 'Adirondacks':\n", text)
        break
else:
    print("Section titled 'Adirondacks' not found.")
