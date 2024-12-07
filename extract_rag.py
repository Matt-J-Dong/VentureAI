import json

# Load the JSON file
with open("./locations_info.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Function to format hours
def format_hours(hours):
    if hours:
        return "\n".join(hours)
    return "No hours available."

# Function to format cuisines
def format_cuisines(cuisines):
    if cuisines:
        return ", ".join(cuisines)
    return "No cuisines specified."

# Function to format an entity
def format_entity(entity, category):
    details = [f"Here are the {category}:\n"]
    for name, info in entity.items():
        details.append(f"{name}")
        #details.append(f"Address: {info.get('address', 'No address provided.')}")
        #details.append(f"Location ID: {info.get('location_id', 'No location ID provided.')}")
        #details.append(f"Description: {info.get('description', 'No description provided.')}")
        # if category == "restaurants":
        #     details.append(f"Hours:\n{format_hours(info.get('hours', []))}")
        #     details.append(f"Cuisine: {format_cuisines(info.get('cuisine', []))}")
        #     details.append(f"Price Level: {info.get('pice_level', 'No price level specified.')}")
        # elif category == "hotels":
        #     details.append(f"Price Level: {info.get('price_level', 'No price level specified.')}")
        # elif category == "attractions":
        #     details.append(f"Hours:\n{format_hours(info.get('hours', []))}")
        #details.append("-" * 40)
    return "\n".join(details)

# Extract and format data
#categories = ["restaurants", "hotels", "attractions"]
categories = ["restaurants"]
output_lines = []
for category in categories:
    #output_lines.append(f"\n--- {category.upper()} ---\n")
    if category in data["Istanbul, Turkey"]:
        output_lines.append(format_entity(data["Istanbul, Turkey"][category], category))
    else:
        output_lines.append(f"No {category} data available.\n")

# Write to a new file
output_file = "formatted_rag_data.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(output_lines))

print(f"Formatted data has been written to {output_file}")