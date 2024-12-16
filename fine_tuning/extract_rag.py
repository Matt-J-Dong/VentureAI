import json
#from testrag import CITY_NAME
import random


# Function to format an entity
def format_entity(entity, category, city_to):
    hours = ["6:00-20:00", "14:00-24:00", "9:30-16:30", "10:30 - 15:30"]
    details = []  # Collect only the restaurant details
    for name, info in entity.items():
        random_choice = random.choice(hours)
        if category == "restaurants":
            details.append(f"{name} {info.get('price_level', 'No price level specified.')} {random_choice}")
    return f"The information below is about {city_to}; use it for your response. Here are the {category}: {', '.join(details)}"  # Add header with comma-separated details

def main(city_to):
    
    # Load the JSON file
    with open("./locations_info.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract and format data
    categories = ["restaurants"]
    output_lines = []
    for category in categories:
        if category in data[city_to]:
            output_lines.append(format_entity(data[city_to][category], category, city_to))
        else:
            output_lines.append(f"No {category} data available.")

    # Write to a new file
    output_file = "rag_data.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("".join(output_lines))  # Write all data as a single line

    print(f"Formatted data has been written to {output_file}")


if __name__ == "__main__":
    main()



# import json
# from testrag import CITY_NAME
# import random

# # Function to format hours
# def format_hours(hours):
#     if hours:
#         return "\n".join(hours)
#     return "No hours available."

# # Function to format cuisines
# def format_cuisines(cuisines):
#     if cuisines:
#         return " ".join(cuisines)
#     return "No cuisines specified."

# # Function to format an entity
# def format_entity(entity, category):
#     details = [f"Here are the {category}: "]
#     hours = ["6:00-20:00", "14:00-24:00", "9:30-16:30", "10:30 - 15:30"]
#     for name, info in entity.items():
#         random_choice = random.choice(hours)
#         #details.append(f"{name} \\")
#         #details.append(f"Address: {info.get('address', 'No address provided.')}")
#         #details.append(f"Location ID: {info.get('location_id', 'No location ID provided.')}")
#         #details.append(f"Description: {info.get('description', 'No description provided.')}")
#         if category == "restaurants":
#         #     details.append(f"Hours:\n{format_hours(info.get('hours', []))}")
#         #     details.append(f"Cuisine: {format_cuisines(info.get('cuisine', []))}")
#             #details.append(f"Price Level: {info.get('pice_level', 'No price level specified.')} \\")
#             details.append(f"{name} {info.get('price_level', 'No price level specified.')} {random_choice}")
#         # elif category == "hotels":
#         #     details.append(f"Price Level: {info.get('price_level', 'No price level specified.')}")
#         # elif category == "attractions":
#         #     details.append(f"Hours:\n{format_hours(info.get('hours', []))}")
#         #details.append("-" * 40)
#     return " ".join(details)

# def main():
#     # Load the JSON file
#     with open("./locations_info.json", "r", encoding="utf-8") as file:
#         data = json.load(file)

#     # Extract and format data
#     #categories = ["restaurants", "hotels", "attractions"]
#     categories = ["restaurants"]
#     output_lines = []
#     for category in categories:
#         #output_lines.append(f"\n--- {category.upper()} ---\n")
#         if category in data[CITY_NAME]:
#             output_lines.append(format_entity(data[CITY_NAME][category], category))
#         else:
#             output_lines.append(f"No {category} data available.\n")

#     # Write to a new file
#     output_file = "formatted_rag_data.txt"
#     with open(output_file, "w", encoding="utf-8") as file:
#         file.write("".join(output_lines))

#     print(f"Formatted data has been written to {output_file}")


# if __name__ == "__main__":
#     main()

