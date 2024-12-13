import csv
import random
'''
Iterate through each row in the csv.
    - extract the restaurants column (Each item will be: [Restaurant $$$])
    - for the prompt replace the extact string with [Restaurant, $$$, 6:00-22:00] [Restaurant (Price: $$$) (Hours: 6:00-22:00)]

'''

input_file = 'output_with_restaurants_in_prompt.csv'
output_file = 'out_with_price_hours_2.csv'

rows = []
with open(input_file, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    for row in reader:
        response = row["response"]
        # Extract restaurant names
        #row["restaurants"] = ", ".join(restaurants)
        rows.append(row)


hours = ["6:00-20:00", "14:00-24:00", "9:30-16:30", "10:30 - 15:30"]


# Randomly assign price tags to each restaurant
for row in rows:
    if "restaurants" in row and row["restaurants"]:  # Check if the restaurants column exists and is not empty
        tagged_restaurants = []
        # Split restaurants into a list, assign a random price tag to each, and join them back into a single string
        restaurants = row["restaurants"].split(", ")
       
        responses = row["response"]
        prompts = row["prompt"]
        for restaurant in restaurants:
            random_choice = random.choice(hours)
            
            # responses = responses.replace(f"{restaurant}", f"{restaurant} {f'(Hours: {random_choice})'}") 
            # #print(restaurant)
            # prompts = prompts.replace(f"{restaurant}", f"{restaurant} {f'(Hours: {random_choice})'}") 
            # tagged_restaurants.append(f"{restaurant} {f'(Hours: {random_choice})'}")
            responses = responses.replace(f"{restaurant}", f"{restaurant} {random_choice}") 
            #print(restaurant)
            prompts = prompts.replace(f"{restaurant}", f"{restaurant} {random_choice}") 
            tagged_restaurants.append(f"{restaurant} {random_choice}")
            
        #print(responses)
        #tagged_restaurants = [f"{restaurant} {random_choice}" for restaurant in restaurants]
        row["restaurants"] = ", ".join(tagged_restaurants)
        row["response"] = responses
        row["prompt"] = prompts


# Write the updated rows to a new CSV file
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)