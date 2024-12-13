'''
This file reads from combine_results_cleaned.csv and extracts all of the restaurants. Then it creates a new
column in the csv and writes all of the restaurants to that column for the corresponding csv


It also randomly adds $$$ to the end of restaurants
'''

import csv
import re
import random
# Input and output file paths
input_file = "../combined_results.csv"
#input_file = "temp.csv"
output_file = "output_with_restaurants_fixed.csv"

ignore_terms = {'Accommodation', 'Accommodation Options:', 'Accommodation:', 'Activities', 'Activities:', 'Activity', 
'Activity:', 'Additional Tips:', 'Afternoon', 'Afternoon Activity:', 'Afternoon Exploration', 'Afternoon Snack:', 
'Afternoon:', 'Arrival', 'Arrival:', 'Breakfast', 'Breakfast:', 'Brunch', 'Brunch:', 'Budget:', 'Cash Transactions:', 
'Check-out:', 'Continue walking:', 'Culinary Events:', 'Day', 'Day Trip:', 'Departure', 'Departure:', 'Destination:', 
'Dietary Preferences:', 'Dining', 'Dining Reservations:', 'Dinner', 'Dinner (if time allows):', 'Dinner:', 'Duration:', 
'Entry Fees', 'Evening', 'Evening Dinner', 'Evening Flight', 'Evening Free Time', 'Evening:', 'Farewell Dinner:', 
'Finish at:', 'Flight', 'Flight Arrival', 'Flight Back', 'Flight Information', 'Flight:', 'Food Culture:', 'Food Options:', 
'Food/Drink', 'Free Morning', 'Free Time', 'Getting around:', 'Hospitality:', 'Hotel Check-in', 'Hotel Check-in:', 
'Hydration and Snacks:', 'Hydration:', 'Last-minute Shopping', 'Last-minute Shopping:', 'Last-minute shopping', 
'Local Recommendations:', 'Lunch', 'Lunch Stop', 'Lunch Stop:', 'Lunch on Trek:', 'Lunch:', 'Meals', 'Mid-Morning:', 
'Mid-morning:', 'Midday:', 'Morning', 'Morning Activity', 'Morning/Afternoon:', 'Morning:', 'Note', 'Note:', 'Notes:', 
'Onboard Lunch:', 'Optional Activities:', 'Overnight', 'Overnight:', 'Reservations:', 'Respect Local Customs:', 
'Sightseeing', 'Sleep', 'Sleep:', 'Snack:', 'Start at:', 'Starting Point:', 'Transportation', 'Travel', 'Travel Back', 
'Travel:', 'Visit:', 'Transport:', 'Transport','Transportation:', 'Culture:','Transportation', 'Culture','Accommodation Check-In:','Accommodation Check-In',
'Cash:', 'Cash','Culture:','Culture', 'Accommodation Check-in', 'Accommodation Check-in:', 'Check-in', 'Early Breakfast', 'Early Breakfast:',
'Early Morning:', 'Tea Experience:', 'Return to Kidal:', 'Explore Rock Paintings:', 'Water:', 'Safety Considerations:', 'Connecting Flight:', 'Visit Guelta:', 'Food and Drink:', 'Hire a Guide:', 'Early Departure:',
'Last-Minute Lunch:', 'By Train:', 'By Car:', '7-Day Restaurant-Focused Trip Plan:', 'Brightwater:', 'Time',
'09:30', 'Total for Accommodation and Meals', '$11', '02:00', 'Meal Costs Overview', 'LE:EN', 
'detoxification meals', 'silent retreat', 'naturopathic lifestyle', '7-Day Restaurant-Focused Trip Itinerary', 'market',
'Exploration', 'Eat', 'Cultural Considerations', 'Transfer', 'Drink & Snack Options', 'boat trip', 'Relaxation', 'Eat Me',
'Drink Me', 'Café', 'Market', 'restaurant', 'Restaurant', 'Restaurant Suggestions', 'Beach Time', 'Pricing', 'Freshness',
'Sunset Viewing', 'Variety', 'How are you?'




    }

def is_not_restaurant(item):
    # Keywords or patterns to exclude
    exclude_keywords = [
        "path", 'depart', "museum", "market", "visit", "shopping", "tour", "activity", "exploration", "check",
        "pack", "transfer", "festival", "hotel", "snack", "souvenirs", "place", "event", "stop", 
        "transport", "time", "depart", "arrival", "breakfast", "lunch", "dinner", "meal", "eat",
        "farewell", "departure", "arrival", "picnic", "stay", "hostel", "lodging", "café", "road",
        "guesthouse", "cultural", "trip", "day", "moment", "afternoon", "morning", "explore", "relax",
        "wine", "sightseeing", "trek", "hike", "mountain", "excursion", "departure", "museum", "beach", 'shopping', 
        'cultural', 'trip', 'Snack', 'Transfer', 'Drink & Snack Options'
    ]
    # Regex to check for patterns like "[Insert here]" or anything with brackets
    if re.search(r"\[.*?\]", item):
        return True

    
    # Check if any of the exclude keywords are in the item (case-insensitive)
    if any(keyword in item.lower() for keyword in exclude_keywords):
        return True
    
    # If the item doesn't seem like a restaurant, exclude it
    return False


def extract_restaurants(text):
    global ignore_terms
    """Extract restaurant names enclosed in asterisks from the text."""

    matches = re.findall(r"\*\*(.*?)\*\*", text)
    
    # Initialize a list to store removed strings
    for match in matches:
        if match.endswith(":"):  
            ignore_terms.add(match[:-1])  
            ignore_terms.add(match)

    # Output the results
    #print("Filtered Matches (Without Colons):")
    #print(filtered_matches)

    # with open('temp.txt', 'w') as file:
    #     for string in removed_strings:
    #         file.write(string + "\n")
    
    # with open('temp.txt', 'r') as file:
    #     for line in file:
    #         filtered_matches.append(line.strip())
    # print(filtered_matches)

    filtered_matches = [
        match for match in matches 
        if match not in ignore_terms 
        and not re.match(r"Day \d+", match) 
        and not re.match(r"\*Day \d+", match) 
        and not re.search(r"\[.*?\]", match)
        and not re.search(r"\(.*?\)", match)
        and not match.lower().startswith("trip plan")
        and not match.lower().startswith("accommodation options")
        and not match.lower().startswith("accommodation")
        and not match.lower().startswith("activities")
        and not match.lower().startswith("activity")
        and not match.lower().startswith("additional tips")
        and not match.lower().startswith("afternoon")
        and not match.lower().startswith("afternoon activity:")
        and not match.lower().startswith("afternoon exploration")
        and not match.lower().startswith("afternoon snack")
        and not match.lower().startswith("afternoon")
        and not match.lower().startswith("arrival")
        and not match.lower().startswith("breakfast")
        and not match.lower().startswith("brunch")
        and not match.lower().startswith("brunch:")
        and not match.lower().startswith("budget:")
        and not match.lower().startswith("cash transactions:")
        and not match.lower().startswith("check-out:")
        and not match.lower().startswith("continue walking:")
        and not match.lower().startswith("culinary events:")
        and not match.lower().startswith("day")
        and not match.lower().startswith("day trip:")
        and not match.lower().startswith("departure")
        and not match.lower().startswith("departure:")
        and not match.lower().startswith("destination:")
        and not match.lower().startswith("dietary preferences:")
        and not match.lower().startswith("dining")
        and not match.lower().startswith("dining reservations:")
        and not match.lower().startswith("dinner")
        and not match.lower().startswith("dinner (if time allows):")
        and not match.lower().startswith("dinner:")
        and not match.lower().startswith("duration:")
        and not match.lower().startswith("entry fees")
        and not match.lower().startswith("evening")
        and not match.lower().startswith("evening dinner")
        and not match.lower().startswith("evening flight")
        and not match.lower().startswith("evening free time")
        and not match.lower().startswith("evening:")
        and not match.lower().startswith("farewell dinner:")
        and not match.lower().startswith("finish at:")
        and not match.lower().startswith("flight")
        and not match.lower().startswith("flight arrival")
        and not match.lower().startswith("flight back")
        and not match.lower().startswith("flight information")
        and not match.lower().startswith("flight:")
        and not match.lower().startswith("food culture:")
        and not match.lower().startswith("food options:")
        and not match.lower().startswith("food/drink")
        and not match.lower().startswith("free morning")
        and not match.lower().startswith("free time")
        and not match.lower().startswith("getting around:")
        and not match.lower().startswith("hospitality:")
        and not match.lower().startswith("hotel check-in")
        and not match.lower().startswith("hotel check-in:")
        and not match.lower().startswith("hydration and snacks:")
        and not match.lower().startswith("hydration:")
        and not match.lower().startswith("last-minute shopping")
        and not match.lower().startswith("last-minute shopping:")
        and not match.lower().startswith("last-minute shopping")
        and not match.lower().startswith("local recommendations:")
        and not match.lower().startswith("lunch")
        and not match.lower().startswith("lunch stop")
        and not match.lower().startswith("lunch stop:")
        and not match.lower().startswith("lunch on trek:")
        and not match.lower().startswith("lunch:")
        and not match.lower().startswith("meals")
        and not match.lower().startswith("mid-morning:")
        and not match.lower().startswith("mid-morning:")
        and not match.lower().startswith("midday:")
        and not match.lower().startswith("morning")
        and not match.lower().startswith("morning activity")
        and not match.lower().startswith("morning/afternoon:")
        and not match.lower().startswith("morning:")
        and not match.lower().startswith("note")
        and not match.lower().startswith("note:")
        and not match.lower().startswith("notes:")
        and not match.lower().startswith("onboard lunch:")
        and not match.lower().startswith("optional activities:")
        and not match.lower().startswith("overnight")
        and not match.lower().startswith("overnight:")
        and not match.lower().startswith("reservations:")
        and not match.lower().startswith("respect local customs:")
        and not match.lower().startswith("sightseeing")
        and not match.lower().startswith("sleep")
        and not match.lower().startswith("sleep:")
        and not match.lower().startswith("snack:")
        and not match.lower().startswith("start at:")
        and not match.lower().startswith("starting point:")
        and not match.lower().startswith("transportation")
        and not match.lower().startswith("travel")
        and not match.lower().startswith("travel back")
        and not match.lower().startswith("travel:")
        and not match.lower().startswith("visit:")
        and not match.lower().startswith("transport:")
        and not match.lower().startswith("transport")
        and not match.lower().startswith("transportation:")
        and not match.lower().startswith("culture:")
        and not match.lower().startswith("culture")
        and not match.lower().startswith("accommodation check-in:")
        and not match.lower().startswith("accommodation check-in")
        and not match.lower().startswith("check-in")
        and not match.lower().startswith("early breakfast")
        and not match.lower().startswith("early breakfast:")
        and not match.lower().startswith("early morning:")
        and not match.lower().startswith("train to")
        and not match.lower().startswith("train back")
        and not match.lower().startswith("arrival in")
        and not "temple #" in match.lower()
        and not match.lower().startswith("7-day trip plan")
        and not match.lower().startswith("dinner near")
        and not match.lower().startswith("travel back")
        and not match.lower().startswith("ferry to")
        and not match.lower().startswith("ferry from")
        and not match.lower().startswith("local") # careful with this one
        and not match.lower().startswith("flight from")
        and not match.lower().startswith("breakfast in")
        and not match.lower().startswith("lunch in")
        and not match.lower().startswith("dinner in")
        and not match.lower().startswith("dinner after")
        and not match.lower().startswith("lunch after")
        and not match.lower().startswith("breakfast after")
        and not "AM" in match
        and not "PM" in match
        and not match.lower().startswith("tips")
        and not match.lower().startswith("drive")
        and not match.lower().startswith("return to")
        and not match.lower().startswith("additional notes")
        and not match.lower().startswith("booking")
        and not match.lower().startswith("by train")
        and not match.lower().startswith("by car")
        and not match.lower().startswith("by ferry")
        and not match.lower().startswith("explore markets")
        and not match.lower().startswith("train dining")
        and not match.lower().startswith("ferry dining")
        and not match.lower().startswith("last meal")
        and not match.lower().startswith("final meal")
        and not match.lower().startswith("explore")
        and not match.lower().startswith("final lunch")
        and not match.lower().startswith("attending")
        and not match.lower().startswith("night:")
        and not match.lower().startswith("final day")
        and not match.lower().startswith("trip plan")
        and not match.lower().startswith("7-day restaurant-focused trip plan")
        and not match.lower().startswith("brightwater:")
        and not match.lower().startswith("take a")
        and not match.lower().startswith("dietary")
        and not match.lower().startswith("last lunch")
        and not match.lower().startswith("check-out")
        and not match.lower().startswith("check out")
        and not match.lower().startswith("arrive in")
        and not match.lower().startswith("return")
        and not match.lower().startswith("last minute")
        and not match.lower().startswith("7-day trip itinerary")
        and not match.lower().startswith("restaurant in")
        and not match.lower().startswith("optional")
        and not match.lower().startswith("all day")
        and not match.lower().startswith("(note")
        and not match.lower().startswith("bob marley")
        and not match.lower().startswith("late morning")
        and not match.lower().startswith("end of")
        and not match.lower().startswith("fresh:")
        and not match.lower().startswith("depart")
        and not match.lower().startswith("last visit")
        and not "shopping" in match.lower()
        and not match.lower().startswith("option to")
        and not match.lower().startswith("pack and")
        and not match.lower().startswith("check out")
        and not "depart" in match.lower()
        and not "visit" in match.lower()
        and not "stay" in match.lower()
        and not "last day" in match.lower()
        and not "museum" in match.lower()
        and not match.lower().startswith("check hours")
        and not match.lower().startswith("mid-")
        and not match.lower().startswith("snacks")
        and not match.lower().startswith("final")
        and not match.lower().startswith(" breakfast")
        and not match.lower().startswith("transfer to")
        and not match.lower().startswith("transfer from")
        and not match.lower().startswith("last-minute")
        and not match.lower().startswith("7-day")
        and not match.lower().startswith("7 day")
        and not match.lower().startswith("leisurely")
        and not match.lower().startswith("calm")
        and not match.lower().startswith("meal")
        and not match.lower().startswith("check into")
        and not match.lower().startswith("historical tour")
        and not match.lower().startswith("mid")
        and not match.lower().startswith("trip to")
        and not match.lower().startswith("pack")
        and not match.lower().startswith(" for")
        and not match.lower().startswith("last")
        and not match.lower().startswith("after a")
        and not match.lower().startswith("cultural")
        and not match.lower().startswith("free")
        and not match.lower().startswith("a restaurant")
        and not match.lower().startswith("swim at")
        and not match.lower().startswith("spend the")
        and not match.lower().startswith("tour")
        and not match.lower().startswith("pick up")
        and not match.lower().startswith("creative")
        and not match.lower().startswith("fresh juice")
        and not match.lower().startswith("your choice")
        and not match.lower().startswith("transfer")
        and not match.lower().startswith("artificial")
        and not match.lower().startswith("transfer")
        and not match.lower().startswith("picnic at")
        and not match.lower().startswith("please")
        and not match.lower().startswith("your")
        and not match.lower().startswith("at a")
        and not match.lower().startswith("eating at")
        and not match.lower().startswith("*great")
        and not match.lower().startswith("\"the check")
        and not match.lower().startswith("the check")
        and not match.lower().startswith("specific")
        and not match.lower().startswith("diving")
        and not match.lower().startswith("have")
        and not match.lower().startswith("enjoy")
        and not match.lower().startswith("trekking")
        and not match.lower().startswith("picnic")
        and not match.lower().startswith("get")
        and not match.lower().startswith("make")
        and not match.lower().startswith("caution")
        and not match.lower().startswith("lodging")
        and not match.lower().startswith("fly")
        and not match.lower().startswith("check")
        and not match.lower().startswith("be mindful")
        and not match.lower().startswith("relaxation")
        and not match.lower().startswith("logistics")
        and not match.lower().startswith("walking")
        and not match.lower().startswith("health")
        and not match.lower().startswith("try")
        and not match.lower().startswith("be open")
        and not match.lower().startswith("be careful")
        and not match.lower().startswith("ferry")
        and not match.lower().startswith("budget")
        and not match.lower().startswith("boat")
        and not match.lower().startswith("respect")
        and not match.lower().startswith("continue")
        and not match.lower().startswith("avoid")
        and not match.lower().startswith("chill")
        and not match.lower().startswith("weather")
        and not match.lower().startswith("airport")
        and not match.lower().startswith("fees")
        and not match.lower().startswith("bonus")
        and not match.lower().startswith("splurge")
        and not match.lower().startswith("contact")
        and not match.lower().startswith("arriving")
        and not match.lower().startswith("trek")
        

    ]
    filtered_matches = [
        match for match in filtered_matches
        if not match.isupper()
        if not match.lower().startswith("trek")
        and not match.lower().startswith("after")
        and not match.lower().startswith("late")
        and not match.lower().startswith("early")
        and not match.lower().startswith("dine")
        and not match.lower().startswith("description")
        and not match.lower().startswith("tip")
        and not match.lower().startswith("ask")
        and not match.lower().startswith("+")
        and not match.lower().startswith("dive")
        and not match.lower().startswith("hiking")
        and not match.lower().startswith("hike")
        and not match.lower().startswith("casual")
        and not match.lower().startswith("venture")
        and not match.lower().startswith("journey")
        and not match.lower().startswith("in-flight")
        and not match.lower().startswith("in flight")
        and not match.lower().startswith("kayaking")
        and not match.lower().startswith("area")
        and not match.lower().startswith("refreshments")
        and not match.lower().startswith("reccomended")
        and not match.lower().startswith("outdoor")
        and not match.lower().startswith("guided")
        and not match.lower().startswith("tour")
        and not match.lower().startswith("consume")
        and not match.lower().startswith("sourcing")
        and not match.lower().startswith("post")
        and not match.lower().startswith("ski")
        and not match.lower().startswith("sampling")
        and not match.lower().startswith("beverages")
        and not match.lower().startswith("noon")
        and not match.lower().startswith("marathon")
        and not match.lower().startswith("cuisine")
        and not match.lower().startswith("permits")
        and not match.lower().startswith("eat in")
        and not match.lower().startswith("food safety")
        and not match.lower().startswith("photo")
        and not "afternoon" in match.lower()
        and not "activity" in match.lower()
        and not "leisure" in match.lower()
        and not match.lower().startswith("session")
        and not match.lower().startswith("reservation")
        and not match.lower().startswith("leisure")
        and not match.lower().startswith("all-day")
        and not match.lower().startswith("downtime")
        and not match.lower().startswith("seek")
        and not match.lower().startswith("amtrak")
        
        

    ]
    #print(filtered_matches)

    
    prefixes = ["Dinner at ", "Breakfast at ", "Lunch at ", "Dine at", "Restaurant:", "Restaurant Suggestion:", 
    '*Restaurant', 'Restaurant Recommendation:', 'Brewery:', 'Food Tour:', 'S:t', 'Final']

# Remove prefixes
    filtered_matches = [
        string if not any(string.startswith(prefix) for prefix in prefixes) 
        else string.split(max(prefixes, key=lambda p: string.startswith(p)))[-1].strip()
        for string in filtered_matches
    ]
    filtered_matches = set(filtered_matches)
    filtered_matches = list(filtered_matches)

    #print(filtered_matches)

    #filtered_matches = [item for item in filtered_matches if is_not_restaurant(item)]
    #print(non_restaurant_items)

    #print(filtered_matches)

    
    return filtered_matches

# Read the input CSV, process each row, and add a new column
rows = []
with open(input_file, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["restaurants"]  # Add new column name
    for row in reader:
        response = row["response"]
        # Extract restaurant names
        restaurants = extract_restaurants(response)
        #print(restaurants)
        # Join the restaurant names into a single string for the new column
        row["restaurants"] = ", ".join(restaurants)
        rows.append(row)

for row in rows:
    for key, value in row.items():
        if isinstance(value, str):  # Check if the value is a string
            row[key] = value.replace("*", "")
            

price_tags = ["$", "$$", "$$$", "$$$$", "$ - $$", "$$ - $$$", "$$$ - $$$$"]
# Randomly assign price tags to each restaurant
for row in rows:
    if "restaurants" in row and row["restaurants"]:  # Check if the restaurants column exists and is not empty
        tagged_restaurants = []
        # Split restaurants into a list, assign a random price tag to each, and join them back into a single string
        restaurants = row["restaurants"].split(", ")
        #random_choice = random.choice(price_tags)
        responses = row["response"]
        for restaurant in restaurants:
            random_choice = random.choice(price_tags)
            responses = responses.replace(f"{restaurant}", f"{restaurant} {random_choice}") 
            tagged_restaurants.append(f"{restaurant} {random_choice}")
        #tagged_restaurants = [f"{restaurant} {random_choice}" for restaurant in restaurants]
        row["restaurants"] = ", ".join(tagged_restaurants)
        row["response"] = responses

with open("restaurants.txt", "w", encoding="utf-8") as file:
    # Write all extracted restaurants to the file
    for row in rows:
        restaurant_text = row["restaurants"]
        if restaurant_text:
            #if len(restaurant_text.split()) < 7:  # Check if less than 7 words
                file.write(restaurant_text + "\n")

# Write the updated rows to a new CSV file
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Processed CSV has been saved to {output_file}")
