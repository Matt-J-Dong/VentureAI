import requests
import json
import os
from dotenv import load_dotenv

# def use_city_name():
#     from testrag import CITY_NAME  # Import inside the function to avoid circular import
#     return CITY_NAME

load_dotenv()
TRIPADVISOR_API_KEY = os.getenv("TRIPADVISOR_API_KEY")
api_key = TRIPADVISOR_API_KEY

def get_location_reviews(locationId, api_key, locations_info, search_query, category, location):
    headers = {"accept": "application/json"}
    url = f"https://api.content.tripadvisor.com/api/v1/location/{locationId}/reviews?key={api_key}"
    response = requests.get(url, headers=headers)
    response = response.json()
    
    # Extract only the title and text fields for each review
    reviews = [
        {
            "title": review.get("title"),
            "review_text": review.get("text")
        }
        for review in response.get("data", [])
    ]

    if search_query in locations_info and category in locations_info[search_query]:
        locations_info[search_query][category][location]["reviews"] = reviews

    return locations_info

def find_search(search_query, category, api_key):
    headers = {"accept": "application/json"}
    url = f"https://api.content.tripadvisor.com/api/v1/location/search?key={api_key}&searchQuery={search_query}&category={category}"
    response = requests.get(url, headers=headers)
    response = response.json()

    # Structure results by search_query and category
    locations_info = {
        search_query: {
            category: {
                entry['name']: {
                    'address': entry['address_obj']['address_string'],
                    'location_id': entry['location_id']
                }
                for entry in response['data']
            }
        }
    }

    return locations_info

def location_details_restaurants(locationId, api_key, locations_info, search_query, category, location):
    headers = {"accept": "application/json"}
    url = f"https://api.content.tripadvisor.com/api/v1/location/{locationId}/details?key={api_key}"
    response = requests.get(url, headers=headers)
    
    try:
        json_response = response.json()

        # Extract required fields
        description = json_response.get('description')
        latitude = json_response.get('latitude')
        longitude = json_response.get('longitude')
        phone = json_response.get('phone')
        rating = json_response.get('rating')
        price_level = json_response.get('price_level')
        weekday_text = json_response.get('hours', {}).get('weekday_text', [])
        
        # Extract cuisine names
        cuisine_list = json_response.get('cuisine', [])
        cuisine_names = [cuisine.get('name') for cuisine in cuisine_list]

        # Update the inner dictionary for the specific location
        if search_query in locations_info and category in locations_info[search_query]:
            locations_info[search_query][category][location].update({
                'description': description,
                #'latitude': latitude,
                #'longitude': longitude,
                #'phone': phone,
                #'rating': rating,
                'hours': weekday_text,
                'cuisine': cuisine_names,
                'price_level': price_level
            })
        
        #locations_info = get_location_reviews(locationId, api_key, locations_info, search_query, category, location)

    except json.JSONDecodeError:
        print("Error decoding JSON from response text.")

    return locations_info

def location_details_hotels(locationId, api_key, locations_info, search_query, category, location):
    headers = {"accept": "application/json"}
    url = f"https://api.content.tripadvisor.com/api/v1/location/{locationId}/details?key={api_key}"
    response = requests.get(url, headers=headers)
    
    try:
        json_response = response.json()

        # Extract required fields for hotels
        description = json_response.get('description')
        address_string = json_response.get('address_obj', {}).get('address_string')
        latitude = json_response.get('latitude')
        longitude = json_response.get('longitude')
        rating = json_response.get('rating')
        price_level = json_response.get('price_level')
        weekday_text = json_response.get('hours', {}).get('weekday_text', [])
        #amenities = json_response.get('amenities', [])

        # Update the inner dictionary for the specific hotel
        if search_query in locations_info and category in locations_info[search_query]:
            locations_info[search_query][category][location].update({
                'description': description,
                #'address_string': address_string,
                #'latitude': latitude,
                #'longitude': longitude,
                #'rating': rating,
                'price_level': price_level,
                #'hours': weekday_text,
                #'amenities': amenities
            })

        #locations_info = get_location_reviews(locationId, api_key, locations_info, search_query, category, location)

    except json.JSONDecodeError:
        print("Error decoding JSON from response text.")

    return locations_info

def location_details_attractions(locationId, api_key, locations_info, search_query, category, location):
    headers = {"accept": "application/json"}
    url = f"https://api.content.tripadvisor.com/api/v1/location/{locationId}/details?key={api_key}"
    response = requests.get(url, headers=headers)
    
    try:
        json_response = response.json()

        # Extract required fields for attractions
        description = json_response.get('description')
        address_string = json_response.get('address_obj', {}).get('address_string')
        latitude = json_response.get('latitude')
        longitude = json_response.get('longitude')
        phone = json_response.get('phone')
        rating = json_response.get('rating')
        weekday_text = json_response.get('hours', {}).get('weekday_text', [])
        groups = json_response.get('groups', [])

        # Update the inner dictionary for the specific attraction
        if search_query in locations_info and category in locations_info[search_query]:
            locations_info[search_query][category][location].update({
                'description': description,
                #'address_string': address_string,
                #'latitude': latitude,
                #'longitude': longitude,
                #'phone': phone,
                #'rating': rating,
                'hours': weekday_text,
                #'groups': groups
            })
        
        #locations_info = get_location_reviews(locationId, api_key, locations_info, search_query, category, location)

    except json.JSONDecodeError:
        print("Error decoding JSON from response text.")

    return locations_info


def location_details_geo(locationId, api_key, locations_info, search_query, category, location):
    headers = {"accept": "application/json"}
    url = f"https://api.content.tripadvisor.com/api/v1/location/{locationId}/details?key={api_key}"
    response = requests.get(url, headers=headers)
    
    try:
        json_response = response.json()

        # Extract required fields for geographic locations
        description = json_response.get('description')
        address_string = json_response.get('address_obj', {}).get('address_string')
        latitude = json_response.get('latitude')
        longitude = json_response.get('longitude')

        # Update the inner dictionary for the specific geographic location
        if search_query in locations_info and category in locations_info[search_query]:
            locations_info[search_query][category][location].update({
                'description': description,
                'address_string': address_string,
                'latitude': latitude,
                'longitude': longitude
            })

        locations_info = get_location_reviews(locationId, api_key, locations_info, search_query, category, location)

    except json.JSONDecodeError:
        print("Error decoding JSON from response text.")

    return locations_info

def get_all_details(search_query, locations_info):
    categories = ['restaurants', 'hotels', 'attractions']

    for category in categories:
        category_info = find_search(search_query, category, api_key)

        locations_info[search_query].update(category_info[search_query])

        for location in locations_info[search_query][category].keys():
            locationId = locations_info[search_query][category][location]['location_id']
            
            if category == 'restaurants':
                locations_info = location_details_restaurants(locationId, api_key, locations_info, search_query, category, location)
            
            elif category == 'hotels':
                locations_info = location_details_hotels(locationId, api_key, locations_info, search_query, category, location)
            elif category == 'attractions':
                locations_info = location_details_attractions(locationId, api_key, locations_info, search_query, category, location)
            # elif category == 'geo':
            #     locations_info = location_details_geo(locationId, api_key, locations_info, search_query, category, location)

    return locations_info

def load_existing_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)  # Load existing JSON data
    return {}


def main(city_to):

    #cities = ['Paris, France', 'Tokyo, Japan', 'London, United Kingdom', 'Madrid, Spain', 'Florence, Spain', 'Kyoto, Japan', 'Istanbul, Turkey']
    # with open('cities.txt', mode='r', encoding='utf-8') as file:
    #     for line in file:
    #         cities.append(line.strip())  

    cities = [city_to]
    #cities = ["Istanbul"]

    # # print(cities)
    existing_data = {}

    for search_query in cities:
        try:
            locations_info = {search_query: {}}
            locations_info = get_all_details(search_query, locations_info)
            existing_data[search_query] = locations_info[search_query]
            print(f"Data for {search_query} processed successfully.")

        except Exception as e:
            # Log the error and continue with the next city
            print(f"Error {e} processing data. TripAvisor API does not support {search_query}")

    with open('locations_info.json', 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    print(f"All data written to locations_info.json")

    

if __name__ == "__main__":
    main()
