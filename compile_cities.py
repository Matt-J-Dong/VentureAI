'''
This file read from the worldcities.csv and compiles a txt of all the cities names

'''


import csv

# Initialize an empty list to store the cities
cities = []

# Open the CSV file
with open('worldcities.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)  # Use DictReader to read the rows as dictionaries
    for row in reader:
        cities.append(row['city'])  # Append the value of the 'city' field

cities = sorted(set(cities))

with open('cities.txt', mode='w', encoding='utf-8') as file:
    for city in cities:
        file.write(city + '\n')  # Write each city on a new line
