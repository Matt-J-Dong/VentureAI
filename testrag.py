import tripadvisor
import extract_rag

#CITY_NAME = "Istanbul"


#print("\nModel loaded. Ready for inference.\n")
city_from = input("Enter the city you are traveling FROM (or type 'exit' to quit): ")
city_to = input("Enter the city you are traveling TO (or type 'exit' to quit): ")
num_days = input("Enter the number of days you will be traveling as a single number (or type 'exit' to quit): ")

prompt = f" I would like to travel from {city_from} to {city_to} for {num_days} days. Give me a trip plan that focuses on restaurants. Provide the price and hours of each restaurant."

#tripadvisor.use_city_name()
tripadvisor.main(city_to)
extract_rag.main(city_to)

with open("rag_data.txt", "r", encoding="utf-8") as file:
    rag_data = file.read().strip()


prompt = rag_data + prompt
print(prompt)