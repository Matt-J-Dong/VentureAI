import pandas as pd
import json

# Load the CSV file
csv_file = "perplexity_train_data.csv"  # Replace with your actual file path
data = pd.read_csv(csv_file)
print(f"Number of lines in the CSV file: {len(data)}")

data = data.dropna()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract the relevant columns: prompt and response
output_data = []
c = 0
for _, row in data.iterrows():
    #if c % 17 == 0:
    output_data.append({
        "prompt": row["prompt"],
        "reference": row["response"]
    })
    c += 1
    if c == 50:
        break
    

# Save the extracted data to a JSON file
json_file = "perplexity_data_input.json"  # Replace with your desired output file name
with open(json_file, "w") as file:
    json.dump(output_data, file, indent=4)

print(f"Data successfully extracted and saved to {json_file}.")


