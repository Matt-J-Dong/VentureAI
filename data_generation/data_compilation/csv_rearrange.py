import pandas as pd

def rearrange_prompt_with_restaurants(file_path, output_file):
    """
    Rearranges the prompt column to place restaurant details first, followed by the travel query.
    
    Args:
        file_path (str): Path to the input CSV file.
        output_file (str): Path to save the updated CSV file.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Check if required columns exist
        if 'prompt' not in data.columns or 'restaurants' not in data.columns:
            raise ValueError("The input CSV must contain 'prompt' and 'restaurants' columns.")
        
        # Function to rearrange the prompt
        def rearrange_prompt(row):
            try:
                # Extract the city name from the prompt
                prompt_text = row['prompt'].rstrip("\n")  # Remove trailing newline
                city_name_start = prompt_text.find("to ") + 3
                city_name_end = prompt_text.find(" for 7 days")
                city_name = prompt_text[city_name_start:city_name_end]

                # Create the rearranged prompt
                restaurant_info = f"The information below is about {city_name}; use it for your response. Here are the restaurants: {row['restaurants']}"
                travel_query = f"I would like to travel from New York City to {city_name} for 7 days. Give me a trip plan that focuses on restaurants. Provide the price and hours of each restaurant."
                return f"{restaurant_info} {travel_query}"
            except Exception as e:
                print(f"Error processing row: {row}. Error: {e}")
                return row['prompt']  # Return the original prompt if an error occurs

        # Apply the function to update the 'prompt' column
        data['prompt'] = data.apply(rearrange_prompt, axis=1)
        
        # Save the updated DataFrame to a new CSV file
        data.to_csv(output_file, index=False)
        print(f"Updated CSV file saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

# Input and output file paths
input_file = "out_with_price_hours_1.csv"
output_file = "out_with_price_hours_rearranged.csv"

# Call the function
rearrange_prompt_with_restaurants(input_file, output_file)
