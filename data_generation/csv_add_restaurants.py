import pandas as pd

def update_prompt_with_restaurants(file_path, output_file):
    """
    Reads a CSV file with 'prompt', 'response', and 'restaurants' columns.
    Removes the newline at the end of the current 'prompt', appends the additional string, 
    and adds a newline at the end of the additional string.
    
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
        
        # Function to extract city_name and append the string
        def update_prompt(row):
            try:
                # Extract city_name from the prompt
                prompt_text = row['prompt'].rstrip("\n")  # Remove trailing newline
                city_name_start = prompt_text.find("to ") + 3
                city_name_end = prompt_text.find(" for 7 days")
                city_name = prompt_text[city_name_start:city_name_end]
                
                # Create the additional string with city_name and restaurants
                additional_info = f"Provide the price and hours for each restaurant. Here are the restaurants: {row['restaurants']}\n"
                return f"{prompt_text} {additional_info}"
            except Exception as e:
                print(f"Error processing row: {row}. Error: {e}")
                return row['prompt']  # Return the original prompt if an error occurs
        
        # Apply the function to update the 'prompt' column
        data['prompt'] = data.apply(update_prompt, axis=1)
        
        # Save the updated DataFrame to a new CSV file
        data.to_csv(output_file, index=False)
        print(f"Updated CSV file saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

# Input and output file paths
input_file = "output_with_restaurants_cleaned.csv"
output_file = "output_with_restaurants_full.csv"

# Call the function
update_prompt_with_restaurants(input_file, output_file)
