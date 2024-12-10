import pandas as pd

def remove_asterisks_from_csv(input_file, output_file):
    """
    Removes all occurrences of '*' from a CSV file and saves the cleaned data to a new file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Replace all occurrences of '*' in the DataFrame with an empty string
        df = df.replace({r'\*': '', r'\#\#\# ': '', r'\#': ''}, regex=True)

        # Save the cleaned DataFrame to a new file
        df.to_csv(output_file, index=False)

        print(f"Cleaned CSV saved to {output_file}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Usage example
input_file = "output_with_restaurants.csv"  # Replace with the path to your input file
output_file = "output_with_restaurants_cleaned.csv"  # Replace with the desired output file path

remove_asterisks_from_csv(input_file, output_file)
