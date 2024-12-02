import pandas as pd

# File paths for the CSV files
file1 = 'results.csv'
file2 = 'results2.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_file = 'combined_results.csv'
combined_df.to_csv(combined_file, index=False)

print(f"Combined file saved as {combined_file}")
