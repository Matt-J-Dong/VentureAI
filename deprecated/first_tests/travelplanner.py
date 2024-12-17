from datasets import load_dataset
import pandas as pd

# Load the dataset
ds = load_dataset("osunlp/TravelPlanner", "train")

df = pd.DataFrame(ds["train"][:])  # Convert entire 'train' split to DataFrame
print(df.head())  # Show the first few rows of the DataFrame
