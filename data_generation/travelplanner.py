from datasets import load_dataset
import pandas as pd

# Load the dataset
train_ds = load_dataset("osunlp/TravelPlanner","train")
val_ds = load_dataset("osunlp/TravelPlanner","validation")
test_ds = load_dataset("osunlp/TravelPlanner","test")

# Convert the 'train' split to DataFrame and print
train_df = pd.DataFrame(train_ds["train"][:])  # Load the entire 'train' split
print("Train Dataset:")
print(train_df)  # Print the first few rows of the train dataset

# Convert the 'val' split to DataFrame and print
val_df = pd.DataFrame(val_ds["validation"][:])  # Load the entire 'val' split
#print("\nValidation Dataset:")
#print(val_df)  # Print the first few rows of the validation dataset

# Convert the 'test' split to DataFrame and print
test_df = pd.DataFrame(test_ds["test"][:])  # Load the entire 'test' split
#print("\nTest Dataset:")
#print(test_df)  # Print the first few rows of the test dataset

print(train_df.columns)
pd.set_option('display.max_colwidth', None)
#print(train_df['query'])
print(train_df['annotated_plan'])
pd.set_option('display.max_colwidth', None)
