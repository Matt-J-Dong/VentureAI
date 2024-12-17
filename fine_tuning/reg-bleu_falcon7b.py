import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json


# Function to calculate BLEU score for a single pair of reference and generated text
def calculate_bleu(reference, generated):
    # Tokenize the sentences
    reference_tokens = [nltk.word_tokenize(reference)]  # Reference must be a list of token lists
    generated_tokens = nltk.word_tokenize(generated)    # Generated output as a token list
    # Apply smoothing function for short texts
    smoothing_function = SmoothingFunction().method4
    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing_function)
    return bleu_score

# TODO just use the perplexity data for now since it already has the generated stuff
with open("perplexity_data_output.json", "r") as file:
    data = json.load(file)
# Calculate BLEU scores for the dataset
bleu_scores = []
for example in data:
    reference = example["reference"]
    generated = example["generated"]
    bleu_score = calculate_bleu(reference, generated)
    bleu_scores.append(bleu_score)
    #print(f"Reference: {reference}")
    #print(f"Generated: {generated}")
    print(f"BLEU Score: {bleu_score:.4f}")
    print("-" * 50)

# Calculate the average BLEU score for the dataset
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")
