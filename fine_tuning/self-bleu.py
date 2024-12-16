import warnings
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

# Suppress BLEU score warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")

# Function to calculate Self-BLEU
def calculate_self_bleu(generated_outputs, weights=(0.25, 0.25, 0.25, 0.25)):
    scores = []
    smoothing = SmoothingFunction().method1  # Apply smoothing
    # Pairwise comparisons between all outputs
    for i, output in enumerate(generated_outputs):
        references = [nltk.word_tokenize(o) for j, o in enumerate(generated_outputs) if i != j]
        hypothesis = nltk.word_tokenize(output)
        # Calculate BLEU score with smoothing
        bleu_score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing)
        scores.append(bleu_score)
    
    # Return the average Self-BLEU score
    return sum(scores) / len(scores)

# # Example outputs from the model for the same prompt
# generated_outputs = [
#     "Day 1: Visit 360 Istanbul for lunch and explore the Bosphorus area.",
#     "Day 1: Enjoy lunch at Zuma Istanbul and walk along the Bosphorus.",
#     "Day 1: Visit 360 Istanbul for lunch and explore the Bosphorus area.",
#     "Day 1: Visit 360 Istanbul for lunch and explore the Bosphorus area.",
#     "Day 1: Have a meal at Istanbul Anatolian Cuisine and relax by the Bosphorus.",
#     "Day 1: Try Istanbul Kebab Cafe and take a stroll along the Bosphorus shore."
# ]

with open('self-bleu_data_output.json', 'r') as file:
    output_data = json.load(file)

# Extract the `generated` field for all entries
generated_outputs = [entry["generated"] for entry in output_data]

for i, output in enumerate(generated_outputs):
    # Calculate BLEU for 4-grams (default BLEU-4)
    self_bleu_score_bleu4 = calculate_self_bleu(generated_outputs)
    print(f"Self-BLEU Score (BLEU-4) for prompt {i}: {self_bleu_score_bleu4:.4f}")

    # Calculate BLEU for 2-grams (BLEU-2)
    self_bleu_score_bleu2 = calculate_self_bleu(generated_outputs, weights=(0.5, 0.5, 0.0, 0.0))
    print(f"Self-BLEU Score (BLEU-2) for prompt {i}: {self_bleu_score_bleu2:.4f}")
