from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cuda")

ARTICLE = """ Please help me plan a trip from St. Petersburg to Rockford spanning 3 days from March 16th to March 18th, 2022. The travel should be planned for a single person with a budget of $1,700.
"""
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
#>>> [{'summary_text': 'Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]
