
import nltk
nltk.download('punkt_tab')
from nltk import tokenize

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import openai
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create VALID_WORDS word set from auxiliary files. 
# wikitionary_popular_words_40k.txt downloaded from https://github.com/dolph/dictionary/blob/master/popular.txt
# valid_words.txt downloaded from https://github.com/dwyl/english-words/blob/master/words.txt

script_dir = os.path.dirname(os.path.abspath(__file__))
for file_name in ["valid_words.txt", "wikitionary_popular_words_40k.txt"]:
    file_path = os.path.join(script_dir, "auxiliary_files", file_name)

    with open(file_path, "r") as file: 
        VALID_WORDS = set(word.strip() for word in file.readlines())

def flatten_comments(comment_forest):
    flattened = []

    def traverse(comment):

        flattened.append({
                "author": comment["author"],
                "body": comment["body"],
                "score": comment["score"]
            })

        # If there are nested replies, traverse them too
        if "replies" in comment and comment["replies"]:
            for reply in comment["replies"]:
                traverse(reply)

    # Start traversing from the top-level "comments" list if it exists
    if "comments" in comment_forest:
        for comment in comment_forest["comments"]:
            traverse(comment)
    return flattened

def get_comment_readability(comment_body):
    openai.api_key = OPENAI_API_KEY

    # prompt = f"Rate the following text on the Flesch-Kincaid readability index (0-100, higher is easier to read). There is one special case: for texts that are one letter or unable to be analyzed by the Flesch Kincaid algorithm, return a score of 5. For all other cases, please apply the formula as standard. Please only return numbers with no text. It is very important that you return only numbers with no additional context or explanation:\n\n{comment_body}\n\nReadability Score:"
    prompt = (
    "Rate the following text on the Flesch-Kincaid readability index (0-100, higher is easier to read). "
    "There is one special case: for texts that are one letter or unable to be analyzed by the Flesch-Kincaid algorithm, return a score of 5. "
    "For all other cases, please apply the formula as standard. "
    "Please only return **a single number** (with no text, explanation, or multiple values):\n\n"
    f"{comment_body}\n\nReadability Score:"
    )


    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a readability expert."},
                      {"role": "user", "content": prompt}]
    )
    score = response["choices"][0]["message"]["content"]
    
    return float(score)  # Ensure it's a float

def get_credibility_subfeatures(comment_forest, valid_words = VALID_WORDS):

    flattened_comment_forest = flatten_comments(comment_forest)

    # initialize subfeatures for aggregation 
    # reputation
    comments_referencing_other_authors_count = 0
    total_vote_score = 0
    authors_set = set() # This will be used to count how many authors in the thread
    # investment 
    total_word_count = 0
    comments_with_links_count = 0
    misspellings_count = 0
    total_readability_score = 0 # This will be averaged over total_comments_count
    
    for comment in flattened_comment_forest:

        comment_body = comment['body']

        authors_set.add(comment['author'])

        # Reputation subfeatures

        # Verify if comment includes references to others

        has_reference = re.search(r'u/([^/,.!? ]+)', comment_body) is not None
        comments_referencing_other_authors_count += has_reference

        # Add vote score to average later

        total_vote_score += comment['score']

        # Investment subfeatures

        # Count words in each comment to later get average comment length.

        total_word_count += len(comment_body.split())

        # Verify if comment includes links

        tokens = ['www.', '.com', 'http://', 'http://']
        has_links = any(token in comment_body for token in tokens)
        comments_with_links_count += has_links

        # Count spelling mistakes according to our VALID_WORDS set

        paragraph_lower = comment_body.lower()
        tokens = tokenize.word_tokenize(paragraph_lower) # like a .split() except smarter (consdires apostrophes, punctuation, etc. )
        misspellings = [token for token in tokens if token.isalpha() and token not in valid_words]
        
        misspellings_count += len(misspellings)

        # Add readability score for averaging later

        total_readability_score += get_comment_readability(comment_body)

    # # reputation  
    # 	get_author_references() # esto encripta is_references + references_others
    # 	get_average_vote_score()
    # 	# investment
    # 	get_average_comments_per_author # simplemente total comments sobre total authors
    # 	get_average_comment_length
    # 	get_average_links_per_comment
    # 	get_average_mistakes_per_comment
    # 	get_average_comment_readability

    
    total_authors = len(authors_set)
    total_comments = len(flattened_comment_forest)
    
    return_dictionary = {
                        "comment_has_author_references_percentage": comments_referencing_other_authors_count / total_comments,
                        "vote_score_mean": total_vote_score / total_comments,
                        "comments_per_author": total_comments / total_authors,
                        "comment_length_mean": total_word_count / total_comments,
                        "comment_has_links_percentage" : comments_with_links_count / total_comments,
                        "misspellings_mean": misspellings_count / total_word_count,
                        "readability_mean": total_readability_score / total_comments
                        }

    return return_dictionary