
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
VALID_WORDS = set()
for file_name in ["valid_words.txt", "wikitionary_popular_words_40k.txt"]:
    file_path = os.path.join(script_dir, "auxiliary_files", file_name)
    with open(file_path, "r") as file: 
        VALID_WORDS.update(set(word.strip() for word in file.readlines()))

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

def get_comment_readability_old(comment_body):
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

def get_comment_readability(comment_body):

    if len(comment_body.split()) < 3: return pd.NA
    openai.api_key = OPENAI_API_KEY

    prompt = (
        "Compute the Flesch–Kincaid readability score (0–100, higher = easier to read) "
        "for the text below. Ignore any URLs or list markers. "
        f"This is the comment body (surrounded by the symbols '<<>>'): <<{comment_body}>>."
        "Reply with the score only, no explanation, just a single number:"
    )


    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a readability expert."},
                      {"role": "user", "content": prompt}]
    )
    score = response["choices"][0]["message"]["content"]
    try:
        return float(score)
    except:
        return pd.NA


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
    total_readability_score = 0 # This will be averaged over total_readable_comments_count
    total_readable_comments_count = 0
    
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

        tokens = ['www.', '.com', 'http://', 'https://']
        has_links = any(token in comment_body for token in tokens)
        comments_with_links_count += has_links

        # Count spelling mistakes according to our VALID_WORDS set

        paragraph_lower = comment_body.lower()
        tokens = tokenize.word_tokenize(paragraph_lower) # like a .split() except smarter (consdires apostrophes, punctuation, etc. )
        misspellings = [token for token in tokens if token.isalpha() and token not in valid_words]
        
        misspellings_count += len(misspellings)

        # Add readability score for averaging later

        readability_score = get_comment_readability(comment_body)
        if not pd.isna(readability_score): 
            total_readability_score += readability_score 
            total_readable_comments_count +=1 


    
    total_authors = len(authors_set)
    total_comments = len(flattened_comment_forest)
    
    return_dictionary = {
                        "comment_has_author_references_proportion": comments_referencing_other_authors_count / total_comments,
                        "vote_score_mean": total_vote_score / total_comments,
                        "comments_per_author": total_comments / total_authors,
                        "comment_length_mean": total_word_count / total_comments,
                        "comment_has_links_proportion" : comments_with_links_count / total_comments,
                        "misspelled_words_proportion": misspellings_count / total_word_count,
                        "readability_mean": total_readability_score / total_readable_comments_count
                        }

    return return_dictionary