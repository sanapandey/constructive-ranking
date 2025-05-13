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

def get_comment_readability(comment_body):
    
    openai.api_key = OPENAI_API_KEY

    # prompt = (
    #     "Compute the Flesch-Kincaid readability score (0-100, higher = easier to read) "
    #     "for the text below, which is surrounded by the symbols '<<' and '>>'. "
    #     "Ignore any URLs or list markers."
    #     "To calculate the score count sentences and syllables precisely "
    #     "and apply the formula: "
    #     "206.835 - 1.015*(words/sentences) - 84.6*(syllables/words). "
    #     f"This is the text: <<{comment_body}>>."
    #     "Reply with the score only, no explanation, just a single number:"
    # )

    prompt = ("Read the text below. Then, indicate the readability of the text, on a scale from 1 (extremely challenging to understand) to 100 (very easy to read and understand). In your assessment, consider factors such as sentence structure, vocabulary complexity, and overall clarity."
    f"This is the text: <<{comment_body}>>."
    "It is extremely that you reply with the score only, no explanation, just a single number:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", 
                   "content": "You are a readibility expert who can only respond in numbers."},
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

    total_comments = 0
    
    for comment in flattened_comment_forest:

        comment_body = comment['body']

        if comment_body == "[removed]" or comment_body == "[deleted]":
            continue

        total_comments += 1

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
    
    return_dictionary = {
                        "comment_has_author_references_proportion": comments_referencing_other_authors_count / total_comments if total_comments else pd.NA,
                        "vote_score_mean": total_vote_score / total_comments if total_comments else pd.NA,
                        "comments_per_author": total_comments / total_authors if total_authors else pd.NA,
                        "comment_length_mean": total_word_count / total_comments if total_comments else pd.NA,
                        "comment_has_links_proportion" : comments_with_links_count / total_comments if total_comments else pd.NA,
                        "misspelled_words_proportion": misspellings_count / total_word_count if total_word_count else pd.NA,
                        "readability_mean": total_readability_score / total_readable_comments_count if total_readable_comments_count else pd.NA,
                        # some extras for debuggin
                        "total_word_count" : total_word_count,
                        "total_comments" : total_comments,
                        "total_coments_readability_scorable" : total_readable_comments_count
                        }

    return return_dictionary