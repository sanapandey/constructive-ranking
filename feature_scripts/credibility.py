
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

# AUTHOR INVESTMENT

# auxiliary functions for calculating author investment 

def get_comment_length(comment_body):
    # returns number of words in comment
    return len(comment_body.split())

def get_comment_has_links(comment_body):

    #TODO hyperlinks hidden in text, API gets them??
    #TODO this approach looking for key tokens is too naive?

    tokens = ['www.', '.com', 'http://', 'http://']

    return any(token in comment_body for token in tokens)

def get_n_spelling_mistakes(comment_body, valid_words):

    paragraph_lower = comment_body.lower()
    tokens = tokenize.word_tokenize(paragraph_lower) # like a .split() except smarter (consdires apostrophes, punctuation, etc. )
    
    misspellings = [token for token in tokens if token.isalpha() and token not in valid_words]
    
    return len(misspellings)

def get_comment_readability(comment_body):
    openai.api_key = OPENAI_API_KEY

    prompt = f"Rate the following text on the Flesch-Kincaid readability index (0-100, higher is easier to read). For texts that are one letter, return a score of 5. Please only return numbers with no text. Otherwise, apply the formula as standard:\n\n{comment_body}\n\nReadability Score:"

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a readability expert."},
                      {"role": "user", "content": prompt}]
    )
    score = response["choices"][0]["message"]["content"]
    return float(score)  # Ensure it's a float

def get_investment_values_thread_average(comment_forest, valid_words):

    ''' 
    returns dicionary = {
        'average length' : thread_total_comment_length/n_comments,
        'average mistakes' : thread_total_spelling_mistakes/n_comments,
        'average links' : thread_total_links/n_comments,
        'score' : average score
        'average readability': thread_total_readability/n_comments
    }
    '''

    queue = comment_forest['comments'].copy()

    assert len(queue) > 0, 'Empty comment_forest.'

    n_comments = 0

    thread_total_comment_length = 0
    thread_total_spelling_mistakes = 0
    thread_total_links = 0
    thread_total_readability = 0
    thread_total_score = 0

    while queue:

        n_comments +=1 
        comment = queue.pop()

        queue.extend(comment['replies'])
        comment_body = comment['body']

        thread_total_comment_length += get_comment_length(comment_body)
        thread_total_spelling_mistakes += get_n_spelling_mistakes(comment_body, valid_words=valid_words)
        thread_total_links += get_comment_has_links(comment_body)
        thread_total_readability += get_credibility_score(comment_body)
        thread_total_score += comment['score']

    dictionary = {
        'length' : thread_total_comment_length/n_comments,
        'mistakes' : thread_total_spelling_mistakes/n_comments,
        'links' : thread_total_links/n_comments,
        'readability': thread_total_readability/n_comments,
        'vote score' :thread_total_score/n_comments
    }

    return dictionary

def get_author_comments_dictionaries(comment_forest):

    '''
    returns dictionary whose keys are author names and values are lists of all comments made by that author.
    (in the return object described above a "comment" is a dictionary with keys 'body' and 'vote score'.) 
    '''

    queue = comment_forest['comments'].copy()

    assert len(queue) > 0, 'Empty comment_forest.'

    dictionary = {}

    while queue:

        comment = queue.pop()
        queue.extend(comment['replies'])
        author = comment['author']
        
        new_dictionary = {'body': comment['body'], 
                          'vote score': comment['score']}

        if author not in dictionary: 
            dictionary[author] = [new_dictionary]
        
        else:
            dictionary[author].append(new_dictionary)

    return dictionary 

def get_author_investment(author_comments, investment_values_thread_average, valid_words):


    author_length_score = np.mean([get_comment_length(comment_body=comment['body']) for comment in author_comments])
    author_link_score = np.mean([get_comment_has_links(comment_body=comment['body']) for comment in author_comments])
    author_mistakes_score = np.mean([get_n_spelling_mistakes(comment_body=comment['body'], valid_words=valid_words) for comment in author_comments])
    author_readability_score = np.mean([get_comment_readability(comment_body=comment['body']) for comment in author_comments])

    
    #update values with weights

    #thread_average values
    avg_length = investment_values_thread_average['length']
    avg_links = investment_values_thread_average['links'] 
    avg_mistakes = investment_values_thread_average['mistakes']
    avg_readability = investment_values_thread_average['readability']


    # normalize author score values by thread averages

    author_length_score = author_length_score/avg_length
    author_link_score = author_link_score/avg_links if avg_links else 0
    author_mistakes_score = author_mistakes_score/avg_mistakes
    author_readability_score = author_readability_score/avg_readability

    author_investment = author_length_score + author_link_score + author_readability_score - author_mistakes_score 

    return author_investment 

def create_author_investment_df(comment_forest, valid_words):

    author_comments_dictionaries = get_author_comments_dictionaries(comment_forest=comment_forest)

    rows = []

    for author, author_comments in author_comments_dictionaries.items():

        author_length_score = np.mean([get_comment_length(comment_body=comment['body']) for comment in author_comments])
        author_link_score = np.mean([get_comment_has_links(comment_body=comment['body']) for comment in author_comments])
        author_mistakes_score = np.mean([get_n_spelling_mistakes(comment_body=comment['body'], valid_words=valid_words) for comment in author_comments])
        author_readability_score = np.mean([get_comment_readability(comment_body=comment['body']) for comment in author_comments])
        author_total_comments = len(author_comments)

        new_row = {'author': author,
                   'total comments' : author_total_comments,
                   'average_length': author_length_score,
                   'average_links': author_link_score,
                   'author_mistakes': author_mistakes_score,
                   'author_readability': author_readability_score
                   }
        rows.append(new_row)

    df = pd.DataFrame(rows)

    # normalize by thread averages 

    df['average_length_normalized'] = (df['average_length'] - df['average_length'].mean()) / df['average_length'].std()
    df['average_links_normalized'] = (df['average_links'] - df['average_links'].mean()) / df['average_links'].std() if df['average_links'].std() else 0
    df['author_mistakes_normalized'] = (df['author_mistakes'] - df['author_mistakes'].mean()) / df['author_mistakes'].std() 
    df['author_readability_normalized'] = (df['author_readability'] - df['author_readability'].mean()) / df['author_readability'].std() 
    df['investment_score'] = df['average_length_normalized'] + df['average_links_normalized'] + df['author_readability_normalized'] - df['author_mistakes_normalized']

    return df

# AUTHOR REPUTATION

def extract_references_from_comment(comment_body):

   references = re.findall(r'u/([^/,.!? ]+)', comment_body) 

   # TODO could be improved by also checking that 
   # 1. it is referencing an author actually in the thread
   # 2. it is not a self reference. 
   # (although I expect both to be hold almost always).

   return references

def create_author_reputation_df(comment_forest):
   '''
   returns dictionaries references_others, is_referenced and average_vote_score
   keys of these dictionaries are authors names and 
   references_others[author] = "how many times did this author references another author" #TODO maybe the actual list instead of the number would be good for testing correctness?
   is_referenced[author] = "how many times was this author referenced by another" #TODO maybe the actual list instead of the number would be good for testing?
   average_vote_score[author] = this author's average vote score
   ''' 

   queue = comment_forest['comments'].copy()

   references_others = {}
   is_referenced = {}

   # Auxiliary dictionaries to calculate average vote score
   total_vote_score = {}
   total_comments = {}

   while queue:

      comment = queue.pop()
      queue.extend(comment['replies'])

      comment_body = comment['body']
      comment_author = comment['author']
      comment_vote_score  = int(comment['score'])
      references_in_comment = extract_references_from_comment(comment_body=comment_body)

      total_vote_score[comment_author] = total_vote_score.get(comment_author, 0) + comment_vote_score
      total_comments[comment_author] = total_comments.get(comment_author, 0) + 1

      for referenced_author in references_in_comment:

         is_referenced[referenced_author] = is_referenced.get(referenced_author, 0) + 1
         references_others[comment_author] = references_others.get(comment_author, 0) + 1

   # tidy up info in a dataframe
   rows = []
   
   for author in total_vote_score.keys():
      new_row = {'author' : author,
                 'total_comments' : total_comments[author],
                 'total_vote_score' : total_vote_score[author], 
                 'is_referenced' : is_referenced.get(author, 0),
                 'references_others' : references_others.get(author, 0),
                 'average_vote_score' : total_vote_score[author]/total_comments[author]
                 }
      rows.append(new_row)

   df = pd.DataFrame(rows)

   df['is_referenced_normalized'] = (df['is_referenced'] - df['is_referenced'].mean()) / df['is_referenced'].std() if df['is_referenced'].std() else 0
   df['references_others_normalized'] = (df['references_others'] - df['references_others'].mean()) / df['references_others'].std() if df['references_others'].std() else 0
   df['vote_score_normalized'] = (df['average_vote_score'] - df['average_vote_score'].mean()) / df['average_vote_score'].std() if df['average_vote_score'].std() else 0

   df['reputation_score'] = df['is_referenced_normalized'] + df['references_others_normalized'] + df['vote_score_normalized']

   return df

# MAIN FUNCTION 

def get_credibility_score(comment_forest, valid_words = VALID_WORDS): 

    if len(comment_forest['comments']) < 2:  #if len(comment_forest['comments']) < 2: 
        return pd.NA

    reputation_df = create_author_reputation_df(comment_forest)
    investment_df = create_author_investment_df(comment_forest, valid_words)

    display(reputation_df)
    display(investment_df)

    mean_reputation = reputation_df['reputation_score'].mean()
    mean_investment = investment_df['investment_score'].mean()

    credibility_score = mean_investment + mean_reputation

    return credibility_score
