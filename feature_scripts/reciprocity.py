from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import nltk
# nltk.download('all')
import os, json
import pandas as pd
import matplotlib.pyplot as plt

import re
import numpy as np

def get_sentiment_tag(sentiment, alpha = 0.5):
    '''
    Returns a string in ['negative', 'neutral', 'positive'] according to the value of the variable sentiment.
    '''
    if abs(sentiment) < alpha:
        return 'neutral'
    return 'positive' if sentiment >= alpha else 'negative'

def reciprocity_defection_list(json_data):
    ''' 
    Returns the average defection point TODO: does average make sense? 
    '''

    sid = SentimentIntensityAnalyzer()

    def process_node(node, parent_sentiment, depth):

        comment_body = node['body']
        sentiment = sid.polarity_scores(comment_body)['compound']
        defection_lengths = []


        # TODO: When done at root node does it compare sentiment with itself?
        
        if sentiment*parent_sentiment < 0: # this detects a defection TODO: hoping sentiment is rarely exactly 0 but I should check stuff. 
            #TODO also, when sentiment is pretty close to 0, whether pos or neg, it seems weird to call it a defefection. 
            # Maybe everything with abs(sentiment) < 0.3 or something should be considered "neutral" and not cause a defection? 
            return [depth]
        
        
        
        # if it hasn't detected a defection yet it should keep going

        for child in node['replies']:

            defection_lengths.extend(process_node(child, sentiment, depth+1))

        return defection_lengths
    
    # renaming these so they work in the recursion (root node has different names for these unfortunately TODO: should fix the scraping with this in mind)
    json_data['body'] = json_data['selftext']
    json_data['replies'] = json_data['comments']
    
    root_sentiment = sid.polarity_scores(json_data['body'])['compound']
    
    defection_lengths = process_node(json_data, root_sentiment, 0)

    return defection_lengths

def reciprocity(comment_forest):

    defection_lengths = reciprocity_defection_list(comment_forest)

    if len(defection_lengths) == 0: 
        return pd.NA

    return sum(defection_lengths)/len(defection_lengths)