from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def defection_list(json_data, neutral_threshold=0.3):
    ''' 
    For each branch, returns the depth of the first defection point,
    or the length of the branch if no defection point is found.
    Neutral comments (comments with abs(sentiment) < neutral_threshold)
    inherit the parent comment sentiment.
    '''

    sid = SentimentIntensityAnalyzer()

    def process_node(node, parent_sentiment, depth):
        comment_body = node['body']
        raw_sentiment = sid.polarity_scores(comment_body)['compound']
    
        # Inherit parent's sentiment if this comment is neutral.
        if abs(raw_sentiment) < neutral_threshold:
            sentiment = parent_sentiment
        else:
            sentiment = raw_sentiment
        
        # Check for defection (sentiment flip relative to parent's sentiment).
        if sentiment * parent_sentiment < 0:
            return [depth]
        
        # If this node is a leaf, return the depth + 1 (No defection occured, so we return branch length).
        if not node['replies']:
            return [depth + 1]
        
        defection_lengths = []
        for child in node['replies']:
            defection_lengths.extend(process_node(child, sentiment, depth + 1))
        return defection_lengths
    
    # Adjust keys so they match the expected names for recursion.
    json_data['body'] = json_data['selftext']
    json_data['replies'] = json_data['comments']
    
    root_sentiment = sid.polarity_scores(json_data['body'])['compound']
    defection_lengths = process_node(json_data, root_sentiment, 0)
    return defection_lengths

def get_defection_score(comment_forest):

    defection_lengths = defection_list(comment_forest)

    if len(defection_lengths) == 0: 
        return pd.NA

    return sum(defection_lengths)/len(defection_lengths)