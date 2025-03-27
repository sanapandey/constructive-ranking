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

def get_defection_score_legacy(comment_forest):

    defection_lengths = defection_list(comment_forest)

    if len(defection_lengths) == 0: 
        return pd.NA

    return sum(defection_lengths)/len(defection_lengths)

# like the legacy version but normalized by branch lengths``

def get_defection_score(comment_forest, neutral_threshold=0.3):
    
    """
    For each branch, computes a normalized defection score between 0 and 1.
    A branch with no defection returns a score of 1, while a branch with an 
    early defection gets a score close to 0. The function returns the average 
    score over all branches.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Adjust keys to match expected names for recursion.
    comment_forest['body'] = comment_forest['selftext']
    comment_forest['replies'] = comment_forest['comments']

    def process_branch(node, parent_sentiment, depth, defection_depth):
        # Compute the sentiment for this node.
        comment_body = node['body']
        raw_sentiment = sid.polarity_scores(comment_body)['compound']
        
        # Inherit parent's sentiment if the comment is neutral.
        if abs(raw_sentiment) < neutral_threshold:
            sentiment = parent_sentiment
        else:
            sentiment = raw_sentiment

        # If we haven't yet seen a defection on this branch and
        # the sentiment flips relative to the parent's sentiment,
        # record the defection at the current depth.
        if defection_depth is None and sentiment * parent_sentiment < 0:
            defection_depth = depth

        # If this node is a leaf, compute the branch length.
        # The branch length is defined as (depth + 1).
        # If a defection occurred, normalize its depth by the branch length.
        # Otherwise, return 1.
        if not node['replies']:
            branch_length = depth + 1
            if defection_depth is None:
                return [1.0]
            else:
                return [defection_depth / branch_length]

        # Otherwise, continue recursively for each reply.
        values = []
        for child in node['replies']:
            values.extend(process_branch(child, sentiment, depth + 1, defection_depth))
        return values

    # Compute the root sentiment.
    root_sentiment = sid.polarity_scores(comment_forest['body'])['compound']
    # Process the tree starting from the root (with no defection recorded yet).
    branch_scores = process_branch(comment_forest, root_sentiment, 0, None)
    # Return the average normalized defection score.
    
    return sum(branch_scores) / len(branch_scores) if branch_scores else None