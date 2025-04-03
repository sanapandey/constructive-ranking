from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def calculate_resilience(json_data, neutral_threshold=0.3):
    ''' 
    Measures resilience in a conversation thread. 
    Resilience is defined as the **average sentiment after the first negative defection**.
    
    - If sentiment starts **neutral/positive** and then **defects to negative**, we track comments **after that**.
    - A **high resilience score** means sentiment **recovers** after defection.
    - A **low resilience score** means sentiment **remains negative** after defection.
    
    Returns:
    - **Average sentiment of post-defection comments** (higher = more resilient).
    - **NaN if no defection occurred**.
    '''

    sid = SentimentIntensityAnalyzer()

    def process_node(node, parent_sentiment, found_defection, post_defection_sentiments):
        """ Recursively traverse the thread, tracking sentiment after defection. """

        comment_body = node['body']
        raw_sentiment = sid.polarity_scores(comment_body)['compound']
    
        # Inherit parent's sentiment if the comment is neutral
        if abs(raw_sentiment) < neutral_threshold:
            sentiment = parent_sentiment
        else:
            sentiment = raw_sentiment
        
        # Check for defection (negative flip from positive/neutral)
        if not found_defection and parent_sentiment >= 0 and sentiment < 0:
            found_defection = True  # Mark defection point
        
        # If defection has occurred, store post-defection sentiment
        if found_defection:
            post_defection_sentiments.append(sentiment)
        
        # Recursively process replies
        for child in node.get('replies', []):
            process_node(child, sentiment, found_defection, post_defection_sentiments)

    # Adjust keys for recursion
    json_data['body'] = json_data.get('selftext', "")
    json_data['replies'] = json_data.get('comments', [])

    root_sentiment = sid.polarity_scores(json_data['body'])['compound']
    post_defection_sentiments = []
    
    process_node(json_data, root_sentiment, found_defection=False, post_defection_sentiments=post_defection_sentiments)
    
    # If no defection occurred, return NaN
    if not post_defection_sentiments:
        return float("NaN")

    # Return average post-defection sentiment (resilience score)
    return sum(post_defection_sentiments) / len(post_defection_sentiments)


def get_resilience_score(comment_forest, neutral_threshold=0.3):
    """
    Computes the **resilience score** of a conversation thread.
    
    - **High resilience (closer to 1)** → After a negative defection, sentiment **recovers**.
    - **Low resilience (closer to -1)** → After a negative defection, sentiment **remains negative**.
    - **NaN** → No defection occurred.
    
    Returns:
    - **Average resilience score across all branches**.
    """

    resilience_scores = [calculate_resilience(branch, neutral_threshold) for branch in comment_forest.get("comments", [])]
    
    # Filter out NaN values (threads without defections)
    valid_scores = [score for score in resilience_scores if not pd.isna(score)]
    
    return sum(valid_scores) / len(valid_scores) if valid_scores else float("NaN")
