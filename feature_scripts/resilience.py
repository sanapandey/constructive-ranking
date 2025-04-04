from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def extract_comments_from_forest(comment_forest):
    """
    Extract comment texts from a comment forest structure
    
    Args:
        comment_forest (List[Dict]): A nested comment structure where each comment
                                     has 'content' and optionally 'replies'
    
    Returns:
        List[str]: Flattened list of comment texts
    """
    comment_texts = []
    
    def extract_recursive(comments):
        for comment in comments:
            if 'body' in comment:  # Extract body text from each comment
                comment_texts.append(comment['body'])
            if 'replies' in comment and comment['replies']:  # Recursively extract replies
                extract_recursive(comment['replies'])
    
    extract_recursive(comment_forest)
    #print(comment_forest)
    return comment_texts


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

    # Extract comments from the comment forest
    comments = extract_comments_from_forest(json_data.get("comments", []))
    
    if not comments:  # If there are no comments, return NaN
        return float("NaN")
    
    def process_node(comment_text, parent_sentiment, found_defection, sentiment_changes):
        """ Process each comment and its replies recursively. """
        raw_sentiment = sid.polarity_scores(comment_text)['compound']
        
        # Inherit parent's sentiment if the comment is neutral
        if abs(raw_sentiment) < neutral_threshold:
            sentiment = parent_sentiment
        else:
            sentiment = raw_sentiment

        
        # Detect defection (negative flip from neutral/positive)
        if not found_defection and parent_sentiment >= 0 and sentiment < -neutral_threshold:
            found_defection = True  # Mark defection point
            #print(f"Defection detected! Parent Sentiment: {parent_sentiment}, New Sentiment: {sentiment}")
        
        # Track sentiment after defection
        if found_defection:
            sentiment_changes.append(sentiment)
            #print(f"Added sentiment after defection: {sentiment}")
        
        return found_defection, sentiment_changes

    root_sentiment = sid.polarity_scores(json_data['selftext'])['compound']
    sentiment_changes = []
    found_defection = False
    
    # Process each comment in the thread
    for comment in comments:
        found_defection, sentiment_changes = process_node(comment, root_sentiment, found_defection, sentiment_changes)
    
    # If no defection occurred, return NaN
    if not sentiment_changes:
        #print("No defection occurred!")
        return float("NaN")
    
    
    # Calculate and return the average sentiment of post-defection comments
    return sum(sentiment_changes) / len(sentiment_changes)


def get_resilience_score(comment_forest, neutral_threshold=0.3):
    """
    Computes the **resilience score** of a conversation thread.
    
    - **High resilience (closer to 1)** → After a negative defection, sentiment **recovers**.
    - **Low resilience (closer to -1)** → After a negative defection, sentiment **remains negative**.
    - **NaN** → No defection occurred.
    
    Returns:
    - **Average resilience score across all branches**.
    """

    if "selftext" in comment_forest:
        return calculate_resilience(comment_forest, neutral_threshold)
    
    # Otherwise process as comment forest
    resilience_scores = [calculate_resilience({"selftext": "", "comments": [branch]}, neutral_threshold) 
                        for branch in comment_forest.get("comments", [])]
    
    valid_scores = [score for score in resilience_scores if not pd.isna(score)]
    resilience_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("NaN")
    #print("negative resilience score: " + resilience_score)
    if resilience_score <= 0.1 and resilience_score >= -0.1: 
        return float("NaN")
    return resilience_score

