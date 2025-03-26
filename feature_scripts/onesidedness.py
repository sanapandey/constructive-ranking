from collections import defaultdict

def get_onesidedness_score(comment_forest):
    """
    Calculates the one-sidedness (Gini coefficient) of author contributions 
    in a comment forest.
    
    Parameters:
    comment_forest (list): A list of comment dictionaries, where each comment
                          has an 'author' field and optionally a 'replies' field
                          containing nested comments.
    
    Returns:
    float: The Gini coefficient representing one-sidedness of the conversation.
           0 means perfect equality, 1 means perfect inequality.
    """
    # flatten the comment forest to get all authors
    authors = []
    
    def extract_authors(comments):
        for comment in comments:
            authors.append(comment['author'])
            if 'replies' in comment and comment['replies']:
                extract_authors(comment['replies'])
    
    extract_authors(comment_forest['comments'])
    
    # counting contributions by each author
    contribution_counts = defaultdict(int)
    for author in authors:
        contribution_counts[author] += 1
    
    # make sorted list of contribution frequencies
    frequencies = sorted(contribution_counts.values())
    
    # edge case: if there's only one author or no authors (post does not have a conversation)
    n = len(frequencies)
    if n <= 1:
        return 0 # No inequality with only one author TODO check if this makes intuitive sense?
    
    # calculate Gini coefficient
    numerator = sum((2 * i - n - 1) * x for i, x in enumerate(frequencies, 1))
    denominator = (n * sum(frequencies)) + 1e-9 #prevent division by 0
        
    gini = numerator / denominator
    return gini