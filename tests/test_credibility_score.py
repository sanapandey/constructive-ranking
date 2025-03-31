import nltk
nltk.download('punkt_tab')
from nltk import tokenize
import pytest
import pandas as pd
import nltk, nltk.sentiment 
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_scripts.credibility import get_credibility_score

def test_credibility_score_with_links_and_investment():
        """
        Test credibility score for a conversation with external links and author investment
        """
        link_rich_conversation = {
            'selftext': 'Initial post about a research topic',
            'comments': [
                {
                    'body': 'Here\'s a detailed analysis with source: https://example.com/research',
                    'author': 'expert_user',
                    'score': '17',
                    'replies': [
                        {
                            'body': 'Great point! And here\'s another perspective: https://another-source.org',
                            'author': 'research_enthusiast', 
                            'score': 15, 
                            'replies': []

                        }, 
                        {
                            'body': 'Excellent! I really liked u/research_enthusiast\'s point about this. I wonder if the article found here could be helpful: www.google.com/help',
                            'author': 'research_fanboy', 
                            'score': 12, 
                            'replies': []
                        }, 
                        {
                            'body': 'I totally agree!',
                            'author': 'research_fangirl', 
                            'score': 7,
                            'replies': []
                        }, 
                    ]
                }
            ]
        }
        
        credibility_score = get_credibility_score(link_rich_conversation)
        assert not pd.isna(credibility_score), "Credibility score should not be NA"
        assert credibility_score > 0, "Conversation with links and investment should have positive credibility"

def test_credibility_score_low_investment():
    """
    Test credibility score for a conversation with minimal investment
    """
    low_investment_conversation = {
        'selftext': 'Short initial post',
        'comments': [
            {
                'body': 'wtf',
                'author': 'low_effort_user_2',
                'score': 1, 
                'replies': [
                      {
                        'body': 'idk',
                        'author': 'low_effort_user_3',
                        'score': 1, 
                        'replies': []
                    }
                ]
            }, 
            {
                'body': 'idk',
                'author': 'low_effort_user_3',
                'score': 1, 
                'replies': []
            }, 
            {
                'body': 'F',
                'author': 'low_effort_user_4',
                'score': 2, 
                'replies': []
            }, 
            {
                'body': 'k',
                'author': 'low_effort_user',
                'score': 2, 
                'replies': []
            }
        ]
    }
        
    credibility_score = get_credibility_score(low_investment_conversation)
    assert not pd.isna(credibility_score), "Credibility score should not be NA"
    assert credibility_score <= 1.0, "Credibility score should be reasonable"