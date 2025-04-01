from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import feature_scripts
import pytest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_scripts.defection import get_defection_score

def test_defection_score_stable_conversation():
        """
        Test defection score for a conversation without major sentiment shifts
        """
        stable_conversation = {
            'selftext': 'Initial positive post',
            'comments': [
                {
                    'body': 'Great point, I agree',
                    'author': 'user1',
                    'replies': [
                        {
                            'body': 'Me too, very insightful',
                            'author': 'user2', 
                            'replies': []
                        }
                    ]
                }
            ]
        }
        
        defection_score = get_defection_score(stable_conversation)
        assert not pd.isna(defection_score), "Defection score should not be NA"
        assert defection_score > 0, "Stable conversation should have a positive defection score"

def test_defection_score_volatile_conversation():
    """
    Test defection score for a conversation with significant sentiment shifts
    """
    volatile_conversation = {
        'selftext': 'Initial positive post',
        'comments': [
            {
                'body': 'I strongly disagree with this',
                'author': 'user1',
                'replies': [
                    {
                        'body': 'Why are you so negative?',
                        'author': 'user2',
                        'replies': [
                            {
                                'body': 'Let\'s find common ground',
                                'author': 'user3', 
                                'replies': []
                            }
                        ]
                    }
                ]
            }
        ]
    }
        
    defection_score = get_defection_score(volatile_conversation)
    assert not pd.isna(defection_score), "Defection score should not be NA"
    assert 0 <= defection_score <= len(volatile_conversation['comments'][0]['replies']) + 1, "Defection score should be within reasonable bounds"
