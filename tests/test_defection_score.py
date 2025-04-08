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

def test_defection_score_conversations_with_hand_calculated_defection_values_1():
    """
    test_json with 2 branches from same positive root node.
    Branches: 1. pos (the root), ntr, ntr, pos, neg
              2. pos (the root), pos, pos, pos, neg
    Expected normalized defection score 0.8 
    """

    conversation08 = {
    "selftext": "I love this! It is amazing.",
    "comments": [
        {
        "body": "The cat sat on the mat.",
        "replies": [
            {
            "body": "I love this!",
            "replies": [
                {
                "body": "Absolutely wonderful!",
                "replies": [
                    {
                    "body": "I hate this!",
                    "replies": []
                    }
                ]
                }
            ]
            }
        ]
        },
        {
        "body": "The cat sat on the mat.",
        "replies": [
            {
            "body": "The cat sat on the mat.",
            "replies": [
                {
                "body": "I love this!",
                "replies": [
                    {
                    "body": "I hate this!",
                    "replies": []
                    }
                ]
                }
            ]
            }
        ]
        }
    ]
    }

    defection_score = get_defection_score(conversation08)
    assert not pd.isna(defection_score), "Defection score should not be NA"
    assert defection_score == 0.8, "Defection score was expected to be 0.8."

def test_defection_score_conversations_with_hand_calculated_defection_values_2():
    """
    three branches:
    1. pos (the root), ntr, pos, neg (defection should give 0.75)
    2. pos (the root), pos, pos      (defection should give 1)
    3. pos (the root), pos, neg, neg (defection should give 0.5)
    expected defection score: 0.75. 
    """
    conversation075 = {
        "selftext": "I love this! It is amazing.",
        "comments": [
            {
                "body": "The cat sat on the mat.",
                "replies": [
                    {
                        "body": "Absolutely wonderful!",
                        "replies": [
                            {
                                "body": "I hate this!",
                                "replies": []
                            }
                        ]
                    }
                ]
            },
            {
                "body": "I love this!",
                "replies": [
                    {
                        "body": "I really love it.",
                        "replies": []
                    }
                ]
            },
            {
                "body": "This is great.",
                "replies": [
                    {
                        "body": "I hate this.",
                        "replies": [
                            {
                                "body": "I hate this.",
                                "replies": []
                            }
                        ]
                    }
                ]
            }
        ]
    }

    defection_score = get_defection_score(conversation075)
    assert not pd.isna(defection_score), "Defection score should not be NA"
    assert defection_score == 0.75, "Defection score was expected to be 0.75."