import pytest
import pandas as pd
import nltk, nltk.sentiment 
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_scripts.coalition import get_coalition_score

def test_coalition_score_diverse_conversation():
        """
        Test coalition score for a conversation with diverse viewpoints
        """
        diverse_conversation = {
            'selftext': 'Initial post about complex topic',
            'comments': [
                {
                    'body': 'I think the first perspective is important',
                    'author': 'user1',
                    'replies': [
                        {
                            'body': 'Interesting point, but have you considered another angle?',
                            'author': 'user2'
                        }
                    ]
                },
                {
                    'body': 'I disagree completely and here\'s why',
                    'author': 'user3',
                    'replies': [
                        {
                            'body': 'You raise a valid counterpoint',
                            'author': 'user1'
                        }
                    ]
                }
            ]
        }
        
        coalition_score = get_coalition_score(diverse_conversation)
        assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
        assert coalition_score > 0.5, "Diverse conversation should have a relatively high coalition score"

def test_coalition_score_homogeneous_conversation():
    """
    Test coalition score for a conversation with very similar viewpoints
    """
    homogeneous_conversation = {
        'selftext': 'Initial post about a topic',
        'comments': [
            {
                'body': 'I completely agree with the original post',
                'author': 'user1',
                'replies': [
                    {
                        'body': 'Absolutely right!',
                         'author': 'user2'
                    },
                    {
                        'body': 'No doubt about it',
                        'author': 'user3'
                    }
                ]
            }
        ]
    }
        
    coalition_score = get_coalition_score(homogeneous_conversation)
    assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
    assert coalition_score < 0.5, "Homogeneous conversation should have a lower coalition score"



