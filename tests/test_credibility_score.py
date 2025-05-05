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
from feature_scripts.credibility import *
from feature_scripts.credibility_v2 import get_credibility_subfeatures

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
        
        credibility_features = get_credibility_subfeatures(link_rich_conversation)
        credibility_score = get_credibility_score(link_rich_conversation) #sum(credibility_features.values()) / len(credibility_features)
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


import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import re

# Test helper functions

def test_get_comment_length():
    """Test word count calculation for comments"""
    assert get_comment_length("This is a test.") == 4
    assert get_comment_length("One") == 1
    assert get_comment_length("") == 0
    assert get_comment_length("Multiple    spaces") == 2

def test_get_comment_has_links():
    """Test link detection in comments"""
    assert get_comment_has_links("Check http://example.com") is True
    assert get_comment_has_links("Visit www.site.org") is True
    assert get_comment_has_links("No links here") is False
    assert get_comment_has_links("Hidden link example.com") is True
    assert get_comment_has_links("Just text") is False

def test_get_n_spelling_mistakes():
    """Test spelling mistake detection"""
    test_words = set(["correct", "spelling", "test"])
    assert get_n_spelling_mistakes("correct spelling test", test_words) == 0  # "Correct" capitalized
    assert get_n_spelling_mistakes("incorret spellling tst", test_words) == 3
    assert get_n_spelling_mistakes("", test_words) == 0
    assert get_n_spelling_mistakes("123 !@#", test_words) == 0

@patch('openai.ChatCompletion.create')
def test_get_comment_readability(mock_openai):
    """Test readability score calculation"""
    mock_openai.return_value = {
        "choices": [{"message": {"content": "75"}}]
    }
    assert get_comment_readability("This is a test sentence.") == 75.0
    
    mock_openai.return_value = {
        "choices": [{"message": {"content": "5"}}]
    }
    assert get_comment_readability("k") == 5.0

def test_extract_references_from_comment():
    """Test author reference extraction"""
    assert extract_references_from_comment("As u/user1 mentioned") == ["user1"]
    assert extract_references_from_comment("u/abc and u/def") == ["abc", "def"]
    assert extract_references_from_comment("No references") == []
    assert extract_references_from_comment("u/name1, u/name2! u/name3?") == ["name1", "name2", "name3"]

# Test core functionality

def test_get_investment_values_thread_average():
    """Test calculation of thread averages for investment metrics"""
    test_thread = {
        'comments': [
            {
                'body': 'First comment with link http://test.com',
                'score': 10,
                'replies': [
                    {
                        'body': 'Short reply',
                        'score': 5,
                        'replies': []
                    }
                ]
            },
            {
                'body': 'Second comment',
                'score': 8,
                'replies': []
            }
        ]
    }
    
    result = get_investment_values_thread_average(test_thread, VALID_WORDS)
    print(result)
    assert result['links'] == 0.5  # 1 link in 2 comments
    assert result['vote score'] == (10 + 5 + 8) / 3
    assert result['length'] == (5 + 2 + 2) / 3

def test_get_author_comments_dictionaries():
    """Test author comment aggregation"""
    test_thread = {
        'comments': [
            {
                'body': 'Comment 1',
                'author': 'user1',
                'score': 10,
                'replies': [
                    {
                        'body': 'Reply 1',
                        'author': 'user2',
                        'score': 5,
                        'replies': []
                    }
                ]
            },
            {
                'body': 'Comment 2',
                'author': 'user1',
                'score': 8,
                'replies': []
            }
        ]
    }
    
    result = get_author_comments_dictionaries(test_thread)
    print(result)
    assert len(result['user1']) == 2
    assert len(result['user2']) == 1
    assert result['user1'][0]['body'] == "Comment 2"
    assert result['user2'][0]['body'] == "Reply 1"
    assert result['user2'][0]['vote score'] == 5

def test_create_author_reputation_df():
    """Test author reputation calculation"""
    test_thread = {
        'comments': [
            {
                'body': 'Mentioning u/user2',
                'author': 'user1',
                'score': 10,
                'replies': [
                    {
                        'body': 'Thanks u/user1',
                        'author': 'user2',
                        'score': 5,
                        'replies': []
                    }
                ]
            },
            {
                'body': 'No references',
                'author': 'user3',
                'score': 8,
                'replies': []
            }
        ]
    }
    
    df = create_author_reputation_df(test_thread)
    assert df.loc[df['author'] == 'user1', 'references_others'].values[0] == 1
    assert df.loc[df['author'] == 'user2', 'is_referenced'].values[0] == 1
    assert df.loc[df['author'] == 'user3', 'references_others'].values[0] == 0
    assert 'reputation_score' in df.columns

def test_create_author_investment_df():
    """Test author investment calculation;  readability + links - grammar/spelling mistakes"""
    test_thread = {
        'comments': [
            {
                'body': 'Detailed comment with link http://test.com',
                'author': 'user1',
                'score': 10,
                'replies': [
                    {
                        'body': 'Short',
                        'author': 'user2',
                        'score': 5,
                        'replies': []
                    }
                ]
            }
        ]
    }
    
    df = create_author_investment_df(test_thread, VALID_WORDS)
    assert len(df) == 2
    assert df.loc[df['author'] == 'user1', 'average_links'].values[0] == 1
    assert df.loc[df['author'] == 'user2', 'average_length'].values[0] == 1
    assert 'investment_score' in df.columns

# Test edge cases

def test_empty_thread():
    """Test behavior with empty thread"""
    empty_thread = {'comments': []}
    with pytest.raises(AssertionError):
        get_investment_values_thread_average(empty_thread, VALID_WORDS)
    
    assert get_credibility_score(empty_thread) is pd.NA

def test_single_comment_thread():
    """Test thread with only one comment"""
    single_thread = {
        'comments': [
            {
                'body': 'Only comment',
                'author': 'user1',
                'score': 1,
                'replies': []
            }
        ]
    }
    assert get_credibility_score(single_thread) is pd.NA

#Note for Sana: think about testing for deleted comments?

# Integration tests

def test_full_credibility_calculation():
    """Test end-to-end credibility score calculation"""
    test_thread = {
        'selftext': 'Initial post',
        'comments': [
            {
                'body': 'Detailed analysis with reference to u/expert and source http://research.org',
                'author': 'researcher',
                'score': 20,
                'replies': [
                    {
                        'body': 'Building on u/researcher work with additional data',
                        'author': 'analyst',
                        'score': 15,
                        'replies': []
                    }
                ]
            },
            {
                'body': 'Simple comment',
                'author': 'casual',
                'score': 2,
                'replies': []
            }
        ]
    }
    
    score = get_credibility_score(test_thread)
    assert not pd.isna(score)
    assert isinstance(score, float)
    # The actual score value will depend on the implementation details
    assert score > 0  # This thread should have positive credibility -- make this a fixed value for the openai bit 