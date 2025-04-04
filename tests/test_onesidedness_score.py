from collections import defaultdict
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_scripts.onesidedness import get_onesidedness_score

def test_onesidedness_balanced_conversation():
        """
        Test one-sidedness score for a balanced conversation
        """
        balanced_conversation = {
            'comments': [
                {'author': 'user1', 'body': 'First perspective'},
                {'author': 'user2', 'body': 'Second perspective'},
                {'author': 'user3', 'body': 'Third perspective', 
                 'replies': [
                     {'author': 'user1', 'body': 'Response to third perspective'},
                     {'author': 'user2', 'body': 'Another response'}
                 ]
                }
            ]
        }
        
        onesidedness_score = get_onesidedness_score(balanced_conversation['comments'])
        assert 0 <= onesidedness_score <= 1, "One-sidedness score should be between 0 and 1"
        assert onesidedness_score < 0.3, "Balanced conversation should have low one-sidedness"

def test_onesidedness_unbalanced_conversation():
    """
    Test one-sidedness score for an unbalanced conversation
    """
    unbalanced_conversation = {
        'comments': [
            {'author': 'dominant_user', 'body': 'First comment'},
            {'author': 'dominant_user', 'body': 'Second comment'},
            {'author': 'dominant_user', 'body': 'Third comment'},
            {'author': 'other_user', 'body': 'Single comment'}
        ]
    }
        
    onesidedness_score = get_onesidedness_score(unbalanced_conversation['comments'])
    assert 0 <= onesidedness_score <= 1, "One-sidedness score should be between 0 and 1"
    assert onesidedness_score > 0.2, "Unbalanced conversation should have high one-sidedness"

def test_onesidedness_perfectly_balanced():
    """
    Test a perfectly balanced conversation where all 15 participants 
    contribute equally. Expecting a very low one-sidedness score.
    """
    balanced_conversation = {
        'comments': [{'author': f'user{i}', 'body': f'Comment {i}'} for i in range(15)]
    }

    onesidedness_score = get_onesidedness_score(balanced_conversation['comments'])
    assert 0 <= onesidedness_score <= 1, "Score should be between 0 and 1"
    assert onesidedness_score < 0.1, "A perfectly balanced conversation should have a very low one-sidedness score"

def test_onesidedness_single_dominant_speaker():
    """
    Test a conversation where a single user dominates, making up 90% of comments.
    Expecting a very high one-sidedness score.
    """
    dominant_conversation = {
        'comments': (
            [{'author': 'dominant_user', 'body': f'Dominant comment {i}'} for i in range(13)] +
            [{'author': 'other_user', 'body': 'Rare comment 1'}, {'author': 'other_user2', 'body': 'Rare comment 2'}]
        )
    }

    onesidedness_score = get_onesidedness_score(dominant_conversation['comments'])
    assert 0 <= onesidedness_score <= 1, "Score should be between 0 and 1"
    assert onesidedness_score > 0.5, "A single dominant speaker should result in high one-sidedness"

def test_onesidedness_few_dominant_voices():
    """
    Test a conversation where 3 users dominate, but there are some other voices.
    Expecting a moderate one-sidedness score.
    """
    few_dominant_conversation = {
        'comments': (
            [{'author': 'user1', 'body': f'Frequent comment {i}'} for i in range(5)] +
            [{'author': 'user2', 'body': f'Frequent comment {i}'} for i in range(5)] +
            [{'author': 'user3', 'body': f'Frequent comment {i}'} for i in range(4)] +
            [{'author': f'user{i}', 'body': f'Rare comment {i}'} for i in range(4, 7)]
        )
    }

    onesidedness_score = get_onesidedness_score(few_dominant_conversation['comments'])
    assert 0 <= onesidedness_score <= 1, "Score should be between 0 and 1"
    assert 0.3 < onesidedness_score < 0.6, "Few dominant voices should yield moderate one-sidedness"

def test_onesidedness_reply_heavy_discussion():
    """
    Test a conversation where replies are the main form of engagement.
    Expecting the score to be influenced by reply authorship.
    """
    reply_heavy_conversation = {
        'comments': [
            {'author': 'starter', 'body': 'Initial topic',
             'replies': [
                 {'author': f'user{i % 5}', 'body': f'Reply {i}'} for i in range(14)
             ]}
        ]
    }

    onesidedness_score = get_onesidedness_score(reply_heavy_conversation['comments'])
    assert 0 <= onesidedness_score <= 1, "Score should be between 0 and 1"
    assert 0.1 < onesidedness_score < 0.5, "Reply-heavy discussions should have low to moderate one-sidedness"

def test_onesidedness_mixed_participation():
    """
    Test a conversation where some users talk a lot, but others still contribute.
    Expecting a mid-range one-sidedness score.
    """
    mixed_conversation = {
        'comments': (
            [{'author': 'frequent_user', 'body': f'Frequent comment {i}'} for i in range(7)] +
            [{'author': 'moderate_user', 'body': f'Moderate comment {i}'} for i in range(5)] +
            [{'author': f'rare_user{i}', 'body': f'Rare comment {i}'} for i in range(3)]
        )
    }

    onesidedness_score = get_onesidedness_score(mixed_conversation['comments'])
    assert 0 <= onesidedness_score <= 1, "Score should be between 0 and 1"
    assert 0.3 < onesidedness_score < 0.6, "Mixed participation with some frequent commenters should yield moderate one-sidedness"

