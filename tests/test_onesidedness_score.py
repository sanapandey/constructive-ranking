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
        assert onesidedness_score < 0.5, "Balanced conversation should have low one-sidedness"

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
    assert onesidedness_score > 0.7, "Unbalanced conversation should have high one-sidedness"
