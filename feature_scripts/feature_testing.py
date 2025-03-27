import pytest
import pandas as pd
import nltk, nltk.sentiment 
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
from coalition import get_coalition_score
from credibility import get_credibility_score
from defection import get_defection_score
from onesidedness import get_onesidedness_score

# Test comment forest scenarios
class TestRankingModelFeatures:
    def test_coalition_score_diverse_conversation(self):
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

    def test_coalition_score_homogeneous_conversation(self):
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

    def test_credibility_score_with_links_and_investment(self):
        """
        Test credibility score for a conversation with external links and author investment
        """
        link_rich_conversation = {
            'selftext': 'Initial post about a research topic',
            'comments': [
                {
                    'body': 'Here\'s a detailed analysis with source: https://example.com/research',
                    'author': 'expert_user',
                    'replies': [
                        {
                            'body': 'Great point! And here\'s another perspective: https://another-source.org',
                            'author': 'research_enthusiast'
                        }
                    ]
                }
            ]
        }
        
        credibility_score = get_credibility_score(link_rich_conversation)
        assert not pd.isna(credibility_score), "Credibility score should not be NA"
        assert credibility_score > 0, "Conversation with links and investment should have positive credibility"

    def test_credibility_score_low_investment(self):
        """
        Test credibility score for a conversation with minimal investment
        """
        low_investment_conversation = {
            'selftext': 'Short initial post',
            'comments': [
                {
                    'body': 'k',
                    'author': 'low_effort_user',
                    'replies': []
                }
            ]
        }
        
        credibility_score = get_credibility_score(low_investment_conversation)
        assert not pd.isna(credibility_score), "Credibility score should not be NA"
        assert credibility_score <= 1.0, "Credibility score should be reasonable"

    def test_onesidedness_balanced_conversation(self):
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

    def test_onesidedness_unbalanced_conversation(self):
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

    def test_defection_score_stable_conversation(self):
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
                            'author': 'user2'
                        }
                    ]
                }
            ]
        }
        
        defection_score = get_defection_score(stable_conversation)
        assert not pd.isna(defection_score), "Defection score should not be NA"
        assert defection_score > 0, "Stable conversation should have a positive defection score"

    def test_defection_score_volatile_conversation(self):
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
                                    'author': 'user3'
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

if __name__ == '__main__':
    pytest.main()