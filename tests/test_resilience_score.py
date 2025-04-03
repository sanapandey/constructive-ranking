import pytest
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd 
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_scripts.resilience import get_resilience_score

def test_resilience_fully_positive():
    """
    A conversation where **no negative defection occurs**.
    Expected result: **NaN** since there is no defection.
    """
    fully_positive_thread = {
        "selftext": "I love nature and the environment!",
        "comments": [
            {"body": "Nature is beautiful!", "replies": [
                {"body": "Absolutely, it's so calming!", "replies": [
                    {"body": "I go hiking every weekend!", "replies": [
                        {"body": "That sounds amazing!", "replies": [
                            {"body": "Yeah, fresh air is the best!", "replies": [
                                {"body": "I want to travel to national parks!", "replies": []}
                            ]}
                        ]}
                    ]}
                ]}
            ]},
            {"body": "Being in nature helps mental health!", "replies": [
                {"body": "Yes! Sunshine makes me feel alive!", "replies": []}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(fully_positive_thread)
    assert pd.isna(resilience_score), "Fully positive thread should return NaN"

def test_resilience_early_defection_recovery():
    """
    A conversation where **early defection happens**, but later comments **recover**.
    Expected result: **High resilience** (~0.5 to 1).
    """
    early_defection_thread = {
        "selftext": "Let's talk about productivity tips.",
        "comments": [
            {"body": "I love using Notion!", "replies": [
                {"body": "Same! It keeps me so organized!", "replies": [
                    {"body": "Honestly, I feel like it's overrated.", "replies": [  # Defection happens here (negative)
                        {"body": "I get why you feel that, but it works well for me!", "replies": [
                            {"body": "Fair enough, I might give it another try.", "replies": [
                                {"body": "Yeah, you can customize it a lot!", "replies": []}
                            ]}
                        ]}
                    ]}
                ]}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(early_defection_thread)
    assert 0.5 <= resilience_score <= 1.0, "Thread should show strong resilience after early defection"

def test_resilience_early_defection_no_recovery():
    """
    A conversation where **early defection happens**, and sentiment stays **negative**.
    Expected result: **Low resilience** (~-1 to -0.5).
    """
    early_negative_thread = {
        "selftext": "Discussing the future of AI.",
        "comments": [
            {"body": "AI is the future!", "replies": [
                {"body": "I think it has great potential.", "replies": [
                    {"body": "Honestly, AI is ruining creativity.", "replies": [  # Defection happens here
                        {"body": "Yeah, I don't trust it at all.", "replies": [
                            {"body": "Corporations just want to replace humans.", "replies": [
                                {"body": "It's going to take all our jobs!", "replies": []}
                            ]}
                        ]}
                    ]}
                ]}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(early_negative_thread)
    assert -1.0 <= resilience_score <= -0.5, "Thread should have low resilience (stays negative after defection)"

def test_resilience_mixed_defections():
    """
    A conversation with **multiple defections**. Some branches recover, others do not.
    Expected result: **Medium resilience** (~0 to 0.5).
    """
    mixed_defection_thread = {
        "selftext": "Opinions on the new movie?",
        "comments": [
            {"body": "It was fantastic!", "replies": [
                {"body": "I enjoyed it too!", "replies": [
                    {"body": "Meh, I think it was overhyped.", "replies": [  # Defection here
                        {"body": "I get that, but I still loved it!", "replies": []}  # Partial recovery
                    ]}
                ]}
            ]},
            {"body": "I didn't like it.", "replies": [
                {"body": "Really? What didn't you like?", "replies": [
                    {"body": "The pacing was awful.", "replies": [
                        {"body": "Yeah, it dragged a lot.", "replies": [
                            {"body": "I almost fell asleep!", "replies": []}  # Stays negative
                        ]}
                    ]}
                ]}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(mixed_defection_thread)
    assert 0.0 <= resilience_score <= 0.5, "Thread should show mixed resilience"

def test_resilience_fully_negative():
    """
    A conversation where **all comments are negative from the start**.
    Expected result: **NaN** (No defection happens).
    """
    fully_negative_thread = {
        "selftext": "Discuss the state of politics today.",
        "comments": [
            {"body": "Politics is a mess.", "replies": [
                {"body": "Yeah, everything is corrupt.", "replies": [
                    {"body": "I have zero hope for the system.", "replies": [
                        {"body": "Same, it's all rigged.", "replies": [
                            {"body": "No one in power actually cares about us.", "replies": []}
                        ]}
                    ]}
                ]}
            ]},
            {"body": "Nothing ever changes.", "replies": [
                {"body": "Exactly, it's all just talk.", "replies": [
                    {"body": "Why even vote anymore?", "replies": []}
                ]}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(fully_negative_thread)
    assert pd.isna(resilience_score), "Fully negative thread should return NaN"
