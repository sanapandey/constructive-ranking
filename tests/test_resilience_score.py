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
                                {"body": "I want to travel to national parks!", "replies": [
                                    {"body": "That would be a dream!", "replies": [
                                        {"body": "I’ve been to Yellowstone, it’s breathtaking!", "replies": [
                                            {"body": "Oh wow, I’ve heard so much about it!", "replies": [
                                                {"body": "You should definitely go! The wildlife is incredible!", "replies": [
                                                    {"body": "I plan to soon, I can’t wait!", "replies": []}
                                                ]}
                                            ]}
                                        ]}
                                    ]}
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]},
            {"body": "Being in nature helps mental health!", "replies": [
                {"body": "Yes! Sunshine makes me feel alive!", "replies": [
                    {"body": "Same here, being outdoors is the best for my mood!", "replies": [
                        {"body": "Exactly, it clears your mind!", "replies": []}
                    ]}
                ]}
            ]},
            {"body": "Hiking is a great way to connect with nature.", "replies": [
                {"body": "I agree, the peace and quiet is therapeutic.", "replies": [
                    {"body": "Plus the scenery is always stunning.", "replies": []}
                ]}
            ]},
            {"body": "I enjoy camping under the stars!", "replies": [
                {"body": "That sounds incredible! I love stargazing.", "replies": [
                    {"body": "It’s so peaceful to just look up and relax.", "replies": []}
                ]}
            ]},
            {"body": "Fresh air is so refreshing!", "replies": [
                {"body": "It’s the best, it just makes everything feel better.", "replies": [
                    {"body": "Totally, you can breathe easy and feel calm.", "replies": []}
                ]}
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
                    {"body": "Honestly, I feel like it's overrated. The app sucks and the interface is really bad. One of the worst apps I've used recently.", "replies": [  # Defection happens here (negative)
                        {"body": "I get why you feel that, but it works well for me!", "replies": [
                            {"body": "Fair enough, I might give it another try.", "replies": [
                                {"body": "Yeah, you can customize it a lot!", "replies": []}
                            ]}
                        ]},
                        {"body": "I just prefer other apps, Notion feels too complicated.", "replies": [  # A second comment with negative sentiment
                            {"body": "I think it takes some time to get used to, but it can be great once you figure it out.", "replies": [
                                {"body": "True, I might revisit it.", "replies": []}
                            ]}
                        ]}
                    ]}
                ]}
            ]},
            {"body": "I personally use Todoist for productivity.", "replies": [
                {"body": "I tried Todoist, but it wasn't flexible enough for me.", "replies": [
                    {"body": "Yeah, I feel that. But I still like it for simple to-do lists.", "replies": []}
                ]}
            ]},
            {"body": "I also use Trello to organize my tasks visually.", "replies": [
                {"body": "Trello's boards are so helpful for visualizing everything!", "replies": [
                    {"body": "Yes, I love how easy it is to move things around.", "replies": []}
                ]}
            ]},
            {"body": "Does anyone here use ClickUp?", "replies": [
                {"body": "I've heard about it but never tried it.", "replies": [
                    {"body": "It's pretty versatile, has lots of features like time tracking and goal setting.", "replies": []}
                ]}
            ]},
            {"body": "I like using Google Keep for quick notes!", "replies": [
                {"body": "Google Keep is great for reminders, very simple to use.", "replies": [
                    {"body": "Exactly, I love how it syncs across all devices.", "replies": []}
                ]}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(early_defection_thread)
    assert 0.3 <= resilience_score <= 1.0, "Thread should show strong resilience after early defection"


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
            ]},
            {"body": "It was okay, some parts were slow.", "replies": [
                {"body": "I agree, some scenes could've been cut.", "replies": [
                    {"body": "But overall, I thought the acting was great!", "replies": [
                        {"body": "Yeah, the cast did an excellent job!", "replies": []}  # Recovery here
                    ]}
                ]}
            ]},
            {"body": "I loved the soundtrack!", "replies": [
                {"body": "Yeah, the music was amazing!", "replies": [
                    {"body": "Definitely! It made some scenes so much better.", "replies": []}  # Recovery
                ]}
            ]},
            {"body": "I couldn't get into it.", "replies": [
                {"body": "Really? Why not?", "replies": [
                    {"body": "The storyline was weak, and the characters were flat.", "replies": [
                        {"body": "I see your point, but I still found it enjoyable.", "replies": []}  # Partial recovery
                    ]}
                ]}
            ]},
            {"body": "The action scenes were fun!", "replies": [
                {"body": "True, they were entertaining.", "replies": [
                    {"body": "But some felt unnecessary.", "replies": [
                        {"body": "I thought they were a good balance of excitement.", "replies": []}  # Recovery here
                    ]}
                ]}
            ]},
            {"body": "Worst movie I’ve seen in a while.", "replies": [
                {"body": "I wouldn’t go that far, but I wasn’t impressed.", "replies": [
                    {"body": "The plot was a mess.", "replies": []}  # Stays negative
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
            {"body": "Politics is a mess. I still try to have some hope though.", "replies": [
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
            ]},
            {"body": "The media just makes everything worse.", "replies": [
                {"body": "It's all fear-mongering.", "replies": [
                    {"body": "I don’t trust anything they say.", "replies": []}
                ]}
            ]},
            {"body": "Politicians are all the same.", "replies": [
                {"body": "They only care about staying in power.", "replies": [
                    {"body": "Nothing gets done, they all have hidden agendas.", "replies": []}
                ]}
            ]},
            {"body": "The government never listens to the people.", "replies": [
                
                {"body": "They only listen when it benefits them.", "replies": []}

            ]},
            {"body": "I can't trust anyone in power.", "replies": [
                {"body": "Me neither, it's all a scam.", "replies": []}
            ]},
            {"body": "The political system is broken.", "replies": [
                {"body": "It's beyond fixing at this point.", "replies": []}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(fully_negative_thread)
    assert resilience_score < 0.1, "Initial neutral statement with fully negative comment section should have a minimal resilience score."
