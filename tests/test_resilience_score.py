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
                        {
  "body": "I just prefer other apps, Notion feels too complicated.",
  "replies": [
    {
      "body": "I used to feel the same way, but once I spent a bit of time with it, I found it to be super powerful! It’s so customizable.",
      "replies": [
        {
          "body": "Exactly! The flexibility is amazing. You can set it up exactly how you want, and the possibilities are endless. It really helps with organization!",
          "replies": [
            {
              "body": "Totally! I love how you can create personalized dashboards for different needs—work, personal life, projects, etc.",
              "replies": []
            }
          ]
        },
        {
          "body": "It does take some time to get comfortable, but once you do, it’s a game-changer. The integrations with other tools are awesome too!",
          "replies": [
            {
              "body": "Definitely! I’ve integrated it with Google Calendar and Slack, and it’s made a huge difference in staying on top of everything.",
              "replies": []
            }
          ]
        }
      ]
    },
    {
      "body": "I’ve found Notion to be a great all-in-one tool once you get the hang of it. It's perfect for organizing all kinds of info in one place.",
      "replies": [
        {
          "body": "Yes, same! I use it for everything—task management, note-taking, even budgeting. It’s a huge time-saver once you set it up to fit your needs.",
          "replies": []
        }
      ]
    },
    {
      "body": "If you give it another shot, I think you’ll find it easier with a few templates or tutorials. There’s a big Notion community that shares tips and workflows.",
      "replies": [
        {
          "body": "That’s a great point! The community really helps you see all the creative ways people use Notion.",
          "replies": []
        }
      ]
    }
  ]
}

                    ]}
                ]}
            ]},
            {"body": "I personally use Todoist for productivity.", "replies": [
                {"body": "I tried Todoist, but it wasn't flexible enough for me.", "replies": [
                    {"body": "Yeah, I feel that. But I still like it for simple to-do lists.", "replies": [
                        {"body": "I love the app, I think it's great!", "replies": [
                            {"body": "Me too! I feel like it's honestly really good for planning as well.", "replies": []}
                        ]}
                    ]}
                ]}
            ]},
            {"body": "I also use Trello to organize my tasks visually.", "replies": [
                {"body": "Trello's boards are so helpful for visualizing everything!", "replies": [
                    {"body": "Yes, I love how easy it is to move things around.", "replies": [
                        {"body": "One of my favorite features of all time is the color coding! So good and satisfying!", "replies": []}
                    ]}
                ]}
            ]},
            {
  "body": "Does anyone here use ClickUp?",
  "replies": [
    {
      "body": "Yes, I absolutely love it! It's been a game-changer for my team and me.",
      "replies": [
        {
          "body": "Same here! The task management and automation features are a lifesaver. We’ve gotten so much more efficient.",
          "replies": [
            {
              "body": "I agree! I can’t imagine going back to anything else. The ability to customize views for different team members is huge!",
              "replies": []
            }
          ]
        },
        {
          "body": "I’ve been using it for a few months now, and the goal setting and time tracking features really help me stay on top of everything.",
          "replies": [
            {
              "body": "Yes, exactly! It’s so satisfying to see everything tracked and organized in one place. Plus, the mobile app is super handy!",
              "replies": []
            }
          ]
        }
      ]
    },
    {
      "body": "I’ve heard great things about ClickUp, but I’ve never used it. Is it really as good as people say?",
      "replies": [
        {
          "body": "Absolutely! It’s really intuitive, and there’s a ton of tutorials to get you started. Once you get the hang of it, you’ll wonder how you ever worked without it.",
          "replies": [
            {
              "body": "I second that! The learning curve isn’t bad, and once you’re familiar, it really speeds up your workflow. Definitely worth the try.",
              "replies": []
            }
                    ]
        }
                ]
    }
    ]
        },

            {"body": "I like using Google Keep for quick notes!", "replies": [
                {"body": "Google Keep is great for reminders, very simple to use.", "replies": [
                    {"body": "Exactly, I love how it syncs across all devices.", "replies": [
                        {"body": "What do you use for task management?", "replies": [
                            {"body": "honestly I'm a huge fan of trellio or evernote--they've helped me get my life together!", "replies": [
                                {"body": "Thanks so much for the tip!", "replies": []}
                            ]}
                        ]}
                    ]}
                ]}
            ]}
        ]
    }
    
    resilience_score = get_resilience_score(early_defection_thread)
    assert 0.5 <= resilience_score <= 1.0, "Thread should show strong resilience after early defection"


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
                {"body": "Your opinion is beyond wrong--you must be an idiot.", "replies": [
                    {"body": "I can't believe people this dumb are online.", "replies": []}  # Negative 
                ]}
            ]},
            {"body": "I couldn't get into it.", "replies": [
                {"body": "Really? Why not?", "replies": [
                    {"body": "You must have no critical thinking skills to have enjoyed that movie--are you stupid?", "replies": [
                        {"body": "Wow, you're really aggressive. I still found it enjoyable.", "replies": []}  # Partial recovery
                    ]}
                ]}
            ]},
            {"body": "The action scenes were fun!", "replies": [
                {"body": "True, they were entertaining.", "replies": [
                    {"body": "But some felt unnecessary.", "replies": [
                        {"body": "Quality was so bad for the action, I feel like it was a waste.", "replies": []}  # Mostly negative
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
    assert 0.0 <= resilience_score <= 0.5, "Thread should show low resilience, as majority defect with only one recovery."


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
