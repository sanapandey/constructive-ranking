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
    "title": "AITAH for refusing to give my cousin a free tattoo just because I’m an artist?",
    "author": "TwinkleMoonlitGlowb",
    "subreddit": "AITAH",
    "rank": 11,
    "score": 794,
    "upvote_ratio": 0.95,
    "num_comments": 342,
    "url": "https://www.reddit.com/r/AITAH/comments/1je519y/aitah_for_refusing_to_give_my_cousin_a_free/",
    "id": "1je519y",
    "selftext": "I (24F) have been a tattoo artist for about three years now. I work at a well-known studio, and while I’m not world-famous or anything, I’ve built a solid clientele and take my work very seriously.\n\nMy cousin Emma (22F) has always been super supportive of my work—liking my posts, sharing my designs, and even saying things like, “Omg, when I finally get a tattoo, I’m coming to you!” I always appreciated it and assumed she meant as a paying client.\n\nWell… nope.\n\nA few weeks ago, Emma texts me and says, “I finally decided what I want! Can you do a half-sleeve for me?” I was excited at first, but then she followed up with, “Obviously, it’ll be free since we’re family, right?”\n\nI thought she was joking, so I laughed and said, “Haha, girl, I love you, but I still gotta pay my bills.” She did not find that funny. She said she figured I’d want to help her out since she’s ‘promoted my work for years’ and that “it’s just ink and a few hours of your time.”\n\nI tried explaining that tattooing isn’t just some casual favor—it’s my literal job. I have to buy expensive supplies, clean my equipment properly, and block out time where I could be working on a paying client. I even offered her a family discount, but she wasn’t having it. She went on a full rant about how I was being greedy and should want to “share my art with people who actually care about me.”\n\nI reminded her that I have plenty of friends and family members who have paid me without issue because they respect my work. She basically scoffed and said, “Guess I’ll just go somewhere else, then.”\n\nI told her she was more than welcome to, and now she’s been passive-aggressively posting on social media about how “money changes people” and “some people let success get to their heads.”\n\nA few relatives are saying I should’ve just done it for free because “family is family,” but I honestly don’t think I should have to give away my work just because we share DNA.\n\nAITAH for refusing to tattoo her for free?",
    "comments": [
        {
            "author": "Liao1",
            "body": "NTA. Never do for free what you do for a living. If ‘family’ doesn’t respect your work, they don’t deserve free labor.",
            "score": 443
        },
        {
            "author": "JohnRedcornMassage",
            "body": "I always pay friends and family. If I care about them, I support their business.",
            "score": 98
        },
        {
            "author": "7grendel",
            "body": "Taking advantage of their paid skills for free feels super scummy.",
            "score": 40
        },
        {
            "author": "CraftandEdit",
            "body": "Are the relatives saying ‘family is family’ going to pitch in for the ink and time? Didn’t think so.",
            "score": 23
        },
        {
            "author": "BlueJayWay",
            "body": "I get it, but isn’t helping family what people do? I wouldn’t expect a *full* sleeve for free, but maybe something small?",
            "score": 120
        },
        {
            "author": "Tats4Life",
            "body": "If you do one for free, suddenly half the family is in line for a ‘small one.’ It adds up fast.",
            "score": 76
        },
        {
            "author": "EmmaLovesInk",
            "body": "If Emma truly supported OP, she would *want* to pay her.",
            "score": 55
        },
        {
            "author": "ColorMeHappy",
            "body": "If she saw the value in it, she wouldn’t expect it for free.",
            "score": 178
        },
        {
            "author": "FrecklesAndFerns",
            "body": "I get why OP feels bad, but if family really supports you, they’ll respect your work.",
            "score": 112
        },
        {
            "author": "JustSomeGuy1992",
            "body": "Would your cousin work her job for free just because you’re related? No? Then why should you?",
            "score": 204
        },
        {
            "author": "ArtisticAlchemy",
            "body": "A tattoo is a *luxury*. People should expect to pay for luxuries.",
            "score": 140
        },
        {
            "author": "NoodleQueen13",
            "body": "Emma isn’t owed a free tattoo just because she hyped OP up online. That’s like saying a restaurant should feed you for free because you left a good Yelp review.",
            "score": 189
        },
        {
            "author": "TurtleSnaps",
            "body": "Honestly, family should be the FIRST people willing to pay full price.",
            "score": 165
        },
        {
            "author": "HollowBones",
            "body": "If you do it for free, it sets a precedent. Family will always expect free work from you.",
            "score": 95
        },
        {
            "author": "StarfishLover",
            "body": "Emma’s reaction says a lot. If she truly valued you, she’d be offering to pay, not guilt-tripping.",
            "score": 210
        },
        {
            "author": "RainbowBubbles",
            "body": "‘Money changes people’ is such a manipulative thing to say when someone won’t give you free labor.",
            "score": 175
        },
        {
            "author": "BookishSphinx",
            "body": "Would Emma’s hairstylist or nail tech do free work for her because they’re friends? No? Then why is OP expected to?",
            "score": 132
        },
        {
            "author": "LemonZest22",
            "body": "Family means supporting each other, not taking advantage.",
            "score": 145
        },
        {
            "author": "OP",
            "body": "I think I get it now. I was torn because she’s family, but I also don’t want to be taken for granted. I’ll stick to my boundaries.",
            "score": 321
        },
        {
            "author": "Liao1",
            "body": "Good call! Stick to your boundaries, OP.",
            "score": 112
        },
        {
            "author": "GlitterGiraffe",
            "body": "If anything, you should charge *more* for family. ‘Inconvenience tax.’",
            "score": 99
        },
        {
            "author": "GrapeSodaLover",
            "body": "Emma can get her ‘free’ tattoo elsewhere—oh wait, no professional artist does that. Lol.",
            "score": 165
        },
        {
            "author": "BigChocoEnergy",
            "body": "If she respected your art, she’d WANT to pay you.",
            "score": 122
        },
        {
            "author": "CozyRaven",
            "body": "This reminds me of when my cousin expected me to bake her wedding cake for free. Some people don’t get how much effort skilled work takes.",
            "score": 144
        },
        {
            "author": "WiseOrchid",
            "body": "People will pay strangers but expect free work from family. It’s backwards.",
            "score": 186
        }
    ]
}

        
        coalition_score = get_coalition_score(diverse_conversation)
        assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
        assert coalition_score > 0.2, "Diverse conversation should have a relatively high coalition score"

def test_coalition_score_homogeneous_conversation():
    """
    Test coalition score for a conversation with very similar viewpoints
    """
    homogeneous_conversation ={
    "title": "AITAH for not letting my friend stay at my place after she showed up uninvited with her dog?",
    "author": "RadiantFairyGlowa",
    "subreddit": "AITAH",
    "rank": 19,
    "score": 268,
    "upvote_ratio": 0.96,
    "num_comments (reported by reddit)": 46,
    "url": "https://www.reddit.com/r/AITAH/comments/1je5pvd/aitah_for_not_letting_my_friend_stay_at_my_place/",
    "id": "1je5pvd",
    "selftext": "So, I (24F) have a pretty small apartment\\u2014like, two bedrooms, one bathroom, and not a lot of space. I love my place, but it\\u2019s definitely cozy.\n\nRecently, my friend Lara (25F) called me and said she was coming to town for a couple of days and asked if she could stay at my place. We\\u2019re close, so I said yes, no problem\\u2014but here\\u2019s the kicker: she showed up with her dog and didn\\u2019t mention it beforehand.\n\nNow, I love animals, but my apartment is not pet-friendly, and I\\u2019m allergic to dogs. I\\u2019ve mentioned my allergies to her before, and she knows I\\u2019m not the biggest fan of having pets in small spaces.\n\nWhen she arrived, I was polite and offered to help her find a nearby pet-friendly hotel, but she started insisting that I should just let her keep her dog at my place for the night. She said, \\u201cIt\\u2019s just for a couple of days, can\\u2019t you just suck it up?\\u201d\n\nI told her that I didn\\u2019t feel comfortable with her dog being there, especially since I would be miserable with my allergies and it\\u2019s really just too small for a dog to be running around. She got really upset, saying that I was being unreasonable and overreacting and that I was \\u201cruining her trip.\\u201d\n\nI ended up telling her she could either leave the dog in a kennel or find another place to stay. She left and didn\\u2019t speak to me for a couple of days. A few of my mutual friends think I should have been more accommodating, but I think I have a right to prioritize my health and comfort.\n\nAITAH for not letting her stay with her dog?",
    "comments": [
        {
            "author": "Competitive_Ask_9179",
            "body": "Big NTA - I hate it when someone expects you to love their dog...",
            "score": 72,
            "replies": []
        },
        {
            "author": "Public_Road_6426",
            "body": "NTA.   I love dogs, but I also love breathing, and sadly the two do not mix well, at least for me.  Asking someone to \"just suck it up\" regarding something that causes them physical (and potentially life threatening) distress is breath-takingly narcissistic.",
            "score": 45,
            "replies": []
        },
        {
            "author": "CocoaAlmondsRock",
            "body": "NTA, and the people who said you should have are idiots. You do not bring uninvited guests -- be they dog, child, or any other animal or human -- to a person's house or event. It's RUDE, and the person hosting does NOT have to accept them.\n\nYour allergies are a valid excuse, but frankly, you don't need an excuse. \"No\" is a complete sentence.",
            "score": 23,
            "replies": []
        },
        {
            "author": "Vegoia2",
            "body": "she is rude and void of manners to try to force anything on you while you were being gracious.",
            "score": 8,
            "replies": []
        },
        {
            "author": "Wise_woman_1",
            "body": "NTA!  Your \\u201cfriend\\u201d is an entitled pos. She wanted you to jeopardize your home (no pets means no pets), your health, your comfort, your happiness and more so her trip wouldn\\u2019t be ruined?!  You need to reconsider your friendship. \\ud83e\\udd2c",
            "score": 6,
            "replies": []
        },
        {
            "author": "yakkerswasneverhere",
            "body": "Your friend is an entitled dick.",
            "score": 5,
            "replies": []
        },
        {
            "author": "Snackinpenguin",
            "body": "So she wants a free place to stay at the last minute, and you\\\u2019re the unreasonable one?  Clearly she doesn\\\u2019t give two effs about your health, allergies or the lingering effects of pet dander in your own house. I doubt she was even offering to pay for having your house deep cleaned after.  She\\\u2019s not a real friend. NTA.",
            "score": 4,
            "replies": []
        },
        {
            "author": "4me2knowit",
            "body": "She did bait and switch deliberately not telling you.  NTA",
            "score": 4,
            "replies": []
        },
        {
            "author": "Saint_Blaise",
            "body": "Another 24F fake post.",
            "score": 3,
            "replies": []
        },
        {
            "author": "HallAccomplished5000",
            "body": "NTA. A simple 'Sorry this is my safe space and I'm allergic to dogs. It's not a suck it up for a few days. It is months of full on cleaning trying to get rid of every little hair on the dog to stop an allergic reaction and months of discomfort'.\\u00a0\n\n\nTell the friends that next time she pulls this stunt you'll send them to your house and see how they like it.\\u00a0\n\n\nYou don't bring animals to someone else's home without asking. Pet hair on the other hand. That is a nightmare to try and contain and keep off your clothes.",
            "score": 2,
            "replies": []
        },
        {
            "author": "Fredredphooey",
            "body": "NTA. If a \"friend\" told me that I should put my health at risk by letting a dog stay with me, I would rethink that relationship.\\u00a0",
            "score": 2,
            "replies": []
        },
        {
            "author": "2_old_for_this_spit",
            "body": "NTA\n\nThe people who tell you to just take allergy medication and suck it up are usually among the fortunate few who have never experienced a severe allergy episode.",
            "score": 2,
            "replies": []
        },
        {
            "author": "MandyRose8713",
            "body": "Even if you were not allergic and had a huge mansion NTA. It is your home therefor your decision. She didn't tell you ahead of time it's not fair to just assume you would be ok with it",
            "score": 3,
            "replies": []
        },
        {
            "author": "PrincessBella1",
            "body": "NTA. Your friend FAFO. She knew you wouldn't be happy with having her dog there so she just thought that she would bring him and deal with the consequences later. She is not your friend. Anyone who gives you grief isn't your friend either.",
            "score": 2,
            "replies": []
        },
        {
            "author": "FairyFartDaydreams",
            "body": "NTA she KNEW absolutely about your allergies. That is why she didn't say anything. This is 100 percent on her",
            "score": 2,
            "replies": []
        },
        {
            "author": "BayAreaPupMom",
            "body": "How does one \"suck up\" allergies? Dog hair gets everywhere. The effects would be lasting after she and her pup were long gone. I hate entitled pet owners like your \"friend.\" She and anyone who sides with her are not people you want to consider friends moving forward. NTA",
            "score": 1,
            "replies": []
        },
        {
            "author": "Similar_Corner8081",
            "body": "NTA I'm a dog lover but it's just rude to show up with a dog without telling you before especially knowing you are allergic. That's some entitlement.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Notahappygardener",
            "body": "I always asked if I could bring my dog and if the answer was no I got a hotel room or did not come.  NTA, she should have asked and you could have told her then and she could decide to come or not.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Cmndr_Cunnilingus",
            "body": "NTA. I have a dog and consider him part of my family. That being said I wouldn't show up on a friends doorstep asking to stay at their house with my niece or nephew in tow without clearing with them first, let alone my dog",
            "score": 1,
            "replies": []
        },
        {
            "author": "MrsRobertPlant",
            "body": "Hell no, she did this on purpose. Her problem",
            "score": 1,
            "replies": []
        },
        {
            "author": "TaxiLady69",
            "body": "NTA. Your friend is very entitled, though. I couldn't imagine being the kind of person who is so selfish that they think another persons comfort in their own home is less important than having their dog with them. By the way, she's not your friend. Friends don't do shit like this.",
            "score": 1,
            "replies": []
        },
        {
            "author": "nin_miawj",
            "body": "Nta it the dog dander would stay longer and the fur, my sister is also allergic to dogs cats and rabbits, her kids can\\\\u2019t go near them or she will end up in the hospital because of the allergies and her asthma",
            "score": 1,
            "replies": []
        },
        {
            "author": "Sleepygirl57",
            "body": "NTA as a person with allergies it\\\u2019s not that simple. Once that dander gets in your house you have to have a major deep clean. Side note your friend is a jerk for trying to force you to do something she knew would make you sick.",
            "score": 1,
            "replies": []
        },
        {
            "author": "whatev6187",
            "body": "NTA - She knew you would say no if she asked about the dog, so she tried to manipulate you.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Nalabu1",
            "body": "You were perfectly right in showing her the door. If her feelings are hurt - TOO BAD, that\\\u2019s self inflicted and not your fault or responsibility. Forget it and move on with your life.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Ok_Cherry_4585",
            "body": "NTA because even after she left, the dog hair and dander lingers and aggravates your allergies.",
            "score": 1,
            "replies": []
        },
        {
            "author": "hedwigflysagain",
            "body": "NTA, She is not a friend.",
            "score": 1,
            "replies": []
        },
        {
            "author": "thingonething",
            "body": "NTA. What is it with these entitled people expecting others to welcome their inevitably poorly behaved dogs into their homes, weddings, you name it. Your friend needed a pet friendly hotel.",
            "score": 1,
            "replies": []
        },
        {
            "author": "suzymwg",
            "body": "I hate how people think they can just bring their dog when they are visiting without asking, and assuming you will love having their dog in your place. \nOne of my pet peeves. \nNTA",
            "score": 1,
            "replies": []
        },
        {
            "author": "mcindy28",
            "body": "NTA She could have told you upfront about the dog and this would have been handled differently from the start.  Your place means your comfort is the most important.",
            "score": 1,
            "replies": []
        },
        {
            "author": "CuteTangelo3137",
            "body": "You should also post this in r/EntitledPeople as your friend's assumption that you would just cave is pretty ballsy. I would turn it around on her that to bring her dog to stay with someone who is allergic to them was really inconsiderate. She should have told you she was bringing her dog when she asked to stay with you and she knows it. She also knew you would say no and that's why she did it. To act like you're the one who did something wrong is BS. She needs a reality check. \n\nI'm glad you made her find someplace else. And you're NTA.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Laquila",
            "body": "These OPs need to harden up and use stronger language. All this airy-fairy \"I don't feel comfortable\" language, and the politeness and offering to help, and long-winded explanations and justification, needs to be changed to: \"No!\".  \n\nThose assholes KNOW what they're doing. She showed up deliberately with the dog, to force you into it. After all, how could you just turn her away?!! You big meanie! Everyone knows you have to ask if you can bring your dog first. Everyone. \n\nNTA.",
            "score": 1,
            "replies": []
        },
        {
            "author": "theHedgehogsDillemma",
            "body": "She\\\u2019s a selfish twat but you should be able to figure that out on your own.",
            "score": 1,
            "replies": []
        },
        {
            "author": "winterworld561",
            "body": "You are allergic to dogs, meaning you will get sick with the dog there. She was out of line expecting you to just 'suck it up'. Fuck her.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Plus_Ad_9181",
            "body": "Why are dog people like this? Your dog isn\\u2019t welcome unless explicitly invited. No that doesn\\u2019t include Walmart or other people\\\u2019s houses.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Classic-Patience-893",
            "body": "You have an allergy, seriously. Why did she think it was OK to just ignore that? She would deliberately make you sick and you consider ger a friend?? She's not your friend. A true friend would not do something that would make you ill on purpose.",
            "score": 1,
            "replies": []
        },
        {
            "author": "BrewDogDrinker",
            "body": "Nta.\n\nThat is not a true friend.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Amaranthim",
            "body": "Another one -\n\n**Your Text is Likely Human written, may include parts generated by AI/GPT**  \n63.46%  \nAI GPT\\*\n\n\\u00a0\\u00a0**Highlighted text is suspected to be most likely generated by AI\\***  \n**1,487 Characters**  \n**283 Words**\n\n  \nPlug it into [zerogpt.com](http://zerogpt.com) and see what you get!",
            "score": 1,
            "replies": []
        },
        {
            "author": "Vegetable-Cod-2340",
            "body": "NTA\n\n\\u2018Can\\u2019t you just suck it up?\\u2019\n\nNo, I need to be completely comfortable and able to breathe in my home that I PAY for, and I won\\u2019t be compromising my health, for you dog, that you didn\\u2019t inform me of before hand.",
            "score": 1,
            "replies": []
        },
        {
            "author": "SamuelVimesTrained",
            "body": "Even if you had a 6 bed 3 bath home\\u2026 why do \\u201cfriends\\u201d find no problems with compromising your health?\nThat are NOT friends.\n\nNTA",
            "score": 1,
            "replies": []
        },
        {
            "author": "MembershipKlutzy1476",
            "body": "your house, your rules.",
            "score": 1,
            "replies": []
        },
        {
            "author": "lexi_Xo31",
            "body": "NTA. She should be the one to suck it up",
            "score": 1,
            "replies": []
        },
        {
            "author": "Reasonable-Sale8611",
            "body": "Good for you for standing up to her when she put you on the spot. She was hoping that by showing up with the dog, you would feel like you had no other choice than to let the dog stay. Her behavior was manipulative, selfish, and wrong. Your other friends who feel you should have \"been more accommodating\" (i.e. allowed her dog to stay) are also wrong. They think you should have put your health second and allowed her to walk all over you like a doormat. Maybe they are just immature or maybe they aren't really your friends.",
            "score": 1,
            "replies": []
        },
        {
            "author": "Snoo_87531",
            "body": "YTA, this story is just too fake.",
            "score": 1,
            "replies": []
        },
        {
            "author": "TeaMistress",
            "body": "This post is AI-generated. Common signs of AI posts include:\n\n* The OP leaves the first comment immediately after the main post, adding context that should have been edited into the main post or offering \"anticipatory\" explanations for questions that haven't even been asked yet.\n* No OP engagement in the comments.\n* Frequent use of words and phrases in quotation marks throughout the post.\n* Using the phrases \"family helps family\", \"fast forward to now\", \"blowing up my phone\", \"my family/friends/coworkers are divided/split\"\n* Using em dashes to connect words.\n* Overly formal or stilted phrasing. Doesn't \"sound\" like a modern person wrote it.\n* Username sounds feminine (intended to be converted to a porn account)\n\nPlease downvote and report.",
            "score": 0,
            "replies": []
        }
    ]
}
        
    coalition_score = get_coalition_score(homogeneous_conversation)
    assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
    assert coalition_score < 0.5, "Homogeneous conversation should have a lower coalition score"



def test_coalition_score_polarized_conversation():
    """
    Test coalition score for a highly polarized conversation with little cross-group interaction.
    """
    polarized_conversation = {
        "title": "Should the government ban all fossil fuels?",
        "author": "EcoWarrior2024",
        "subreddit": "Politics",
        "comments": [
            {"author": "GreenFuture", "body": "Fossil fuels should be banned immediately. Climate change is real!", "score": 300},
            {"author": "ClimateDenier42", "body": "Banning fossil fuels would destroy our economy. This is fearmongering.", "score": 280},
            {"author": "EcoWarrior", "body": "If you care about the future, you’d support renewables over dirty energy.", "score": 250},
            {"author": "CoalLover", "body": "Renewables can’t replace coal. This is nonsense.", "score": 270},
            {"author": "Solar4Life", "body": "We can transition if we prioritize clean energy investment.", "score": 200},
            {"author": "DrillBabyDrill", "body": "Climate change is a hoax. Fossil fuels are the backbone of civilization.", "score": 310},
            {"author": "CarbonZero", "body": "Solar and wind are the future. We need a phase-out plan.", "score": 190},
            {"author": "BigOilFan", "body": "Oil and gas jobs support millions of families.", "score": 210},
            {"author": "EcoGeek", "body": "Investing in green energy creates more jobs than fossil fuels ever will.", "score": 180},
            {"author": "CoalDefender", "body": "China and India are still burning coal. Why should we stop first?", "score": 250},
            {"author": "SustainableNow", "body": "Because leadership matters. We need to set an example.", "score": 200},
            {"author": "OilTycoon", "body": "The economy will crash if we abandon fossil fuels too soon.", "score": 275},
            {"author": "NatureFan", "body": "Every year we delay, the damage gets worse.", "score": 190},
            {"author": "IndustryExpert", "body": "The solution is a gradual transition, not an immediate ban.", "score": 230},
            {"author": "RadicalGreens", "body": "Gradual isn’t enough. We need action *now*.", "score": 220},
        ]
    }
    coalition_score = get_coalition_score(polarized_conversation)
    assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
    assert coalition_score < 0.2, "Highly polarized conversations should have a low coalition score"

def test_coalition_score_unclear_opinion_groups():
    """
    Test coalition score for a discussion where users engage with opposing viewpoints constructively.
    """
    unclear_opinion_groups_conversation = {
        "title": "Should we have a universal basic income?",
        "author": "EconDebate",
        "subreddit": "Economics",
        "comments": [
            {"author": "UBIAdvocate", "body": "UBI would reduce poverty and increase economic security.", "score": 200},
            {"author": "MarketPurist", "body": "But wouldn't it disincentivize work? Maybe a targeted program would work better.", "score": 180},
            {"author": "NeutralThinker", "body": "What if we trialed UBI in select cities first to see its effects?", "score": 220},
            {"author": "UBIAdvocate", "body": "That’s a good idea. A small-scale experiment could show us the benefits.", "score": 190},
            {"author": "MarketPurist", "body": "Fair point. Maybe we should also consider the funding model for it.", "score": 170},
            {"author": "EconomistDave", "body": "Studies show UBI improves health outcomes and reduces crime.", "score": 210},
            {"author": "TaxpayerConcerned", "body": "Where does the money come from? Higher taxes?", "score": 200},
            {"author": "UBIRealist", "body": "We could fund it by reducing bureaucracy in welfare programs.", "score": 190},
            {"author": "SkepticJohn", "body": "Won’t people just quit their jobs and live off UBI?", "score": 180},
            {"author": "DataDriven", "body": "Evidence from pilot programs suggests that most people keep working.", "score": 190},
            {"author": "AndrewYangFan", "body": "UBI isn’t about making people lazy; it’s about stability.", "score": 250},
            {"author": "WorkEthicGuy", "body": "I like the idea, but we need accountability to prevent abuse.", "score": 210},
            {"author": "LibertarianJoe", "body": "Maybe UBI works, but government intervention is always messy.", "score": 170},
            {"author": "CompromiseCandidate", "body": "How about a hybrid model with incentives for work?", "score": 200},
            {"author": "UBIAdvocate", "body": "That’s a reasonable compromise. It’s worth testing.", "score": 190},
        ]
    }
    coalition_score = get_coalition_score(unclear_opinion_groups_conversation)
    assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
    assert coalition_score < 0.3, "Discussions that don't have strong coalitions initially should have lower scores."

def test_coalition_score_echo_chamber():
    """
    Test coalition score for an echo chamber where there is little to no disagreement.
    """
    echo_chamber = {
        "title": "Should billionaires be taxed at 90%?",
        "author": "TaxTheRich",
        "subreddit": "Socialism",
        "comments": [
            {"author": "Marxist123", "body": "Yes, billionaires hoard wealth and exploit workers.", "score": 400},
            {"author": "LeninFan", "body": "Absolutely. Redistribution is the only way forward.", "score": 350},
            {"author": "FairnessFirst", "body": "We need a wealth cap. No one needs more than $10M.", "score": 375},
            {"author": "DownWithCEOs", "body": "Billionaires shouldn't exist. Full stop.", "score": 390},
            {"author": "EatTheRich", "body": "Tax them at 100%. They don’t deserve it.", "score": 380},
            {"author": "RevolutionNow", "body": "The rich are parasites. Time for change.", "score": 360},
            {"author": "PowerToPeople", "body": "We should nationalize major industries.", "score": 340},
            {"author": "SocialistJake", "body": "Economic equality is more important than profit.", "score": 370},
            {"author": "WealthRedistributor", "body": "Luxury taxes on the rich should be even higher.", "score": 380},
            {"author": "RedFlag", "body": "The proletariat must rise!", "score": 310},
            {"author": "MarxWasRight", "body": "Capitalism is inherently exploitative.", "score": 390},
            {"author": "NoMoreBillionaires", "body": "There is no ethical way to become a billionaire.", "score": 375},
            {"author": "ProTax", "body": "A progressive tax is the only fair system.", "score": 350},
            {"author": "AntiGreed", "body": "We must hold the ultra-rich accountable.", "score": 360},
            {"author": "TaxThe1Percent", "body": "The rich must pay their fair share.", "score": 370},
        ]
    }
    coalition_score = get_coalition_score(echo_chamber)
    assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
    assert coalition_score < 0.1, "Echo chambers should have an extremely low coalition score"

def test_coalition_score_mixed_engagement():
    """
    Test coalition score for a conversation where some users engage across coalitions but others do not.
    """
    mixed_engagement= {
        "title": "Should social media platforms ban political ads?",
        "author": "PolicyDebater",
        "subreddit": "Technology",
        "comments": [
            {"author": "FreeSpeechMax", "body": "No, banning political ads is censorship!", "score": 220},
            {"author": "RegulateAds", "body": "Yes, political ads are full of misinformation.", "score": 210},
            {"author": "NeutralObserver", "body": "Maybe they should just require fact-checking instead?", "score": 180},
            {"author": "BigTechSkeptic", "body": "Social media companies have too much power already.", "score": 230},
            {"author": "FactCheckAdvocate", "body": "Platforms should label misleading ads, not ban them.", "score": 190},
            {"author": "BanAllAds", "body": "They should ban all ads, not just political ones!", "score": 200},
            {"author": "PoliticalFreedom", "body": "Voters should decide for themselves, not big tech.", "score": 250},
            {"author": "CivicDuty", "body": "But what if misleading ads influence elections unfairly?", "score": 240},
            {"author": "CompromiseFinder", "body": "Maybe ads should be allowed but with strict rules?", "score": 220},
            {"author": "MediaWatcher", "body": "A complete ban could have unintended consequences.", "score": 170},
            {"author": "DemocracyFirst", "body": "Transparency in ad funding is more important than bans.", "score": 180},
            {"author": "BanThemNow", "body": "Political ads do more harm than good. Just get rid of them.", "score": 260},
            {"author": "FreeMarketJoe", "body": "If you don’t like them, don’t watch them. Simple.", "score": 210},
            {"author": "AIRegulator", "body": "AI should monitor political ads for false claims.", "score": 190},
            {"author": "PoliticalRealist", "body": "Regulating ads is tricky, but some oversight is necessary.", "score": 200},
        ]
    }
    coalition_score = get_coalition_score(mixed_engagement)
    assert 0.0 <= coalition_score <= 1.0, "Coalition score should be between 0 and 1"
    assert 0.35 < coalition_score < 0.65, "Mixed engagement should have a moderate coalition score"

def test_coalition_score_small_discussion():
    """
    Test coalition score when there are very few comments.
    """
    small_discussion = {
        "title": "Is pineapple on pizza good?",
        "author": "FoodDebate",
        "subreddit": "Food",
        "comments": [
            {"author": "PineappleLover", "body": "Yes! Sweet and savory is the best combination.", "score": 100},
            {"author": "HateFruitOnPizza", "body": "Disgusting. Pineapple doesn’t belong on pizza.", "score": 90},
        ]
    }
    coalition_score = get_coalition_score(small_discussion)
    assert pd.isna(coalition_score), "Coalition score should be NaN for small discussions that can not be clustered meaningfully."


