function formatRedditPostToHTML(redditData) {
  // Format the main post
  let html = `
    <div style="font-family: 'Helvetica', Arial, sans-serif; max-width: 800px; margin: 0 auto; color: #1a1a1b;">
      <!-- Post header -->
      <div style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; margin-bottom: 12px; border: 1px solid #edeff1;">
        <div style="font-weight: bold; font-size: 18px; margin-bottom: 6px; color: #222;">${redditData.title}</div>
        <div style="font-size: 12px; color: #787c7e; margin-bottom: 4px;">
          Posted in <span style="color: #0079d3;">r/${redditData.subreddit}</span> by 
          <span style="color: #0079d3;">u/${redditData.author}</span> &middot; 
          ${redditData.score} points &middot; ${redditData.num_comments} comments
        </div>
        <div style="font-size: 12px; color: #787c7e;">
          <span style="color: #46d160;">${(redditData.upvote_ratio * 100).toFixed(0)}% Upvoted</span>
        </div>
      </div>
      
      <!-- Post body -->
      <div style="margin-bottom: 24px; line-height: 1.5; font-size: 14px; white-space: pre-line;">${redditData.selftext}</div>
      
      <!-- Comments section -->
      <div style="border-left: 4px solid #e9ebee; padding-left: 8px;">
        <div style="font-size: 12px; color: #787c7e; margin-bottom: 12px; font-weight: bold;">
          ${redditData.num_comments} Comments
        </div>
  `;

  // Add each comment
  redditData.comments.forEach(comment => {
    html += formatComment(comment, 0);
  });

  html += `</div></div>`;
  return html;
}

function formatComment(comment, depth) {
  // Calculate indentation based on comment depth
  const marginLeft = depth * 16;
  const borderColor = depth % 2 === 0 ? '#e9ebee' : '#dae0e6';
  
  let html = `
    <div style="margin-bottom: 16px; margin-left: ${marginLeft}px;">
      <div style="border-left: ${depth > 0 ? '2px' : '0'} solid ${borderColor}; padding-left: ${depth > 0 ? '8px' : '0'};">
        <div style="font-size: 12px; color: #787c7e; margin-bottom: 4px;">
          <span style="color: #0079d3; font-weight: bold;">u/${comment.author}</span> &middot; 
          ${comment.score} points
        </div>
        <div style="margin-bottom: 8px; line-height: 1.4; font-size: 14px;">${comment.body}</div>
        <div style="font-size: 12px; color: #878a8c; margin-bottom: 8px;">
          <span style="margin-right: 8px;">\u2B06</span>
          <span style="margin-right: 8px;">\u2B07</span>
          <span style="margin-right: 8px;">Reply</span>
          <span>Share</span>
        </div>
  `;

  // Add replies if they exist
  if (comment.replies && comment.replies.length > 0) {
    comment.replies.forEach(reply => {
      html += formatComment(reply, depth + 1);
    });
  }

  html += `</div></div>`;
  return html;
}



const redditJson = {
  "title": "NASA astronauts splash down on Earth after 9 months stranded in space",
  "author": "f1sh98",
  "subreddit": "Conservative",
  "rank": 2,
  "score": 1093,
  "upvote_ratio": 0.93,
  "num_comments (reported by reddit)": 71,
  "url": "https://www.foxnews.com/video/5614615980001",
  "id": "1jehe6z",
  "selftext": "",
  "comments": [
      {
          "author": "DandierChip",
          "body": "Awesome news!",
          "score": 71,
          "replies": []
      },
      {
          "author": "KyleforUSA",
          "body": "Thank you Musk and space X.   Suck it Boeing.",
          "score": 294,
          "replies": [
              {
                  "author": "Batbuckleyourpants",
                  "body": "The last few years have not been kind to Boeing at all.",
                  "score": 20,
                  "replies": []
              }
          ]
      },
      {
          "author": "murderinthedark",
          "body": "Super happy they got back safe.  Now homie can get that crazy hair cut.  lol <3",
          "score": 66,
          "replies": []
      },
      {
          "author": "gittenlucky",
          "body": "It\u2019s funny seeing all the \u201cElon musk\u2019s ____ <does bad thing>\u201d all over Reddit leftist subs, then this happens and the posts are \u201castronauts return safe\u201d with no mention of musk\u2026",
          "score": 25,
          "replies": []
      },
      {
          "author": "Flare4roach",
          "body": "I\u2019m sure the Left will put aside politics to celebrate that Elon brought fellow Americans home after being abandoned.",
          "score": 146,
          "replies": [
              {
                  "author": "_AlexSupertramp_",
                  "body": "Lol",
                  "score": 54,
                  "replies": []
              },
              {
                  "author": "Drawer-Imaginary",
                  "body": "Too busy vandalizing teslas and then complaining about more expensive insurance prices to notice",
                  "score": 45,
                  "replies": []
              }
          ]
      },
      {
          "author": "roaming_art",
          "body": "USA, USA, USA!!!",
          "score": 40,
          "replies": []
      },
      {
          "author": "Sure-Wishbone-4293",
          "body": "Thank you to the Trump administration for taking care of this. \n\nThe Biden administration didn\u2019t do anything about this issue.",
          "score": 119,
          "replies": [
              {
                  "author": "D_Ethan_Bones",
                  "body": "They did negative, they blocked things from happening that could have happened without their petty interference. \n\nNow we get to see people folding their arms over their chests and going out of their way to *loudly* deny credit to Musk.",
                  "score": 66,
                  "replies": [
                      {
                          "author": "Sure-Wishbone-4293",
                          "body": "\ud83d\udc4d\ud83c\udffb yup!",
                          "score": 17,
                          "replies": []
                      }
                  ]
              }
          ]
      },
      {
          "author": "intrigue-bliss4331",
          "body": "Biden\u2019s regime left 2 Americans stranded because they hate Trump that much. Just realize there is no depth to which the left won\u2019t sink.",
          "score": 35,
          "replies": [
              {
                  "author": "hey_ringworm",
                  "body": "Didn\u2019t want Elon (who had endorsed Trump) to get the win before the election.\n\nThere is no low to which Biden/Democrats would not sink in order to retain power.",
                  "score": 13,
                  "replies": []
              }
          ]
      },
      {
          "author": "CombatDeffective",
          "body": "What? No! President Biden exiled them from Earth for a reason! (/s)",
          "score": 6,
          "replies": []
      },
      {
          "author": "Silly_Ad_4612",
          "body": "Unreal. Thank God. Crazy news.\u00a0",
          "score": 3,
          "replies": []
      },
      {
          "author": "Praising_God_777",
          "body": "Praising God that they\u2019re home! I\u2019ve been praying for months!",
          "score": 4,
          "replies": []
      },
      {
          "author": "None",
          "body": "[removed]",
          "score": 1,
          "replies": []
      },
      {
          "author": "BossJackson222",
          "body": "And liberals will still try to say that Elon Musk had nothing to do with this. And that it was racist somehow lol. Or some Nazi bullshit made up crap lol.",
          "score": 1,
          "replies": []
      }
  ]
}
const htmlOutput = formatRedditPostToHTML(redditJson);
console.log(htmlOutput); // This is the HTML you can paste into Qualtrics