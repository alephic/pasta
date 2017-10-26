import praw
import json
import sys

with open('reddit_account.json') as f:
  account = json.load(f)
reddit = praw.Reddit(client_id='vd-xnw2Bk85jUg', client_secret='UAI2gPb5hzMyCI0n5VSFH2Jvkpc', password=account['password'], user_agent='scrubbs_scraper', username=account['username'])

subreddit = reddit.subreddit(sys.argv[1])

gen = subreddit.submissions()

textposts = []
print()
for sub in gen:
  if len(sub.selftext) > 0:
    textposts.append(sub.selftext)
    print("\rScraped %d posts" % len(textposts), end='')
filename = '%s.json' % sys.argv[1]
with open(filename, mode='w') as f:
  print("Dumping to %s" % filename)
  json.dump(textposts, f)
