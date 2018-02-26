import csv
import tweepy
import numpy as np
from textblob import TextBlob

import sys
import jsonpickle
import os

import unicodedata

# credentials
consumer_key = 'Ob2LcJa53MA2MtjEXUi7LSLFn'
consumer_secret = 'ry4ZsCGt5wKdh3HfGkI7GtxAwPdRlaKaMJYgiP7Z9vOQvGmPm9'

access_token = '383543853-GAFnHNI9o5JWAUiUXlvE4vHdjjfwIJY0fdBndrsB'
access_token_secret = 'wsFMYvJOt2ia5ZhMIjz8wxb6pq26Zl72vgJGlmsCdzPZO'


auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
 
api = tweepy.API(auth, wait_on_rate_limit=True,
				 wait_on_rate_limit_notify=True)
 
if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

searchQuery = '@Maybelline'  # this is what we're searching for Maybelline New York (@Maybelline)
maxTweets = 15000 # Some arbitrary large number
tweetsPerQry = 15  # this is the max the API permits
since_date = "2017-01-01"

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1 #L

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))

with open('@Maybelline.csv', 'w') as csvfile:
    fieldnames = ['Name', 'Twitter Handle', 'Total Tweets', 'Favourites_Count', 'Followers', 'User_Id','User_Verified','User Location',
                 'Date of Tweet', 'Tweet Id', 'Tweet Text', 'Language', 'Tweet Source', 'Tweet Retweet', 'Tweet Reply To Id',
                 'Reply To Name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceId)
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                row = {}
                row['Name'] = str(tweet.user.name).encode("utf8")
                row['Twitter Handle'] = str(tweet.user.screen_name).encode("utf8")
                row['Total Tweets'] = str(tweet.user.statuses_count).encode("utf8")
                row['Favourites_Count'] = str(tweet.user.favourites_count).encode("utf8")
                row['Followers'] = str(tweet.user.followers_count).encode("utf8")
                row['User_Id'] = str(tweet.user.id).encode("utf8")
                row['User_Verified'] = str(tweet.user.verified).encode("utf8")
                row['User Location'] = str(tweet.user.location).encode("utf8")
                row['Date of Tweet'] = str(tweet.user.created_at).encode("utf8")
                row['Tweet Id'] = str(tweet.id).encode("utf8")
                row['Tweet Text'] = str(tweet.text).encode("utf8")
                row['Language'] = str(tweet.lang).encode("utf8")
                row['Tweet Source'] = str(tweet.source).encode("utf8")
                row['Tweet Retweet'] = str(tweet.retweet_count).encode("utf8")
                row['Tweet Reply To Id'] = str(tweet.in_reply_to_user_id).encode("utf8")
                row['Reply To Name'] = str(tweet.in_reply_to_screen_name).encode("utf8")
                writer.writerow(row)
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break

print ("Downloaded {0} tweets".format(tweetCount))