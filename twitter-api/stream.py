from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
import numpy as np
import json
import csv

from utils import write_tweet_to_csv

BOUDING_BOXES = {
    'world': [-180,-90,180,90],
    'norway': [4.79,58.03,12.16,64.9],
    'oslo': [10.3781,59.8193,11.1334,60.0312],
    'new_york': [-74,40,-73,41],
    'san_francisco': [-122.75,36.8,-121.75,37.8],
    'london': [-0.408517,51.3473219,0.201708,51.6673219],
    'manchester': [-2.345808,53.428913,-2.146656,53.536998],
    'liverpool': [-3.094251,53.372417,-2.8951,53.480646],
}

CITY = 'london'

class StdOutListener(StreamListener):
    def __init__(self):
        self.tweet_count = 0

    def on_data(self, data):
        tweet = json.loads(data)
        if tweet.get('geo'):
            if tweet.get('extended_tweet') or not tweet.get('truncated'):
                print(tweet['text'].replace('\n', ' ').replace('\r', ''))
                write_tweet_to_csv(tweet, './data/streaming.tsv')
        return True

    def on_error(self, status):
        print(status)
        return True

    def on_timeout(self):
        print('Timeout...')
        return True # Don't kill the stream


if __name__ == '__main__':
    with open('./secrets.json') as file:
        keys = json.load(file)

    # This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
    auth.set_access_token(keys['access_token'], keys['access_token_secret'])
    api = API(auth)

    stream = Stream(auth, l)
    stream.filter(locations=BOUDING_BOXES[CITY])

    # search_results = api.search(geocode='51.509865,-0.118092,15km', tweet_mode='extended', count=100)
    # for item in search_results:
    #     tweet = item._json
    #     if tweet.get('geo'):
    #         if tweet.get('extended_tweet') or not tweet.get('truncated'):
    #             print(tweet['text'].replace('\n', ' ').replace('\r', ''))
    #             write_tweet_to_csv(tweet, './data/streaming.tsv')
