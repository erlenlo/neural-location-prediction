from TwitterAPI import TwitterAPI
from utils import write_tweet_to_csv
import csv
import json
import time

BOUDING_BOXES = {
    'world': '-180,-90,180,90',
    'norway': '4.79,58.03,12.16,64.9',
    'oslo': '10.3781,59.8193,11.1334,60.0312',
    'new_york': '-74,40,-73,41',
    'san_francisco': '-122.75,36.8,-121.75,37.8',
    'london': '-0.408517,51.3473219,0.201708,51.6673219',
    'manchester': '-2.345808,53.428913,-2.146656,53.536998',
    'liverpool': '-3.094251,53.372417,-2.8951,53.480646',
}

CITY = 'london'
API_TYPE = "standard"

SEARCH_TERM = 'python'
LANGUAGE = 'en'
PRODUCT = '30day'
ENV = "Development"

PREMIUM_URI = f'tweets/search/{PRODUCT}/{ENV}'
STANDARD_URI = 'statuses/filter'

def open_twitter_request():
    with open('./secrets.json') as file:
        keys = json.load(file)

    api = TwitterAPI(
        keys['consumer_key'],
        keys['consumer_secret'],
        keys['access_token'],
        keys['access_token_secret'],
    )

    return api.request(PREMIUM_URI if API_TYPE == 'premium' else STANDARD_URI,
                       {
                        #    'lang': LANGUAGE,
                           'point_radius': '[51.509865 -0.118092 15km]',
                           'has:geo': True,
                           'is:retweet': False,
                           'tweet_mode': 'extended',
                       }
                       if API_TYPE == 'premium' else {
                           # 'lang': LANGUAGE,
                           'locations': BOUDING_BOXES[CITY],
                       })


if __name__ == '__main__':
    next = ''
    while True:
        try:
            request = open_twitter_request()

            if request.status_code != 200:
                break

            for tweet in request:
                if tweet.get('geo'):
                    if tweet.get('extended_tweet') or not tweet.get('truncated'):
                        print(tweet['text'].replace('\n', ' ').replace('\r', ''))
                        write_tweet_to_csv(tweet, f'./data/tweets_stream_{CITY}.tsv')

            json = request.json()
            if 'next' not in json:
                break

            next = json['next']
        except Exception as e:
            print(e)
            print("Connection reset failed.")
            time.sleep(60 * 15)
            request = open_twitter_request()
