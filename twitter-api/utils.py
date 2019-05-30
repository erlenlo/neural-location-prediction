import pandas as pd
import numpy
import csv


def write_tweet_to_csv(tweet, file_name):
    tweet_obj = {
        'id_str': tweet['id_str'],
        'created_at': tweet['created_at'],
        'lang': tweet['lang'],
        'latitude': tweet['geo']['coordinates'][0] if tweet['geo'] else '',
        'longitude': tweet['geo']['coordinates'][1] if tweet['geo'] else '',
        'place': tweet['place']['full_name'] if tweet['place'] else '',
        'retweet': tweet['retweet_count'] != 0,
        'friends_count': tweet['user']['friends_count'],
        'followers_count': tweet['user']['followers_count'],
        'username': tweet['user']['name'],
        'user_screen_name': tweet['user']['screen_name'],
        'user_time_zone': tweet['user']['time_zone'],
        'user_utc_offset': tweet['user']['utc_offset'],
        'user_lang': tweet['user']['lang'],
        'user_description': tweet['user']['description'].replace('\n', ' ').replace('\r', '') if tweet['user']['description'] else '',
    }
    if tweet.get('full_text'):
        tweet_obj['text'] = tweet['full_text'].replace('\n', ' ').replace('\r', '')
    elif tweet.get('extended_tweet'):
        tweet_obj['text'] = tweet['extended_tweet']['full_text'].replace('\n', ' ').replace('\r', '')
    else:
        tweet_obj['text'] = tweet['text'].replace('\n', ' ').replace('\r', '')

    with open(file_name, encoding="utf-8", newline='', mode='a') as file:
        writer = csv.writer(file, delimiter ='\t')
        writer.writerow(list(tweet_obj.values()))
