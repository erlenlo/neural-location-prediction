import pandas as pd
import numpy as np
import csv
import time
import re
import pickle
import datetime
from dateutil import parser
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from neural.single.preprocess import timer, read_london_dataset, one_hot_encode

TEXT_COLUMN = 'tweet_text'
POSTING_TIME_COLUMN = 'created_at'
USERNAME_COLUMN = 'username'
USER_LANGUAGE_COLUMN = 'user_language'
USER_DESCRIPTION_COLUMN = 'user_description'

LABEL_COLUMN = 'gcid'

TEST_SHARE = 0.05
MAX_TWEET_LENGTH = 40


def get_labels(dataset):
    return sorted(dataset[LABEL_COLUMN].unique().tolist())


def one_hot(possible_values, values):
    one_hot = np.zeros((len(values), len(possible_values)))
    for index, val in enumerate(values):
        one_hot_index = possible_values.index(val)
        one_hot[index][one_hot_index] = 1
    return one_hot


@timer
def preprocess_text(texts, tokenizer=None):
    print('Preprocessing text')

    tweets_texts = join_attributes(texts)

    if tokenizer is None:
        tknzr = Tokenizer()
        tknzr.fit_on_texts(tweets_texts)
        with open('./neural/models/partial/tokenizer_meta.pickle', 'wb') as handle:
            pickle.dump(tknzr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        tknzr = tokenizer

    tweets_texts = tknzr.texts_to_sequences(tweets_texts)
    tweets_texts = pad_sequences(tweets_texts, maxlen=MAX_TWEET_LENGTH, value=0, padding='post')

    return tweets_texts


def join_attributes(tweets):
    return [tokenize(' '.join(tweet[:])) for tweet in tweets]

def tokenize(text):
    tweet_tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
    reg_tknzr = RegexpTokenizer(r'\w+')
    text = text.lower()
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove url
    tokens = tweet_tknzr.tokenize(text)
    tokens = reg_tknzr.tokenize(' '.join(tokens))
    return tokens


@timer
def preprocess_posting_time(posting_times):
    print('Preprocessing posting time')
    posting_times = posting_times.apply(lambda x: parser.parse(x).hour)
    hours = range(0, 24)
    return one_hot(hours, posting_times.values)


@timer
def training_and_test_sets(dataset, test_share=0.05, regression=False):
    text_encoding = preprocess_text(dataset[[TEXT_COLUMN, USERNAME_COLUMN, USER_LANGUAGE_COLUMN, USER_DESCRIPTION_COLUMN]].values)
    posting_time_encoding = preprocess_posting_time(dataset[POSTING_TIME_COLUMN])

    if regression:
        labels = dataset[['latitude', 'longitude']].values
    else:
        labels = one_hot_encode(dataset[LABEL_COLUMN])

    text_train, text_test, time_train, time_test, y_train, y_test = train_test_split(
        text_encoding, posting_time_encoding, labels, test_size=test_share, shuffle=True, stratify=labels)

    print(text_train.shape)
    print(text_test.shape)
    print(time_train.shape)
    print(time_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return {
        'text_train': text_train,
        'text_test': text_test,
        'time_train': time_train,
        'time_test': time_test,
        'y_train': y_train,
        'y_test': y_test,
    }


def load_tokenizer(model_type):
    with open('./neural/models/multiple/tokenizer_{}.pickle'.format(model_type), 'rb') as handle:
        return pickle.load(handle)
