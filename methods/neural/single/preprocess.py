import pandas as pd
import numpy as np
import csv
import time
import re
import pickle
import math
import string

from dateutil import parser

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


DATA_COLUMNS = ['tweet_text']
LABEL_COLUMN = 'gcid'

MAX_TWEET_LENGTH = 25
EMBEDDING_SIZE = 100


def timer(func):
    def wrap(*args, **kw):
        start = time.time()
        ret = func(*args, **kw)
        stop = time.time()
        print('{:s} function took {:.2f} s'.format(
            func.__name__, (stop-start)))
        return ret
    return wrap


@timer
def preprocess(dataset, word2vec_filename, tokenizer_name='tokenizer', grid=True, train_word2vec=True, test_share=0.2):    
    if 'created_at' in DATA_COLUMNS:
        dataset['created_at'] = dataset['created_at'].apply(lambda x: parser.parse(x).hour)

    data = join_attributes(dataset[DATA_COLUMNS].values)
    targets = one_hot_encode(dataset[LABEL_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=test_share, shuffle=False, stratify=None)

    vocabulary = x_train + x_test
    w2c_model = get_word2vec_model(word2vec_filename, vocabulary=vocabulary, train=train_word2vec)

    textTokenizer = Tokenizer()
    textTokenizer.fit_on_texts(vocabulary)

    # Save tokenizer for validation
    with open('./neural/models/{tokenizer_name}.pickle'.format(**locals()), 'wb') as handle:
        pickle.dump(textTokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    embedding_matrix = get_embedding_matrix(textTokenizer, w2c_model)

    x_train = textTokenizer.texts_to_sequences(x_train)
    x_test = textTokenizer.texts_to_sequences(x_test)

    x_train = pad_sequences(x_train, maxlen=MAX_TWEET_LENGTH, padding='post')
    x_test = pad_sequences(x_test, maxlen=MAX_TWEET_LENGTH, padding='post')

    return embedding_matrix, x_train, y_train, x_test, y_test


def get_word2vec_model(filename, vocabulary=None, train=True):
    if train:
        model = Word2Vec(vocabulary, min_count=1,
                         size=EMBEDDING_SIZE, workers=4)
        model.train(vocabulary, total_examples=len(vocabulary), epochs=10)
        model.save(filename)
    else:
        model = Word2Vec.load(filename)
    return model


def get_embedding(model, word):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros(EMBEDDING_SIZE)


def get_embedding_matrix(tokenizer, w2c_model):
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1

    embedding_matrix = np.zeros((num_words, EMBEDDING_SIZE))
    for word, index in word_index.items():
        if index > num_words:
            continue
        embedding_vector = get_embedding(w2c_model, word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


def join_attributes(tweets):
    tweet_tokens = [[str(item) for item in tweet[:-1]] + tokenize(' '.join(tweet[-1:])) for tweet in tweets]
    return tweet_tokens


def one_hot_encode(categories):
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    return onehot_encoder.fit_transform(np.array(categories).reshape(-1, 1))


def one_hot(targets, count):
    one_hot = np.zeros((len(targets), count))
    for index, target in enumerate(targets):
        one_hot[index][target] = 1
    return one_hot


def tokenize(text):
    tweet_tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
    reg_tknzr = RegexpTokenizer(r'\w+')
    text = text.lower()
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove url
    tokens = tweet_tknzr.tokenize(text)
    tokens = reg_tknzr.tokenize(' '.join(tokens))
    return tokens


def remove_stopwords(text):
    stop_words = stopwords.words('english')
    return [word.lower() for word in text if word not in stop_words]


@timer
def remove_rare_tokens(tweets, n=5):
    tokens = np.hstack(tweets)
    freq_dist = FreqDist(tokens)
    rare_tokens = []

    rare = 0
    non_rare = 0
    for key, value in freq_dist.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value < n:
            rare += 1
            rare_tokens.append(key)
        else:
            non_rare += 1
    print('Rare tokens: {}'.format(rare))
    print('Non rare tokens: {}'.format(non_rare))

    return [remove_rare_tokens_from_tweet(tweet, rare_tokens) for tweet in tweets]

def remove_rare_tokens_from_tweet(tweet, rare_tokens):
    return [token for token in tweet if token not in rare_tokens]


def find_average_tweet_length(filename):
    data = read_dataset(filename)
    tweets = data['tweet_text'].values.tolist()
    tweets = [tokenize(str(text)) for text in tweets]

    total = 0
    maximum = 0
    for t in tweets:
        total += len(t)
        if len(t) > maximum:
            maximum = len(t)

    print('Average: {}'.format(total / len(tweets)))
    print('Max: {}'.format(maximum))

    return total / len(tweets)


def read_dataset(filename, grid=False):
    columns = [
        "utc_time",
        "country",
        "country_code",
        "place_type",
        "place_name",
        "language",
        "username",
        "user_screen_name",
        "timezone_offset",
        "number_of_friends",
        "tweet_text",
        "latitude",
        "longitude",
    ]

    if grid:
        columns.insert(0, 'gcid')
    dataset = pd.read_csv(filename, sep="\t", names=columns)
    dataset = dataset.dropna(subset = ['tweet_text'])
    return dataset


def read_paris_dataset(filename, grid=False):
    columns = [
        "latitude",
        "longitude",
        'tweet_text'
    ]

    if grid:
        columns.insert(0, 'gcid')
    dataset = pd.read_csv(filename, sep="\t", names=columns)
    dataset = dataset.dropna(subset = ['tweet_text'])
    return dataset


def read_london_dataset(filename, grid=False):
    columns = [
        'id_str',
        'created_at',
        'language',
        'latitude',
        'longitude',
        'place',
        'retweet',
        'friends_count',
        'followers_count',
        'username',
        'user_screen_name',
        'user_time_zone',
        'user_utc_offset',
        'user_language',
        'user_description',
        'tweet_text',
    ]
    if grid:
        columns.insert(0, 'gcid')
    dataset = pd.read_csv(filename, sep="\t", names=columns)
    dataset = dataset.dropna(subset = ['tweet_text'])
    dataset['user_description'].fillna('', inplace=True)
    dataset['user_language'].fillna('', inplace=True)

    return dataset


def custom_train_test_split(dataset, training_share=0.95):
    gcids = dataset['gcid'].unique().tolist()
    
    tweets_train = []
    tweets_test = []

    for gcid in gcids:
        tweets_gci = dataset[dataset['gcid'] == gcid]
        if tweets_gci.shape[0] < 3:
            tweets_gci = pd.concat([tweets_gci for _ in range(3)])

        msk = np.random.rand(len(tweets_gci)) < training_share

        if all(x for x in msk):
            msk[-1] = False

        if all(not x for x in msk):
            msk[:2] = [True] * 2

        tweets_train.append(tweets_gci[msk])
        tweets_test.append(tweets_gci[~msk])

    tweets_train = shuffle(pd.concat(tweets_train))
    tweets_test = shuffle(pd.concat(tweets_test))

    return tweets_train, tweets_test

