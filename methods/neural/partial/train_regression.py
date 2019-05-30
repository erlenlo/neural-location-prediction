import numpy as np

from keras.models import load_model
from keras.layers import Dense, concatenate, BatchNormalization
from keras import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from neural.single.preprocess import read_london_dataset
from neural.partial.preprocess import load_tokenizer
from neural.partial.partial_models import build_text_model, build_attributes_model


def train_regression(dataset, n_lstm, n_dense, num_lstm, num_dense, batch_size, epochs, max_length):

    sets = training_and_test_sets(dataset, test_share=0.05, regression=True)
    tokenizer = load_tokenizer('meta_regression')

    y_train = sets['y_train']
    y_test = sets['y_test']

    lat_max, lon_max = y_train.max(axis=0)
    y_train[:, 0] *= 1 / lat_max
    y_train[:, 1] *= 1 / lon_max
    y_test[:, 0] *= 1 / lat_max
    y_test[:, 1] *= 1 / lon_max

    num_labels = 2

    tknzr_dim = len(tokenizer.word_index) + 1
    text_model = build_text_model(n_lstm, n_dense, num_lstm, num_dense, num_labels, max_length, tknzr_dim, classify=False)
    categories_model = build_attributes_model(sets['time_train'], n_dense, num_labels, classify=False)

    merged = concatenate([attributes_model.output, text_model.output])

    merged = BatchNormalization()(merged)

    merged = Dense(n_dense, activation='relu')(merged)

    predictions = Dense(num_labels, activation='sigmoid', name="predictions")(merged)

    model = Model(inputs=[text_model.input]+attributes_model.input, outputs=predictions)

    opt = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae'])

    model.fit([sets['text_train'], sets['time_train']], y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=2,
              validation_data=([sets['text_test'], sets['time_test']], y_test)
              )

    model.save('{MODEL_PATH}/model_meta_regression.h5')


NUM_LSTM = 2
NUM_DENSE = 1
N_LSTM = 512
N_DENSE = 128

BATCH_SIZE = 64
EPOCHS = 50

MAX_TWEET_LENGTH = 30

MODEL_PATH = './neural/models/partial/regression'

if __name__ == '__main__':
    dataset = read_london_dataset('./datasets/grid/adaptive/london/tweets.tsv', grid=True)
    train_regression(dataset, NUM_LSTM, NUM_DENSE, N_LSTM, N_DENSE, BATCH_SIZE, EPOCHS, MAX_TWEET_LENGTH)
