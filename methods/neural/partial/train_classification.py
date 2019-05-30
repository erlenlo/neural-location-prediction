import numpy as np
from keras.models import load_model
from keras.layers import Dense, concatenate, BatchNormalization
from keras import Model
from keras.optimizers import Adam

from neural.single.preprocess import read_london_dataset
from neural.partial.preprocess import training_and_test_sets, load_tokenizer
from neural.partial.partial_models import build_text_model, build_attributes_model


def set_trainable_status(model, status):
    for layer in model.layers:
        layer.trainable = status


def train_classification(dataset, n_lstm, n_dense, num_lstm, num_dense, batch_size, epochs, max_length):

    sets = training_and_test_sets(dataset, test_share=0.05)
    tokenizer = load_tokenizer('meta')

    y_train = sets['y_train']
    y_test = sets['y_test']

    num_labels = y_train.shape[1]

    tknzr_dim = len(tokenizer.word_index) + 1
    text_model = build_text_model(n_lstm, n_dense, num_lstm, num_dense, num_labels, max_length, tknzr_dim, classify=True, trainable=False)
    categories_model = build_attributes_model(sets['time_train'], n_dense, num_labels, classify=True, trainable=False)

    merged = concatenate([categories_model.output, text_model.output])

    merged = BatchNormalization()(merged)

    merged = Dense(n_dense, activation='relu')(merged)

    predictions = Dense(num_labels, activation='softmax', name="predictions")(merged)

    model = Model(inputs=[text_model.input]+[categories_model.input], outputs=predictions)

    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    model.fit([sets['text_train'], sets['time_train']], y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=2,
              validation_data=([sets['text_test'], sets['time_test']], y_test)
              )

    model.save(f'{MODEL_PATH}/model_meta.h5')


NUM_LSTM = 2
NUM_DENSE = 1
N_LSTM = 512
N_DENSE = 128

BATCH_SIZE = 64
EPOCHS = 50

MAX_TWEET_LENGTH = 30

MODEL_PATH = './neural/models/multiple'

if __name__ == '__main__':
    dataset = read_london_dataset('./datasets/grid/adaptive/london/tweets.tsv', grid=True)
    train_classification(dataset, N_LSTM, N_DENSE, NUM_LSTM, NUM_DENSE, BATCH_SIZE, EPOCHS, MAX_TWEET_LENGTH)
