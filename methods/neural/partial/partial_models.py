import warnings
import time
import numpy as np
import pickle

import tensorflow as tf
from keras.initializers import Constant
from keras import Input, Model
from keras.layers import LSTM, Dense, Flatten, Dropout, Embedding, BatchNormalization, concatenate
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from neural.partial.preprocess import training_and_test_sets
from neural.partial.preprocess import MAX_TWEET_LENGTH


tf.logging.set_verbosity(tf.logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)


def build_text_model(n_lstm, n_dense, num_lstm, num_dense, num_labels, max_length, embedding_dim, embedding_size=100, trainable=True, classify=True):
    branchInput = Input(shape=(None,), name="textInput")

    branch = Embedding(embedding_dim, embedding_size, input_length=max_length)(branchInput)

    for _ in range(num_lstm - 1):
        branch = LSTM(n_lstm, return_sequences=True, dropout=0.4)(branch)
    branch = LSTM(n_lstm, dropout=0.4)(branch)
    branch = BatchNormalization()(branch)

    for _ in range(num_dense - 1):
        branch = Dense(n_dense, activation="relu")(branch)
        branch = Dropout(0.4)(branch)
        
    branch = Dense(n_dense, activation="relu")(branch)
    branch = Dropout(0.4, name="text")(branch)

    branchOutput = Dense(n_dense, activation="relu")(branch)

    if classify:
        branchOutput = Dense(num_labels, activation="softmax")(branch)

    model = Model(inputs=branchInput, outputs=branchOutput)

    if trainable:
        opt = Adam(lr=0.0001, decay=1e-6)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def build_attributes_model(posting_times, n_dense, num_labels, trainable=False, classify=True):
    posting_input = Input(shape=(posting_times.shape[1],))
    postingDense = Dense(n_dense, activation="relu")(posting_input)
    postingDense = Dropout(0.4)(postingDense)

    branchOutput = Dense(n_dense)(postingDense)

    if classify:
        branchOutput = Dense(num_labels, activation="softmax")(postingDense)
    
    model = Model(inputs=posting_input, outputs=branchOutput)
    
    if trainable:
        opt = Adam(lr=0.0001)
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
    
    return model


def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, model_name):
    tensorboard = TensorBoard(log_dir='./neural/logs/{}'.format(model_name))
    model.fit(
        x_train,
        y_train,
        verbose=2,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard],
    )
    model.save('./neural/models/multiple/{}.h5'.format(model_name))
