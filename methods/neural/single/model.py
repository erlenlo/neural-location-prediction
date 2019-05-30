import time
import numpy as np

from gensim.models import Word2Vec

import tensorflow as tf
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout, Embedding, Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from neural.single.preprocess import MAX_TWEET_LENGTH

tf.logging.set_verbosity(tf.logging.ERROR)

def build_model(
    n_lstm, n_dense, num_lstm, num_dense, lr, num_targets, max_length, embedding_dim, embedding_size=100, embedding=None
):
    model = Sequential()

    if embedding is not None:
        model.add(
            Embedding(
                embedding_dim,
                embedding_size,
                embeddings_initializer=Constant(embedding),
                input_length=max_length,
                trainable=False,
            )
        )
    else:
        model.add(Embedding(embedding_dim, embedding_size, input_length=max_length))


    for _ in range(num_lstm - 1):
        model.add(LSTM(n_lstm, return_sequences=True, dropout=0.4))

    model.add(LSTM(n_lstm, dropout=0.4))
    model.add(BatchNormalization())


    for _ in range(num_dense):
        model.add(Dense(n_dense, activation="relu"))
        model.add(Dropout(0.4))

    model.add(BatchNormalization())
    
    model.add(Dense(num_targets, activation="softmax"))

    opt = Adam(lr=lr, decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_model(
    model, x_train, y_train, x_test, y_test, batch_size, epochs, model_name
):
    tensorboard = TensorBoard(log_dir="./neural/logs/{model_name}".format(**locals()))
    model.fit(
        x_train,
        y_train,
        verbose=2,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard],
    )
    model.save("./neural/models/{}.h5".format(model_name))
