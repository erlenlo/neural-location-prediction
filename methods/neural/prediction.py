import argparse
import pickle
import numpy as np
import csv
import tensorflow as tf

from dateutil import parser
from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from geopy.distance import geodesic

from neural.partial.preprocess import TEXT_COLUMN, POSTING_TIME_COLUMN, USERNAME_COLUMN, USER_LANGUAGE_COLUMN, USER_DESCRIPTION_COLUMN
from neural.partial.preprocess import preprocess_text, preprocess_posting_time, get_labels

from neural.single.preprocess import DATA_COLUMNS, MAX_TWEET_LENGTH
from neural.single.preprocess import read_dataset, read_paris_dataset, read_london_dataset, join_attributes, one_hot_encode
from neural.single.grid import read_grid_cells

tf.logging.set_verbosity(tf.logging.ERROR)


def preprocess_partial(dataset, tokenizer, grid=True):
    text_encoding = preprocess_text(dataset[[TEXT_COLUMN, USERNAME_COLUMN, USER_LANGUAGE_COLUMN, USER_DESCRIPTION_COLUMN]].values, tokenizer)
    posting_time_encoding = preprocess_posting_time(dataset[POSTING_TIME_COLUMN])    

    target_column = 'gcid' if grid else 'place_name'
    targets = one_hot_encode(dataset[target_column])

    coordinates = dataset[['latitude', 'longitude']].values

    return [text_encoding, posting_time_encoding], targets, coordinates


def preprocess(dataset, tokenizer, grid=True):
    if 'created_at' in DATA_COLUMNS:
        dataset['created_at'] = dataset['created_at'].apply(lambda x: parser.parse(x).hour)

    tweets = dataset[DATA_COLUMNS].values.tolist()
    tweets = join_attributes(tweets)
    tweets = tokenizer.texts_to_sequences(tweets)
    tweets = pad_sequences(tweets, maxlen=MAX_TWEET_LENGTH, padding='post')

    target_column = 'gcid' if grid else 'place_name'
    targets = one_hot_encode(dataset['gcid'])

    coordinates = dataset[['latitude', 'longitude']].values

    return tweets, targets, coordinates


def predict_test_set(model, dataset_training, dataset, tokenizer, model_name, grid_path, grid=True, partial=False):
    if partial:
        data, targets, coordinates = preprocess_partial(dataset, tokenizer, grid=grid)
    else:
        data, targets, coordinates = preprocess(dataset, tokenizer, grid=grid)

    labels = read_grid_cells(grid_path)

    correct = 0
    errorDistances = []

    predictions = model.predict(data)

    for index, pred in enumerate(predictions):
        result_index = np.argmax(pred)
        # predicted_coord = labels[result_index].center()
        predicted_coord = centroid(find_coordinates_by_gci(dataset_training, result_index))
        target_coord = tuple(coordinates[index])
        try:
            errorDistance = geodesic(target_coord, predicted_coord).meters
            errorDistances.append(errorDistance)

            if targets[index][result_index] == 1:
                correct += 1
        except ValueError:
            print(result_index)

    accuracy = correct / dataset.shape[0]
    acc_d_05 = acc_d(errorDistances, 500)
    acc_d_1 = acc_d(errorDistances, 1000)
    acc_d_2 = acc_d(errorDistances, 2000)

    output = {
        'modelName': model_name,
        'medianErrorDistanceInMeters': np.median(errorDistances),
        'meanErrorDistanceInMeters': np.mean(errorDistances),
        'Accuracy': accuracy,
        'Acc@0.5km': acc_d_05,
        'Acc@1.0km': acc_d_1,
        'Acc@2.0km': acc_d_2
    }

    for i in output:
        print(f'{i}: {output[i]}')

    return output


def find_coordinates_by_gci(dataset, gcid):
    filtered = dataset[dataset['gcid'] == gcid]
    return filtered[['latitude', 'longitude']].values


def centroid(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


def acc_d(distances, threshold):
    count = sum(d <= threshold for d in distances)
    return count / len(distances)


def load_tokenizer(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def arguments():
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument(
        "-p",
        dest="partial",
        type=bool,
        default=False,
        help="Predict partial model",
    )
    return parser.parse_args()


def write_csv_from_dict(filename, data, header):
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for item in data:
            writer.writerow(item)


GRID = True
CITY = 'london'
GRID_TYPE = 'adaptive'
CELL_SIZE = ''

if __name__ == '__main__':

    args = arguments()

    training_set = read_london_dataset(f'./datasets/grid/{GRID_TYPE}/{CITY}/tweets.tsv', grid=GRID)
    test_set = read_london_dataset(f'./datasets/grid/{GRID_TYPE}/{CITY}/tweets_test.tsv', grid=GRID)
        
    grid_path = f'./datasets/grid/{GRID_TYPE}/{CITY}/grid.tsv'

    model_path = f'./neural/models'
    tokenizer = load_tokenizer(f'{model_path}/tokenizer.pickle')

    model_name = f'classifier-{2}-{512}-lstm-{1}-{128}-dense-{0.0001}-lr.h5'
    model = load_model(f'{model_path}/{model_name}')

    predict_test_set(model, training_set, test_set, tokenizer, model_name, grid_path, grid=GRID)

                                
