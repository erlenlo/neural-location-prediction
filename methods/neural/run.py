import time
import argparse

from neural.single.model import build_model, train_model
from neural.single.preprocess import preprocess, read_dataset, read_paris_dataset, read_london_dataset, MAX_TWEET_LENGTH


def create_arguments():
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument(
        "-d",
        dest="dataset_path",
        type=str,
        default="datasets/tweets.tsv",
        help="path to dataset",
    )
    parser.add_argument(
        "-w",
        dest="word2vec_path",
        type=str,
        default="neural/models/w2v.model",
        help="path to word2vec model",
    )
    parser.add_argument(
        "-e",
        dest="epochs",
        type=int,
        default=50,
        help="number of epochs (neural network model)",
    )
    parser.add_argument(
        "-g",
        dest="grid",
        type=bool,
        default=True,
        help="Grid cells as target",
    )
    parser.add_argument(
        "--bs",
        dest="batch_size",
        type=int,
        default=64,
        help="batch size (neural network model)",
    )
    parser.add_argument(
        "--nl",
        dest="num_lstm",
        type=int,
        default=2,
        help="number of lstm layers (neural network model)",
    )
    parser.add_argument(
        "--nd",
        dest="num_dense",
        type=int,
        default=1,
        help="number of dense layers (neural network model)",
    )
    parser.add_argument(
        "--sl",
        dest="n_lstm",
        type=int,
        default=512,
        help="size of lstm (neural network model)",
    )
    parser.add_argument(
        "--sd",
        dest="n_dense",
        type=int,
        default=128,
        help="size of dense (neural network model)",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.0001,
        help="learning rate (neural network model)",
    )
    parser.add_argument(
        "--es",
        dest="embedding_size",
        type=int,
        default=100,
        help="embedding size (neural network model)",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = create_arguments()

    print("Reading dataset: {args.dataset_path} ...".format(**locals()))
    dataset = read_london_dataset(args.dataset_path, grid=args.grid)

    print("Preprocessing data...")
    embedding_matrix, x_train, y_train, x_test, y_test = preprocess(
        dataset, args.word2vec_path, grid=args.grid, test_share=0.05)

    print("Training classifier...")
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model_name = "classifier-{args.num_lstm}-{args.n_lstm}-lstm-{args.num_dense}-{args.n_dense}-dense-{args.lr}-lr".format(**locals())
    model = build_model(
        args.n_lstm,
        args.n_dense,
        args.num_lstm,
        args.num_dense,
        args.lr,
        y_train.shape[1],
        MAX_TWEET_LENGTH,
        embedding_matrix.shape[0],
        args.embedding_size)

    train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        args.batch_size,
        args.epochs,
        model_name,
    )
