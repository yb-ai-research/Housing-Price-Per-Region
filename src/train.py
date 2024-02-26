import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import argparse
from model_fetcher import get_model
import config


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def train(args):
    df = pd.read_csv(args.path)
    model = get_model(args.model)
    df = df.dropna()
    labels = df["median_house_value"]
    print(labels)
    print(df.columns)
    df = df.drop(labels=["ocean_proximity"], axis="columns")

    scores = cross_val_score(model, X=df, y=labels,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str,
        default="decision_tree_gini"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=config.TRAINING_FILE
    )

    args = parser.parse_args()
    train(args)

