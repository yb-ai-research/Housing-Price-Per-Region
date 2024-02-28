import os
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import argparse
from model_fetcher import get_model
import config
from tqdm import tqdm
from sklearn import metrics
import joblib
import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline


def display_scores(scores):
    print(pd.Series(scores).describe())


def get_y(df):
    return df["median_house_value"]


def get_X(df):
    return df.drop(labels=["median_house_value", "kfold"], axis="columns")


def train(fold_name, model, df_train, df_valid, select_from_model):
    model = get_model(model)

    df_train = df_train.dropna()
    feature_pipeline = preprocessing.get_preprocessing_pipeline()

    y_train = get_y(df_train)
    X_train = get_X(df_train)

    y_valid = get_y(df_valid)
    X_valid = get_X(df_valid)

    if select_from_model:
        model_selector = SelectFromModel(model)
        pipeline = make_pipeline(
            feature_pipeline,
            model_selector,
            model
        )
    else:
        pipeline = make_pipeline(
            feature_pipeline,
            model
        )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_valid)
    rmse = metrics.mean_squared_error(y_valid, preds, squared=False)
    print(f"Fold={fold_name}, RMSE={rmse}")

    joblib.dump(
        pipeline,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold_name}_{fold_name}.bin")
    )
    return rmse


def main(args):
    df = pd.read_csv(args.path)
    model = args.model
    select_from_model = args.select_from_model

    if args.fold_mode == "manual":
        scores = []

        for kfold in tqdm(df["kfold"].unique()):
            df_train = df[df.kfold != kfold].reset_index(drop=True)
            df_valid = df[df.kfold == kfold].reset_index(drop=True)
            score = train(kfold, model, df_train, df_valid, select_from_model)
            scores.append(score)
        print("Printing final scores...")
        display_scores(scores)

    else:
        X = get_X(df)
        y = get_y(df)
        scores = cross_val_score(get_model(model), X=X, y=y,
                                 scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        display_scores(rmse_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="linear_regression"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=config.TRAINING_STRATIFIED_KFOLD_FILE
    )
    parser.add_argument(
        "--fold_mode",
        type=str,
        default="manual"
    ),
    parser.add_argument(
        "--select_from_model",
        action="store_true"
    )

    args = parser.parse_args()
    main(args)

