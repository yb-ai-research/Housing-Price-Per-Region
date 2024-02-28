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


def display_scores(scores):
    print(pd.Series(scores).describe())


def get_y(df):
    return df["median_house_value"]


def get_X(df, should_fit, pipeline):
    df = df.drop(labels=["median_house_value", "kfold"], axis="columns")
    if should_fit:
        res = pipeline.fit_transform(df)
    else:
        res = pipeline.transform(df)
    # print(f"Features after preprocessing: {pipeline.get_feature_names_out()}")
    return res


def train(fold_name, model, df_train, df_valid):
    model = get_model(model)
    df_train = df_train.dropna()
    feature_pipeline = preprocessing.get_preprocessing_pipeline()
    y_train = get_y(df_train)
    X_train = get_X(df_train, should_fit=True, pipeline=feature_pipeline)

    df_valid = df_valid.dropna()
    y_valid = get_y(df_valid)
    X_valid = get_X(df_valid, should_fit=False, pipeline=feature_pipeline)

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = metrics.mean_squared_error(y_valid, preds, squared=False)
    print(f"Fold={fold_name}, RMSE={rmse}")

    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold_name}_{fold_name}.bin")
    )
    return rmse


def main(args):
    df = pd.read_csv(args.path)
    model = args.model

    if args.fold_mode == "manual":
        scores = []

        for kfold in tqdm(df["kfold"].unique()):
            df_train = df[df.kfold != kfold].reset_index(drop=True)
            df_valid = df[df.kfold == kfold].reset_index(drop=True)
            score = train(kfold, model, df_train, df_valid)
            scores.append(score)
        print("Printing final scores...")
        display_scores(scores)

    else:
        X = get_X(df, should_fit=True, pipeline=preprocessing.get_preprocessing_pipeline())
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
    )

    args = parser.parse_args()
    main(args)

