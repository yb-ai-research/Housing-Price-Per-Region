from sklearn import model_selection
import pandas as pd
import config
import numpy as np
import argparse


RANDOM_SEED = 42


def create_folds(data, labels, n_splits: int, model="KFold"):
    data = data.sample(frac=1).reset_index(drop=True)

    if model == "KFold":
        kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    elif model == "StratifiedKFold":
        kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    else:
        raise RuntimeError(f"Invalid model={model}. Supported values=[KFold,StratifiedKFold]")
    for f, (t_, v_) in enumerate(kf.split(X=data, y=labels)):
        data.loc[v_, "kfold"] = f

    return data


def create_folds_and_persist(df, labels, dest_file, model, n_splits):
    df = create_folds(df, labels=labels, n_splits=n_splits, model=model)
    df = df.drop("income_cat", axis="columns")
    df.to_csv(dest_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mini",
        action="store_true"
    )
    args = parser.parse_args()
    if args.mini:
        n_splits = 3
        n_training_instances = 5000
    else:
        n_splits = 10
        n_training_instances = None

    df = pd.read_csv(config.TRAINING_FILE)
    if args.mini:
        stratified_fname = config.MINI_TRAINING_STRATIFIED_KFOLD_FILE
        regular_fname = config.MINI_TRAINING_KFOLD_FILE
        df = df[:n_training_instances]
    else:
        stratified_fname = config.TRAINING_STRATIFIED_KFOLD_FILE
        regular_fname = config.TRAINING_KFOLD_FILE

    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    create_folds_and_persist(df, df["income_cat"], stratified_fname, model="StratifiedKFold", n_splits=n_splits)
    create_folds_and_persist(df, None, regular_fname, model="KFold", n_splits=n_splits)
