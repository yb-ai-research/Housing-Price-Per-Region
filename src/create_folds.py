from sklearn import model_selection
import pandas as pd
import config
import numpy as np

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


def create_folds_and_persist(data, labels, dest_file, model):
    df = create_folds(data, labels=labels, n_splits=10, model=model)
    df = df.drop("income_cat", axis="columns")
    df.to_csv(dest_file, index=False)


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    create_folds_and_persist(df, df["income_cat"], config.TRAINING_STRATIFIED_KFOLD_FILE, model="StratifiedKFold")
    create_folds_and_persist(df, None, config.TRAINING_KFOLD_FILE, model="KFold")
