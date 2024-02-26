from sklearn import model_selection
import pandas as pd

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


if __name__ == "__main__":
    df = pd.read_csv("../datasets/housing/housing.csv")
    df = create_folds(df, labels=None, n_splits=10)
    df.to_csv("../datasets/housing/housing_folds.csv", index=False)
