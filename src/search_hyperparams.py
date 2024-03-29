from sklearn.model_selection import GridSearchCV
import argparse
import preprocessing
import model_fetcher
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import config
import argparse
from scipy.stats import randint, loguniform, expon


def get_y(df):
    return df["median_house_value"]


def get_X(df):
    return df.drop(labels=["median_house_value", "kfold"], axis="columns")


class ParamGrids:
    def __init__(self, name, model_name, param_grid, param_distribs):
        self.name = name
        self.param_grid = param_grid
        self.model_name = model_name
        self.param_distribs = param_distribs


def get_params(type):
    return {
        "random_forest": ParamGrids(
            name="random_forest",
            model_name="rf",
            param_grid=[
                {
                    'preprocessing__geo__n_clusters': [5, 8, 10],
                    'random_forest__max_features': [4, 6, 8]
                },
                {
                    'preprocessing__geo__n_clusters': [10, 15],
                    'random_forest__max_features': [6, 8, 10]
                }
            ],
            param_distribs={
                "preprocessing__geo__n_clusters": randint(low=3, high=50),
                "random_forest__max_features": randint(low=2, high=20),
            }
        ),
        "svr": ParamGrids(
            name="svr",
            model_name="svr",
            param_grid=[
                {
                    'svr__C': [0.01, 0.1, 1, 10, 100, 1000, 10000.0],
                    'svr__gamma': [0.01, 0.1, 1, 10]
                }
            ],
            param_distribs={
                "svr__C": loguniform(20, 10000),
                "svr__gamma": expon(scale=1.0),
            }
        ),
        "svr_linear": ParamGrids(
            name="svr_linear",
            model_name="svr_linear",
            param_grid=[
                {
                    'svr_linear__C': [0.01, 0.1, 1, 10, 100, 1000, 10000.0]
                }
            ],
            param_distribs={
                "svr_linear__C": loguniform(20, 10000),
            }
        )
    }[type]


def search(type):
    params = get_params(type)
    model = model_fetcher.get_model(params.model_name)
    pipeline = Pipeline([
        ("preprocessing", preprocessing.get_preprocessing_pipeline()),
        (params.name, model)
    ])
    param_grid = params.param_grid

    random_search = RandomizedSearchCV(
        pipeline,
        params.param_distribs,
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42
    )
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
    df = pd.read_csv(config.MINI_TRAINING_STRATIFIED_KFOLD_FILE)
    # df = pd.read_csv(config.TRAINING_STRATIFIED_KFOLD_FILE)
    X = get_X(df)
    y = get_y(df)
    random_search.fit(X, y)
    print(-random_search.best_score_)
    print(random_search.best_params_)
    cv_res = pd.DataFrame(random_search.cv_results_)
    cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
    print(cv_res.head())
    print(cv_res)

    grid_search.fit(X, y)
    print(-grid_search.best_score_)
    print(grid_search.best_params_)

    cv_res = pd.DataFrame(grid_search.cv_results_)
    cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
    print(cv_res.head())
    print(cv_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="random_forest"
    )
    args = parser.parse_args()
    search(args.type)


