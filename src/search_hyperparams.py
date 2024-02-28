from sklearn.model_selection import GridSearchCV
import argparse
import preprocessing
import model_fetcher
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import config


def get_y(df):
    return df["median_house_value"]


def get_X(df):
    return df.drop(labels=["median_house_value", "kfold"], axis="columns")


def search():
    model = model_fetcher.get_model("rf")
    pipeline = Pipeline([
        ("preprocessing", preprocessing.get_preprocessing_pipeline()),
        ("random_forest", model)
    ])
    param_grid = [
        {
            'preprocessing__geo__n_clusters': [5, 8, 10],
            'random_forest__max_features': [4, 6, 8]
        },
        {
            'preprocessing__geo__n_clusters': [10, 15],
            'random_forest__max_features': [6, 8, 10]
        }
    ]
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
    df = pd.read_csv(config.TRAINING_STRATIFIED_KFOLD_FILE)
    X = get_X(df)
    y = get_y(df)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

    cv_res = pd.DataFrame(grid_search.cv_results_)
    cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
    print(cv_res.head())
    print(cv_res)


if __name__ == "__main__":
    search()

