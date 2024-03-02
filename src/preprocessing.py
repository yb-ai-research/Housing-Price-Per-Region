from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn import impute
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


class ClassSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.kmeans_ = None

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} Similarity" for i in range(self.n_clusters)]


class MedianNeighborPrices(MetaEstimatorMixin, BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, p=2, weights="auto"):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.regressor = None

    def fit(self, X, y=None, sample_weight=None):
        self.regressor = KNeighborsRegressor(
            n_neighbors = self.n_neighbors,
            p = self.p,
            weights = self.weights
        )
        self.regressor.fit(X, y)
        return self

    def transform(self, X):
        return np.median(self.regressor.kneighbors(X))

    def get_feature_names_out(self, names=None):
        return [f"Median of {self.n_neighbors} Neighbors"]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def get_default_num_pipeline():
    return make_pipeline(
        impute.SimpleImputer(strategy="median"),
        StandardScaler()
    )


def get_ratio_feature_name(_, feature_names_in):
    return ["ratio_of_" + "_".join(feature_names_in)]  # feature names out


def pipeline_add_ratio():
    return make_pipeline(
        impute.SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=get_ratio_feature_name),
        StandardScaler()
    )


def pipeline_category():
    return make_pipeline(
        impute.SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )


def pipeline_log():
    return make_pipeline(
        impute.SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )


def get_preprocessing_pipeline():
    transformer = ColumnTransformer([
        ("bedroom", pipeline_add_ratio(), ["total_bedrooms", "total_rooms"]),
        ("people_per_house", pipeline_add_ratio(), ["population", "households"]),
        ("rooms_per_house", pipeline_add_ratio(), ["total_rooms", "households"]),
        ("log", pipeline_log(), ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", ClassSimilarity(), ["latitude", "longitude"]),
        # ("neighbor_median", MedianNeighborPrices()),
        ("category", pipeline_category(), make_column_selector(dtype_include=object)),
    ], remainder=get_default_num_pipeline())

    return transformer


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../datasets/housing/housing_train_stratified_kfold.csv")
    transformer = FunctionTransformer(column_ratio)
    preprocessing = get_preprocessing_pipeline()
    res = preprocessing.fit_transform(df)
    print(res)
    print(res.shape)
    print(preprocessing.get_feature_names_out())
