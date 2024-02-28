from sklearn import ensemble
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def get_model(model_name, random_state=42):
    return {
        "linear_regression": LinearRegression(),

        "decision_tree_gini": tree.DecisionTreeRegressor(
            criterion="gini",
            random_state=random_state
        ),
        "decision_tree_entrophy": tree.DecisionTreeRegressor(
            criterion="entrophy",
            random_state=random_state
        ),
        "rf": ensemble.RandomForestRegressor(random_state=random_state),
        "svr_linear": SVR(
            kernel="linear"
        ),
        "svr": SVR(
            kernel="rbf"
        ),
    }[model_name]