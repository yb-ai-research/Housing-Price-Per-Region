from sklearn import ensemble
from sklearn import tree

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entrophy": tree.DecisionTreeClassifier(
        criterion="entrophy"
    ),
    "rf": ensemble.RandomForestClassifier(),
}


def get_model(model_name):
    return models[model_name]