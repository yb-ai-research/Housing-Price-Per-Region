from sklearn.model_selection import train_test_split
import data_loader
import pandas as pd
import config
import numpy as np

housing = data_loader.load_house_data()
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

strat_train_set.to_csv(config.TRAINING_FILE, index=False)
strat_test_set.to_csv(config.TEST_FILE, index=False)