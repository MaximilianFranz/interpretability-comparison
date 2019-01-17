
import sklearn
import shap
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_bike(file):
    """
    Note that some values are normalized already and that we don't specifically
    encode categorical featuers, which isn't best practice.

    Returns bike sharing dataset as
        X, relevant attributes
        Y, target count of bikes
        names, of the attributes
    """
    data = pd.read_csv(file)
    Y = np.array(data.pop('cnt')) # target is count of bikes

    # Remove meta-data and irrelevant information
    data = data.drop(columns=['instant', 'dteday', 'registered', 'casual', 'atemp', 'yr'])
    names = data.columns
    X = np.array(data)

    return X, Y, names

shap.initjs()

X, Y, names = load_bike('https://raw.githubusercontent.com/MaximilianFranz/interpretability-comparison/master/data/bike_sharing/day.csv')

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80)


rf = sklearn.ensemble.RandomForestRegressor(n_estimators=500)
rf.fit(train, labels_train)

explainer = shap.TreeExplainer(rf, train, feature_dependence="independent")
shap_values = explainer.shap_values(test)

print(shap_values)

ind = 0
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], test[ind,:],
    feature_names=names
)
