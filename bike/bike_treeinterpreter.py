import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sklearn.linear_model
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_iris
from treeinterpreter import treeinterpreter as ti
from load_bike_data import load_bike

X, Y, names = load_bike()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80, random_state=42)

log = sklearn.linear_model.LogisticRegression()
log.fit(train, labels_train)

print('LR - MSE', mean_squared_error(labels_test, log.predict(test

# fit a scikit-learn's classifier model
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=500)
rf.fit(train, labels_train)
print('global importances: ', sorted(list(zip(rf.feature_importances_, names))))

INST = 0

prediction, bias, contributions = ti.predict(rf, test[INST].reshape(1,-1))
contributions = contributions[0]

print(contributions)
print('features: ', test[INST])
print('feature_names: ', names)
# Analyse feature contribution for class 0 - setosa
for c, feature in sorted(zip(contributions,
                                 names),
                             key=lambda x: -abs(x[0])):
        print (feature, c)
print(bias)
print(prediction)
