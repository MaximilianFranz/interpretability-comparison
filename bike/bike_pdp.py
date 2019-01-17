from sklearn.datasets import load_iris
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

from load_bike_data import load_bike

X, Y, names = load_bike()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80, random_state=42)

clf = GradientBoostingRegressor(n_estimators=500).fit(train, labels_train)

print('feature names ', names)
fig, axs = plot_partial_dependence(clf, train, [6, 7, 8, 0],
                                   feature_names=names,
                                   n_jobs=4,
                                   grid_resolution=50,
                                   n_cols=4)

fig.suptitle('Partial dependence of regression task')
plt.subplots_adjust(top=0.9)
plt.show()
