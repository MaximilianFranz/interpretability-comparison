from sklearn.datasets import load_iris
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

iris = sklearn.datasets.load_iris()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)

clf = GradientBoostingRegressor(n_estimators=500).fit(train, labels_train)

names=iris.feature_names
print(iris.target_names)

label=iris.target_names[2]

fig, axs = plot_partial_dependence(clf, train, [(2,3)],
                                       feature_names=names,
                                       label=label,
                                       n_jobs=4,
                                       grid_resolution=50,
                                       n_cols=4)

fig.suptitle('Partial dependence of class ' + label + ' on petal-length and -width')
plt.subplots_adjust(top=0.9)
plt.show()
