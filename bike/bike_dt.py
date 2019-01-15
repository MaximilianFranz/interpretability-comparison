import lime
import sklearn
import numpy as np
import sklearn.model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from graph_export import export_tree

from load_bike_data import load_bike

X, Y, names = load_bike()

train, test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(train, y_train)
print(y_test)
print(tree.predict(test))
print('MSE: ', mean_squared_error(y_test, tree.predict(test)))
export_tree(tree, '../export/bike/tree.pdf', names)
