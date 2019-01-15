import sklearn
from sklearn import tree
from sklearn.datasets import load_iris
import sklearn.ensemble
import sklearn.metrics
from graph_export import export_tree

iris = sklearn.datasets.load_iris()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)

tree = tree.DecisionTreeClassifier(max_depth=3)
tree.fit(train, labels_train)
feature_names = iris.feature_names
export_tree(tree, 'export/tree.pdf', feature_names)
