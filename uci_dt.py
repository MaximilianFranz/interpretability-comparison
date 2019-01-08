import sklearn
from .. import load_data
from ..graph_export import export_tree
from sklearn.metrics import accuracy_score

X, Y = load_uci()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80)

tree = sklearn.tree.DecisionTreeClassifier(max_depth=2)
tree.fit(train, labels_train)
export_tree(tree, 'export/treeuci.pdf')

print(accuracy_score(labels_test, tree.predict(test)))
