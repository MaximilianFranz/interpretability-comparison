from RuleListClassifier.RuleListClassifier import *
from sklearn.datasets.mldata import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()

feature_labels = iris.feature_names
y = iris.target # target labels (0 or 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(iris.data, y, train_size=0.7) # split

# train classifier (allow more iterations for better accuracy; use BigDataRuleListClassifier for large datasets)
model = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=False)
model.fit(Xtrain, ytrain, feature_labels=feature_labels)

print ("RuleListClassifier Accuracy:", model.score(Xtest, ytest), "Learned interpretable model:\n", model)
print ("RandomForestClassifier Accuracy:", RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest))
