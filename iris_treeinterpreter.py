
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import load_iris
from treeinterpreter import treeinterpreter as ti

iris = sklearn.datasets.load_iris()

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80, random_state=142)


# fit a scikit-learn's classifier model
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

INST = 0

prediction, bias, contributions = ti.predict(rf, test[INST].reshape(1,-1))
contributions = contributions[0]

num_class = rf.predict(test[INST].reshape(1,-1))

print(num_class)
print(contributions)

# Analyse feature contribution for class 0 - setosa
for c, feature in sorted(zip(contributions[:, num_class], 
                                 iris.feature_names), 
                             key=lambda x: -abs(x[0])):
        print (feature, c)

