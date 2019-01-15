"""
Use Bayesian Rules as interpretable by design

Run with Python2 environment venv2
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sbrl.RuleListClassifier import *

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = iris.target
train, test, labels_train, labels_test = train_test_split(data, target, train_size=0.80)

print iris.feature_names
print iris.target_names

#setosa or not setosa classificaiton
labels_train = (labels_train+1)/2
labels_train = 1 - labels_train
# new_labels = []
# for label in labels_train:
#     if label == 1:
#         new_labels.append(0)
#     elif label == 2:
#         new_labels.append(1)
#     else:
#         new_labels.append(label)
#
# new_labels = np.array(new_labels)


model = RuleListClassifier(max_iter=10000, class1label="setosa", verbose=False)
model.fit(train, labels_train, feature_labels=iris.feature_names)

print model
