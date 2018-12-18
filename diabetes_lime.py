import lime
import sklearn
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import load_iris
from lime.lime_tabular import LimeTabularExplainer

data = pd.read_csv('data/dataset_diabetes/diabetic_data.csv')

target = data.pop('readmitted')
print(target)
feature_names = data.columns

print(data.head())
