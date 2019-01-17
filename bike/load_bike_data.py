
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_bike():
    """
    Note that some values are normalized already and that we don't specifically
    encode categorical featuers, which isn't best practice. 

    Returns bike sharing dataset as
        X, relevant attributes
        Y, target count of bikes
        names, of the attributes
    """
    data = pd.read_csv('../data/bike_sharing/day.csv')
    Y = np.array(data.pop('cnt')) # target is count of bikes

    # Remove meta-data and irrelevant information
    data = data.drop(columns=['instant', 'dteday', 'registered', 'casual', 'atemp', 'yr'])
    names = data.columns
    X = np.array(data)

    return X, Y, names
