
import panadas as pd
import numpy as np

def load_bike():
    """
    Returns bike sharing dataset as
        X, relevant attributes
        Y, target count of bikes
        names, of the attributes
    """
    data = pd.read_csv('data/bike_sharing/day.csv')
    Y = np.array(data.pop('cnt')) # target is count of bikes
    names = data.columns
    
    # Remove meta-data and irrelevant information
    data.drop(['instant', 'dteday', 'registered', 'casual', 'atemp'])
    X = np.array(data)

    return X, Y, names
