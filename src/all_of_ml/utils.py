import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from time import perf_counter as pc




def load_chd():
    """Fetch, prepare and return the california housing dataset"""

    print("[INFO] Fetching and preprocessing data... ", end="")
    start = pc()

    d = fetch_california_housing(as_frame=True)
    X = d.data
    y = d.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    X_test = np.column_stack((np.ones(len(X_test)), X_test))

    end = pc()
    print("Done in: ", end-start)
    return X_train, X_test, y_train, y_test

