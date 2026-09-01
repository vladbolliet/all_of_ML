import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import perf_counter as pc




def load_chd(normalize_data=False):
    """
    Fetch, prepare and return the california housing dataset.
    Returns: X_train, X_test, y_train, y_test
    """

    print("[INFO] Fetching and preprocessing data... ", end="")
    start = pc()

    d = fetch_california_housing(as_frame=True)
    X = d.data
    y = d.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    X_test = np.column_stack((np.ones(len(X_test)), X_test))

    if normalize_data:
        scaler = StandardScaler()
        # learn the mean and std deviation for each feature (except bias column)
        scaler.fit(X_train[:, 1:])
        # apply transformation to each column
        X_train[:, 1:] = scaler.transform(X_train[:, 1:])
        X_test[:, 1:] = scaler.transform(X_test[:, 1:])

    end = pc()
    print("Done in: ", end-start)
    return X_train, y_train, X_test, y_test

def grad_MSE(X_train, y_train, beta):
    """Returns the gradient of mean squared error."""
    return 2/len(X_train) * X_train.T @ (X_train @ beta - y_train)