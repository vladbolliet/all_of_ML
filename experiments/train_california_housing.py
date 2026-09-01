import numpy as np
from all_of_ml.utils import load_chd
from all_of_ml.linear_models import analytical_linear_regression, sklearn_linear_regression, gd_linear_regression


def main():
    X_train, y_train, X_test, y_test = load_chd(normalize_data=True)
    analytical_solution = analytical_linear_regression(X_train, y_train, X_test, y_test)
    sklean_solution = sklearn_linear_regression(X_train, y_train, X_test, y_test)
    gd_solution = gd_linear_regression(X_train, y_train, X_test, y_test, np.zeros(X_train.shape[1]), pow(10, -3), pow(10,-3))

if __name__ == "__main__":
    main()