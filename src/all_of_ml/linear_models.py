import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from time import perf_counter as pc



def analytical_linear_regression(X_train, y_train, X_test, y_test):
    """Fit linear regression using the analytical least-squares solution."""

    print("[INFO] Calculating parameters with analytical solution... ", end = "")
    start = pc()

    b = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train # or can use np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)

    end = pc()
    time = end - start
    print("Done in", time)
    print("[INFO] Assessing accuracy of analytical solution... ", end="")
    y_pred = X_test @ b
    err = MSE(y_test, y_pred)
    print("Error: ", err)

    return b

def sklearn_linear_regression(X_train, y_train, X_test, y_test):
    """Fit linear regression using sklearn."""

    print("[INFO] Calculating parameters and assessing accuracy of skicit-learn's linear regressor... ", end="")
    start = pc()

    reg = LinearRegression().fit(X_train, y_train)

    end = pc()
    print("Done in: ", end-start, " ; ", end="")
    err = MSE(y_test, reg.predict(X_test))
    print("Error: ", err)