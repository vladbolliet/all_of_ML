import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from time import perf_counter as pc
from all_of_ml.utils import grad_MSE

def analytical_linear_regression(X_train, y_train, X_test, y_test):
    """
    Fit linear regression using the analytical least-squares solution.
    Returns: b (model)
    """

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
    """
    Fit linear regression using sklearn.
    Returns: reg
    """

    print("[INFO] Calculating parameters and assessing accuracy of skicit-learn's linear regressor... ", end="")
    start = pc()

    reg = LinearRegression().fit(X_train, y_train)

    end = pc()
    print("Done in: ", end-start, " ; ", end="")
    err = MSE(y_test, reg.predict(X_test))
    print("Error: ", err)
    return reg

def gd_linear_regression(X_train, y_train, X_test, y_test, beta, alpha, epsilon):
    """
    Linear regression with gradient descent.
    Parameters:
    - beta: the initial model
    - alpha: learning rate
    - epsilon: criterion for stopping the GD: $||nabla L(theta_n)||_2 < epsilon$
    Returns: b (model).
    """

    # validate alfa
    mu_max = np.linalg.eigvalsh(X_train.T @ X_train)[-1] # get largest eigenvalue
    print("[INFO] mu_max = ", mu_max)
    if alpha >= len(X_train) / mu_max or alpha <= 0:
        raise ValueError("alpha must be > 0 and < len(X_train) / mu_max")

    # gd
    start = pc()
    grad_norm = 999999
    while grad_norm >= epsilon:
        grad = grad_MSE(X_train, y_train, beta)
        beta -= alpha * grad
        grad_norm = grad.T @ grad
        print("Percentage of completion: ", epsilon/grad_norm * 100, "%", end='\r')
    end = pc()
    print("[INFO] Linear regression GD time: ", end-start, " ; ", end ="")

    # assess accuracy
    y_pred = X_test @ beta
    err = MSE(y_test, y_pred)
    print("Error: ", err)

    return beta
