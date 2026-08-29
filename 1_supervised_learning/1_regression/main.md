
# Regression

- Regression = a type of supervised learning problem where the goal is to predict a continuous numerical value from input features.
- MSE (mean squared error) = a function $MSE(\theta) = \frac{1}{n} \sum_{j=1}^{n} (y^{(j)} - \hat{y}^{(j)})^2$ (note: $\hat{y}$ depends on $\theta$)
- least squares = a general optimization problem of minimizing the sum of squared residuals (equivalently minimizing MSE, modulo a factor of 1/n).
- OLS (ordinary least squares): an unweighted version of least-squares where each observation contributes equally