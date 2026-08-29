
# Supervised learning

- Supervised learning = a type of machine learning problem where the goal is to learn a mapping from inputs to known target outputs using labeled training data
- **dataset**: $\mathcal{D}=\left\{\left(x^{(j)},y^{(j)}\right)\;\middle|\;j\in\{1,\ldots,n\}\right\},\qquad x^{(j)}\in\mathbb{R}^{p} (input),\quad y^{(j)}\in\mathbb{R} (output/target)$, p = number of input features, n = number of samples
- **prediction** = $\hat{y} = f(x, \theta)$, for a specific x
- residual = $r^{(j)} = y^{(j)}-\hat{y}^{(j)}$ 
- **loss function** = a function $l(\hat{y}, y)$ measuring how wrong a model's prediction is compared to the true value
- **training** = the process of finding  $\theta$ that minimizes the loss on the training data ; "model fits well to the training data" = model's predictions $\hat{y}$ are close to $y$ (accross the training set)
- "generalisation" = the ability of the learned model to make accurate predictions on **unseen data** drawn from the same underlying distribution
- goal: find the model that best fits to the training data while also generalizing well (to unseen data)