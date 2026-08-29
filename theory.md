# Machine learning

- Machine learning = the field of AI where computers learn patterns from data to make predictions or decisions
- **model** = a function $f(x, \theta)$, $\theta \in \mathbb{R}^q$ (parameters), with $q \in \mathbb{N}$ depending on the model ; "finding a model" = finding $\theta$
## Supervised learning

- Supervised learning = a type of machine learning problem where the goal is to learn a mapping from inputs to known target outputs using labeled training data
- **dataset**: $\mathcal{D}=\left\{\left(x^{(j)},y^{(j)}\right)\;\middle|\;j\in\{1,\ldots,n\}\right\},\qquad x^{(j)}\in\mathbb{R}^{p} (input),\quad y^{(j)}\in\mathbb{R} (output/target)$, p = number of input features, n = number of samples
- **prediction** = $\hat{y} = f(x, \theta)$, for a specific x
- residual = $r^{(j)} = y^{(j)}-\hat{y}^{(j)}$ 
- **loss function** = a function $l(\hat{y}, y)$ measuring how wrong a model's prediction is compared to the true value
- **training** = the process of finding  $\theta$ that minimizes the loss on the training data ; "model fits well to the training data" = model's predictions $\hat{y}$ are close to $y$ (accross the training set)
- "generalisation" = the ability of the learned model to make accurate predictions on **unseen data** drawn from the same underlying distribution
- goal: find the model that best fits to the training data while also generalizing well (to unseen data)
### Regression

- Regression = a type of supervised learning problem where the goal is to predict a continuous numerical value from input features.
- MSE (mean squared error) = a function $MSE(\theta) = \frac{1}{n} \sum_{j=1}^{n} (y^{(j)} - \hat{y}^{(j)})^2$ (note: $\hat{y}$ depends on $\theta$)
- least squares = a general optimization problem of minimizing the sum of squared residuals (equivalently minimizing MSE, modulo a factor of 1/n).
- OLS (ordinary least squares): an unweighted version of least-squares where each observation contributes equally
#### Linear regression

- Linear regression = a method for solving a regression problem where we assume that the target value is a linear combination of the input features
- linear regression model = a function $f(x, \beta) = \beta_0 + \sum_{i=1}^{p}{\beta_i x_i}$  , where $\beta = (\beta_i)_{i \in \{0,\ldots,p\}} \in \mathbb{R}^{p+1} (parameters), \beta_0 = bias$
- goal: find $\beta^* =argmin_{\beta\in\mathbb{R}^{p+1}} MSE(\beta)$ [[#^8ea73f]]
- vector form of problem:
	- We note $\beta=\begin{pmatrix}\beta_0\\\vdots\\\beta_p\end{pmatrix},\quad y=\begin{pmatrix}y^{(1)}\\\vdots\\y^{(n)}\end{pmatrix},\quad X=\begin{pmatrix}1&x_1^{(1)}&\cdots&x_p^{(1)}\\\vdots&\vdots&\ddots&\vdots\\1&x_1^{(n)}&\cdots&x_p^{(n)}\end{pmatrix}$
	- Now we have: $y = X \beta$ and $L(\beta) = MSE(\beta) = \frac{1}{n}||y-X\beta||_2^2$  
	- These matrix notations encode the entire dataset: whereas the previous equations described a single observation, the matrix notation simply collects all observations row by row into one matrix.
- two ways to find it:
	1. analytical/normal equation (exact solution): $\beta^{*}=(X^T X)^{-1}X^T y$
		- proof: [[#^72e02d]]
	2. gradient descent
		- an iterative optimization algorithm for finding $\theta$ that minimizes the loss
		- how it works: 
			1. initialise $\theta$ 
			2. choose $\alpha > 0$ (learning rate) (choosing it too large can cause divergence, and too low causes slow convergence)
			3. update $\theta$ as follows: $\theta_{n+1} = \theta_n - \alpha \nabla_{\theta}L(\theta_n)$ 
			4. stop after a number of iterations or until a stopping criterion is satisfied (for example, $||\nabla L(\theta_n)||_2 < \epsilon$ , for a given $\epsilon$)
		- for $L = MSE$, we have $\nabla_{\beta} L(\beta) = \frac{2}{n} X^T (X\beta-y)$ (proof: TODO)



# Refs
- $argmin_{x \in A} f(x) = \{x\in A \mid \forall a \in A, f(x) \leq f(a)\}$  (the x's which minimize f) ^8ea73f
- proof of the normal equation for linear regression: ^72e02d
	- to find $\beta^*$ , we simply need to find the minimum of $L(\beta)$ by solving $\frac{\partial}{\partial \beta} L(\beta) = 0$
	- $\frac{\partial}{\partial \beta} L(\beta) = \frac{\partial}{\partial \beta}\frac{1}{n}\|y-X\beta\|_2^2=\frac{\partial}{\partial \beta}\frac{1}{n}(y-X\beta)^T(y-X\beta)=\frac{\partial}{\partial \beta}\frac{1}{n}(y^T-\beta^TX^T)(y-X\beta)=\frac{\partial}{\partial \beta}\frac{1}{n}\left(y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta\right)=\frac{1}{n}\frac{\partial}{\partial \beta}\left(y^Ty-2y^TX\beta+\beta^TX^TX\beta\right)$
		- $\frac{\partial}{\partial \beta}y^Ty=0$
		- $\frac{\partial}{\partial \beta}(-2y^TX\beta)=-2\frac{\partial}{\partial \beta}(y^TX\beta)=^1-2(y^TX)^T=-2X^Ty$ (ref 1: [[#^ca43a4]])
		- $\frac{\partial}{\partial \beta}\beta^TX^TX\beta=2X^TX\beta$ (ref: [[#^4fad0c]])
	- So $\frac{\partial}{\partial \beta} L(\beta) =\frac{1}{n} (-2X^T y + 2 X^T X \beta) = 0 \Rightarrow X^T X \beta = X^T y$, so if (X^T * X) is invertible, then $\beta = (X^T X)^{-1} X^T y$ 
	- sub-proof refs:
		- Sizes: $y^T: 1 \times n\quad \land \quad X:n \times p \quad \Rightarrow \quad y^T X : 1 \times (p+1).$ We note $a = (a_0,\cdots,a_p) = y^T X : 1 \times (p+1)$. We have  $y^T X \beta = a \beta = (a_0,\cdots,a_p) \cdot \begin{pmatrix}\beta_0 \\ \vdots \\ \beta_p \end{pmatrix} = \sum_{i=0}^{p} a_i \beta_i$ . Therefore, $\frac{\partial}{\partial \beta}a\beta=\begin{pmatrix}\frac{\partial}{\partial \beta_0}\sum_{i=0}^{p} a_i \beta_i\\\vdots\\\frac{\partial}{\partial \beta_p}\sum_{i=0}^{p} a_i \beta_i\end{pmatrix}=\begin{pmatrix}a_0\\\vdots\\a_p\end{pmatrix}=a^T$ . ^ca43a4
		- Let $a = X^T X$. We have $\beta^T a \beta = \begin{pmatrix}\beta_0 & \cdots & \beta_p\end{pmatrix} \cdot \begin{pmatrix}a_{00} & \cdots & a_{0p} \\ \vdots & \vdots & \vdots \\ a_{p0} & \cdots & a_{pp}\end{pmatrix} \cdot \begin{pmatrix}\beta_0 \\ \vdots \\ \beta_p\end{pmatrix} = \sum_{i,j=0}^p \beta_i \ a_{ij} \ \beta{j}$ . For $k \in \{0,\cdots,p\}$, we want to find $\frac{\partial}{\partial \beta_k}\sum_{i,j=0}^p \beta_i \ a_{ij} \ \beta{j}$. We do for i,j != k, i=k and j!=k, i!=k and j=k, i=k and j=k, and we find $\frac{\partial}{\partial \beta_k}\sum_{i,j=0}^p \beta_i \ a_{ij} \ \beta{j} = \sum_{j=0}^{p} a_{kj} \beta_j + \sum_{i=0}^{p} a_{ik} \beta_i = (A\beta)_k + (A^T\beta)_k=((A+A^T)\beta)_k$     ^4fad0c

# Concepts

```

│   ├── Information Theory
│   │   ├── Entropy
│   │   ├── Cross-entropy
│   │   ├── KL divergence
│   │   └── Mutual information
│   │
│   └── Optimization
│       ├── Objective functions
│       ├── Gradient descent
│       ├── Stochastic / mini-batch gradient descent
│       ├── Momentum
│       ├── AdaGrad / RMSProp
│       ├── Adam
│       └── Regularization & constraints
│
├── 1. THE CORE MACHINE-LEARNING FRAMEWORK
│   │
│   ├── Data
│   ├── Model / hypothesis
│   ├── Parameters
│   ├── Predictions
│   ├── Loss / objective function
│   ├── Optimization
│   ├── Generalization
│   ├── Bias–variance tradeoff
│   └── Regularization
│
├── 2. SUPERVISED LEARNING
│   │
│   ├── Regression
│   │   │
│   │   ├── Linear Regression
│   │   │   ├── Ordinary least squares
│   │   │   ├── Normal equation
│   │   │   └── Gradient descent
│   │   │
│   │   ├── Polynomial Regression
│   │   ├── Ridge Regression
│   │   ├── Lasso Regression
│   │   ├── Elastic Net
│   │   ├── Support Vector Regression
│   │   ├── k-Nearest Neighbors Regression
│   │   ├── Decision Tree Regression
│   │   ├── Random Forest Regression
│   │   └── Gradient-Boosted Regression
│   │
│   └── Classification
│       │
│       ├── Logistic Regression
│       ├── Naive Bayes
│       │   ├── Gaussian
│       │   ├── Multinomial
│       │   └── Bernoulli
│       ├── k-Nearest Neighbors
│       ├── Support Vector Machines
│       ├── Decision Trees
│       ├── Random Forests
│       ├── AdaBoost
│       ├── Gradient Boosting
│       └── XGBoost-style Boosting
│
├── 3. UNSUPERVISED LEARNING
│   │
│   ├── Clustering
│   │   ├── K-Means
│   │   ├── K-Medoids
│   │   ├── Hierarchical Clustering
│   │   ├── DBSCAN
│   │   ├── Gaussian Mixture Models
│   │   └── Expectation-Maximization
│   │
│   ├── Dimensionality Reduction
│   │   ├── PCA
│   │   ├── Kernel PCA
│   │   ├── t-SNE
│   │   └── UMAP
│   │
│   ├── Density Estimation
│   │   ├── Histograms
│   │   ├── Kernel Density Estimation
│   │   └── Gaussian Mixture Models
│   │
│   └── Anomaly Detection
│       ├── Statistical methods
│       ├── Isolation Forest
│       └── One-Class SVM
│
├── 4. NEURAL NETWORKS & DEEP LEARNING
│   │
│   ├── The Artificial Neuron
│   │   ├── Weighted sum
│   │   ├── Bias
│   │   └── Activation functions
│   │
│   ├── Feedforward Neural Networks
│   │   ├── Perceptron
│   │   ├── Multi-layer perceptron
│   │   └── Universal approximation
│   │
│   ├── Training Neural Networks
│   │   ├── Forward propagation
│   │   ├── Loss functions
│   │   ├── Computational graphs
│   │   ├── Backpropagation
│   │   ├── Chain rule
│   │   └── Gradient-based optimization
│   │
│   ├── Regularization
│   │   ├── L1 / L2
│   │   ├── Dropout
│   │   ├── Batch normalization
│   │   └── Early stopping
│   │
│   ├── Convolutional Neural Networks
│   │   ├── Convolution
│   │   ├── Filters / kernels
│   │   ├── Feature maps
│   │   └── Pooling
│   │
│   ├── Sequential Models
│   │   ├── RNNs
│   │   ├── Vanishing / exploding gradients
│   │   ├── LSTMs
│   │   └── GRUs
│   │
│   ├── Attention
│   │   ├── Query
│   │   ├── Key
│   │   ├── Value
│   │   ├── Dot-product attention
│   │   └── Softmax
│   │
│   └── Transformers
│       ├── Self-attention
│       ├── Multi-head attention
│       ├── Positional encoding
│       ├── Encoder
│       ├── Decoder
│       └── Large language models
│
├── 5. GENERATIVE MACHINE LEARNING
│   │
│   ├── Probabilistic Generative Models
│   ├── Autoregressive Models
│   ├── Autoencoders
│   │   ├── Encoder / decoder
│   │   └── Latent space
│   ├── Variational Autoencoders
│   │   ├── Variational inference
│   │   ├── Reparameterization trick
│   │   └── KL divergence
│   ├── Generative Adversarial Networks
│   │   ├── Generator
│   │   ├── Discriminator
│   │   └── Minimax optimization
│   └── Diffusion Models
│       ├── Forward diffusion
│       ├── Noise distributions
│       ├── Reverse process
│       └── Denoising / score learning
│
├── 6. REINFORCEMENT LEARNING
│   │
│   ├── Markov Decision Processes
│   │   ├── States
│   │   ├── Actions
│   │   ├── Rewards
│   │   └── Transition probabilities
│   │
│   ├── Value Functions
│   ├── Bellman Equations
│   ├── Dynamic Programming
│   │   ├── Policy evaluation
│   │   ├── Policy iteration
│   │   └── Value iteration
│   ├── Monte Carlo Methods
│   ├── Temporal-Difference Learning
│   ├── SARSA
│   ├── Q-Learning
│   ├── Deep Q-Networks
│   ├── Policy Gradients
│   ├── Actor-Critic
│   └── Modern Deep RL
│
└── 7. THE BIG PICTURE
    │
    ├── Geometry
    │   └── KNN, SVM, PCA, neural representations
    │
    ├── Probability
    │   └── Naive Bayes, GMMs, VAEs, diffusion, RL
    │
    ├── Linear Algebra
    │   └── Regression, PCA, neural networks, transformers
    │
    ├── Calculus
    │   └── Gradient descent, backpropagation, policy gradients
    │
    ├── Information Theory
    │   └── Decision trees, cross-entropy, VAEs
    │
    └── Optimization
        └── The mathematical engine connecting almost everything
```