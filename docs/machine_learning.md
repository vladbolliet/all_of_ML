# Machine learning

- Machine learning = the field of AI where computers learn patterns from data to make predictions or decisions
- **model** = a function $f(x, \theta)$, $\theta \in \mathbb{R}^q$ (parameters), with $q \in \mathbb{N}$ depending on the model ; "finding a model" = finding $\theta$

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