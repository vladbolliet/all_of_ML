
- Linear regression = a method for solving a regression problem where we assume that the target value is a linear combination of the input features
- linear regression model = a function $f(x, \beta) = \beta_0 + \sum_{i=1}^{p}{\beta_i x_i}$  , where $\beta = (\beta_i)_{i \in \{0,\ldots,p\}} \in \mathbb{R}^{p+1} (parameters), \beta_0 = bias$
- goal: find $\beta^* =argmin_{\beta\in\mathbb{R}^{p+1}} MSE(\beta)$ [[#^381afb]]
- vector form of problem:
	- We note $\beta=\begin{pmatrix}\beta_0\\\vdots\\\beta_p\end{pmatrix},\quad y=\begin{pmatrix}y^{(1)}\\\vdots\\y^{(n)}\end{pmatrix},\quad X=\begin{pmatrix}1&x_1^{(1)}&\cdots&x_p^{(1)}\\\vdots&\vdots&\ddots&\vdots\\1&x_1^{(n)}&\cdots&x_p^{(n)}\end{pmatrix}$
	- Now we have: $y = X \beta$ and $L(\beta) = MSE(\beta) = \frac{1}{n}||y-X\beta||_2^2$  
	- These matrix notations encode the entire dataset: whereas the previous equations described a single observation, the matrix notation simply collects all observations row by row into one matrix.
- two ways to find it:
	1. analytical/normal equation (exact solution): $\beta^{*}=(X^T X)^{-1}X^T y$
		- proof: [[#^fc4d76]]
		- This will give the best linear model for the given features and training dataset (under squared-error loss)
	2. gradient descent
		- an iterative optimization algorithm for finding $\theta$ that minimizes the loss
		- how it works: 
			1. initialise $\theta$ 
			2. choose $\alpha > 0$ (learning rate) (choosing it too large can cause divergence, and too low causes slow convergence ; $\alpha$ needs to be $< \frac{N}{||X||^2_2}$ where $||X||^2_2$ means the largest eigenvalue of $X^T X$ for having convergence (see proof))
			3. update $\theta$ as follows: $\theta_{n+1} = \theta_n - \alpha \nabla_{\theta}L(\theta_n)$ 
			4. stop after a number of iterations or until a stopping criterion is satisfied (for example, $||\nabla L(\theta_n)||_2 < \epsilon$ , for a given $\epsilon$)
		- for $L = MSE$, we have $\nabla_{\beta} L(\beta) = \frac{2}{n} X^T (X\beta-y)$
		- proof that $\nabla_{\beta} L(\beta) = \frac{2}{n} X^T (X\beta-y)$ and that gradient descent converges to $\beta^*$ (or to a solution of $(X^T X) \beta^* = X^T y$, if $X^T X$ is not invertible): [[#^e053c8 |here]]
		- when doing gradient descent, we often normalize our data.
			- "normalize data" = put the input features on roughly the same numerical scale / have them not too far from each other
			- why do we normalize ? 
				1. if one feature has huge values and the other one has tiny values, their contributions to the gradient will be wildly different, and the GD will take huge steps in a direction and small in another direction, and will therefore move in zigzags, instead of moving efficiently towards the minimum
				2. without normalisation, $||X||^2_2$ can be very big and therefore $\alpha$ needs to be very small to attain convergence
			- normalized data always needs to be reversible to its initial state
			- a common approach of normalisation is *standardization*: $x' = \frac{x-\mu}{\sigma}$ where $\mu$ is the mean of that feature and $\sigma$ is the standard deviation
		- in practice, to choose $\alpha$, we normalize the data and try a range of learning rates (e.g. $10^i,i\in \{-5,...,0\}$), run GD on all of them, and see which performs best
- experiments:
	1. predict house prices with california dataset (here TODO)
	2. mnist with linear regression (first understand how it works related to data and linearity etc) as rounding numbers (problem: if you are trying to guess 9, but you guessed 8.2 --> 8, then the prediction is wrong, but if you guessed 0, it's also wrong but much more 'WRONG' in the sense of squared error, however in normal sense, you're as wrong guessing 8 and guessing 0 ; they're both just 'wrong')
	3. mnist with linear regression but also with classes (onehot encoded vectors for answers) (i'm not sure how this works because regression is regression, not classification)

# Refs
- $argmin_{x \in A} f(x) = \{x\in A \mid \forall a \in A, f(x) \leq f(a)\}$  (the x's which minimize f) ^381afb
- proof of the normal equation for linear regression: ^fc4d76
	- to find $\beta^*$ , we simply need to find the minimum of $L(\beta)$ by solving $\frac{\partial}{\partial \beta} L(\beta) = 0$
	- $\frac{\partial}{\partial \beta} L(\beta) = \frac{\partial}{\partial \beta}\frac{1}{n}\|y-X\beta\|_2^2=\frac{\partial}{\partial \beta}\frac{1}{n}(y-X\beta)^T(y-X\beta)=\frac{\partial}{\partial \beta}\frac{1}{n}(y^T-\beta^TX^T)(y-X\beta)=\frac{\partial}{\partial \beta}\frac{1}{n}\left(y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta\right)=\frac{1}{n}\frac{\partial}{\partial \beta}\left(y^Ty-2y^TX\beta+\beta^TX^TX\beta\right)$
		- $\frac{\partial}{\partial \beta}y^Ty=0$
		- $\frac{\partial}{\partial \beta}(-2y^TX\beta)=-2\frac{\partial}{\partial \beta}(y^TX\beta)=^1-2(y^TX)^T=-2X^Ty$ (ref 1: [[#^ca43a4]])
		- $\frac{\partial}{\partial \beta}\beta^TX^TX\beta=2X^TX\beta$ (ref: [[#^4fad0c]])
	- So $\frac{\partial}{\partial \beta} L(\beta) =\frac{1}{n} (-2X^T y + 2 X^T X \beta) = 0 \Rightarrow X^T X \beta = X^T y$, so if (X^T * X) is invertible, then $\beta = (X^T X)^{-1} X^T y$ 
	- sub-proof refs:
		- Sizes: $y^T: 1 \times n\quad \land \quad X:n \times p \quad \Rightarrow \quad y^T X : 1 \times (p+1).$ We note $a = (a_0,\cdots,a_p) = y^T X : 1 \times (p+1)$. We have  $y^T X \beta = a \beta = (a_0,\cdots,a_p) \cdot \begin{pmatrix}\beta_0 \\ \vdots \\ \beta_p \end{pmatrix} = \sum_{i=0}^{p} a_i \beta_i$ . Therefore, $\frac{\partial}{\partial \beta}a\beta=\begin{pmatrix}\frac{\partial}{\partial \beta_0}\sum_{i=0}^{p} a_i \beta_i\\\vdots\\\frac{\partial}{\partial \beta_p}\sum_{i=0}^{p} a_i \beta_i\end{pmatrix}=\begin{pmatrix}a_0\\\vdots\\a_p\end{pmatrix}=a^T$ . ^ca43a4
		- Let $a = X^T X$. We have $\beta^T a \beta = \begin{pmatrix}\beta_0 & \cdots & \beta_p\end{pmatrix} \cdot \begin{pmatrix}a_{00} & \cdots & a_{0p} \\ \vdots & \vdots & \vdots \\ a_{p0} & \cdots & a_{pp}\end{pmatrix} \cdot \begin{pmatrix}\beta_0 \\ \vdots \\ \beta_p\end{pmatrix} = \sum_{i,j=0}^p \beta_i \ a_{ij} \ \beta{j}$ . For $k \in \{0,\cdots,p\}$, we want to find $\frac{\partial}{\partial \beta_k}\sum_{i,j=0}^p \beta_i \ a_{ij} \ \beta{j}$. We do for i,j != k, i=k and j!=k, i!=k and j=k, i=k and j=k, and we find $\frac{\partial}{\partial \beta_k}\sum_{i,j=0}^p \beta_i \ a_{ij} \ \beta{j} = \sum_{j=0}^{p} a_{kj} \beta_j + \sum_{i=0}^{p} a_{ik} \beta_i = (A\beta)_k + (A^T\beta)_k=((A+A^T)\beta)_k$     ^4fad0c
- proof for GD![[01_gd_proof.jpeg]] ^e053c8