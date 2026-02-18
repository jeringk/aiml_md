# Pre-requisites — Advanced Deep Learning (ADL)

Topics assumed to be learnt before starting this course. Each section includes the key concepts, formulas, and examples you should be comfortable with before encountering them in the ADL modules.

## Topics

- [[#Mathematics — Linear Algebra|Mathematics — Linear Algebra]]
  - [[#Vectors & Matrices|Vectors & Matrices]]
  - [[#Eigendecomposition|Eigendecomposition]]
  - [[#Singular Value Decomposition (SVD)|Singular Value Decomposition (SVD)]]
  - [[#Determinants & Jacobians|Determinants & Jacobians]]
  - [[#Quadratic Forms & Completing the Square|Quadratic Forms & Completing the Square]]
- [[#Mathematics — Probability & Statistics|Mathematics — Probability & Statistics]]
  - [[#Probability Distributions|Probability Distributions]]
  - [[#Bayes' Theorem|Bayes' Theorem]]
  - [[#Maximum Likelihood Estimation (MLE)|Maximum Likelihood Estimation (MLE)]]
  - [[#KL Divergence|KL Divergence]]
  - [[#Jensen-Shannon Divergence (JSD)|Jensen-Shannon Divergence (JSD)]]
  - [[#Chain Rule of Probability|Chain Rule of Probability]]
  - [[#Importance Sampling|Importance Sampling]]
  - [[#Conditional Independence & Markov Property|Conditional Independence & Markov Property]]
- [[#Mathematics — Calculus & Optimization|Mathematics — Calculus & Optimization]]
  - [[#Gradients & Chain Rule|Gradients & Chain Rule]]
  - [[#Gradient Descent Variants|Gradient Descent Variants]]
- [[#Mathematics — Information Theory|Mathematics — Information Theory]]
  - [[#Gaussian Integrals|Gaussian Integrals]]
  - [[#Entropy|Entropy]]
  - [[#Cross-Entropy|Cross-Entropy]]
- [[#Machine Learning Fundamentals|Machine Learning Fundamentals]]
  - [[#Supervised Learning|Supervised Learning]]
  - [[#Unsupervised Learning|Unsupervised Learning]]
  - [[#Regularization|Regularization]]
  - [[#EM Algorithm|EM Algorithm]]
  - [[#Kernel Methods|Kernel Methods]]
- [[#Deep Learning Fundamentals|Deep Learning Fundamentals]]
  - [[#Neural Network Architecture & Parameter Counting|Neural Network Architecture & Parameter Counting]]
  - [[#Backpropagation|Backpropagation]]
  - [[#Activation Functions|Activation Functions]]
  - [[#Loss Functions|Loss Functions]]
  - [[#Convolutional Neural Networks (CNNs)|Convolutional Neural Networks (CNNs)]]
  - [[#Recurrent Neural Networks (RNNs)|Recurrent Neural Networks (RNNs)]]
  - [[#Training Techniques|Training Techniques]]
- [[#Dimensionality Reduction (Pre-Module 2)|Dimensionality Reduction (Pre-Module 2)]]
  - [[#Principal Component Analysis (PCA)|Principal Component Analysis (PCA)]]
  - [[#t-SNE / UMAP|t-SNE / UMAP]]
  - [[#Manifold Hypothesis|Manifold Hypothesis]]
- [[#Probabilistic Graphical Models (Pre-Module 9)|Probabilistic Graphical Models (Pre-Module 9)]]
  - [[#Directed vs Undirected Models|Directed vs Undirected Models]]
  - [[#Markov Chains|Markov Chains]]
  - [[#MCMC Methods|MCMC Methods]]
- [[#Natural Language Processing (Pre-Module 10)|Natural Language Processing (Pre-Module 10)]]
  - [[#Tokenization|Tokenization]]
  - [[#Word Embeddings Concept|Word Embeddings Concept]]
  - [[#N-gram Language Models|N-gram Language Models]]
  - [[#Attention Mechanism|Attention Mechanism]]
  - [[#Positional Encoding|Positional Encoding]]
- [[#Time Series Analysis (Pre-Module 11)|Time Series Analysis (Pre-Module 11)]]
  - [[#Stationarity|Stationarity]]
  - [[#ARIMA Components|ARIMA Components]]
  - [[#ACF and PACF|ACF and PACF]]
- [[#Game Theory (Pre-Module 7)|Game Theory (Pre-Module 7)]]
  - [[#Minimax Games|Minimax Games]]
  - [[#Nash Equilibrium|Nash Equilibrium]]
- [[#Physics & Statistical Mechanics (Pre-Module 9)|Physics & Statistical Mechanics (Pre-Module 9)]]
  - [[#Boltzmann Distribution|Boltzmann Distribution]]
  - [[#Partition Function|Partition Function]]
  - [[#Langevin Dynamics|Langevin Dynamics]]
- [[#Programming & Tools|Programming & Tools]]
  - [[#Python|Python]]
  - [[#NumPy|NumPy]]
  - [[#PyTorch|PyTorch]]
  - [[#Visualization|Visualization]]

---

## Mathematics — Linear Algebra

### Vectors & Matrices
- Vector and matrix multiplication, transpose ($A^T$), inverse ($A^{-1}$), identity matrix
- **Norms**: L1 norm $\|x\|_1 = \sum\|x_i\|$, L2 norm $\|x\|_2 = \sqrt{\sum x_i^2}$, Frobenius norm $\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}$
  - where:
    - $x_i$ = individual elements of vector $x$
    - $a_{ij}$ = element at row $i$, column $j$ of matrix $A$
- Example: given $x = [3, -4]$, L1 norm = $\|3\|+\|-4\|$ = 7, L2 norm = $\sqrt{9+16}$ = 5
- The Frobenius norm is used in the Contractive Autoencoder penalty (Module 3): $\left\|\frac{\partial f_\theta(x)}{\partial x}\right\|_F^2$
  - where:
    - $f_\theta$ = encoder function
    - $\frac{\partial f_\theta}{\partial x}$ = Jacobian of encoder w.r.t. input

### Eigendecomposition
- A square matrix $A$ can be decomposed as $A = V \Lambda V^{-1}$
  - where:
    - $V$ = matrix whose columns are eigenvectors
    - $\Lambda$ = diagonal matrix of corresponding eigenvalues $\lambda_1, \lambda_2, ..., \lambda_n$
- **Eigenvalue equation**: $Av = \lambda v$
  - where:
    - $A$ = square matrix
    - $v$ = eigenvector (non-zero)
    - $\lambda$ = eigenvalue (scalar)
  - Meaning: multiplying $A$ by $v$ only scales it by $\lambda$, does not change direction
- Example: for a covariance matrix, eigenvalues = variance along each principal direction, eigenvectors = those principal directions
- Used directly in PCA (Module 2): the top-$k$ eigenvectors of the covariance matrix $C = \frac{1}{n}X^TX$ give the principal components
  - where:
    - $X \in \mathbb{R}^{n \times d}$ = centered data matrix ($n$ samples, $d$ features)
    - $C \in \mathbb{R}^{d \times d}$ = covariance matrix

### Singular Value Decomposition (SVD)
- Any matrix $A_{m \times n} = U \Sigma V^T$
  - where:
    - $U \in \mathbb{R}^{m \times m}$ = left singular vectors (orthogonal)
    - $\Sigma \in \mathbb{R}^{m \times n}$ = diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
    - $V \in \mathbb{R}^{n \times n}$ = right singular vectors (orthogonal)
- **Truncated SVD**: keep only top-$k$ singular values for low-rank approximation
- **Spectral norm** = largest singular value $\sigma_1(W)$ — used in Spectral Normalization for GANs (Module 7):
  $$W' = \frac{W}{\sigma_1(W)}$$
  - where:
    - $W$ = weight matrix of a neural network layer
    - $\sigma_1(W)$ = largest singular value of $W$
    - $W'$ = normalized weight matrix (ensures 1-Lipschitz constraint)

### Determinants & Jacobians
- **Determinant** $\det(A)$ measures how a linear transformation scales volume: $\det(A) = 0$ means the transformation collapses a dimension (singular matrix)
- **Jacobian matrix** for a function $f: \mathbb{R}^n \to \mathbb{R}^m$:
  $$J_{ij} = \frac{\partial f_i}{\partial x_j}$$
  - where:
    - $f_i$ = $i$-th component of output
    - $x_j$ = $j$-th component of input
    - $J \in \mathbb{R}^{m \times n}$ = Jacobian matrix
- **Jacobian determinant** $\|\det(J)\|$ measures how a nonlinear transformation stretches/compresses probability density locally
- Example: if $f(x,y) = (2x, 3y)$, then $J = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ and $\det(J) = 6$ — the transformation scales area by 6×
- Critical for Normalizing Flows (Module 5):
  $$\log p_X(x) = \log p_Z(z) + \log\|\det J^{-1}\|$$
  - where:
    - $p_X$ = data distribution
    - $p_Z$ = base (latent) distribution
    - $z = f^{-1}(x)$ = inverse mapping from data to latent space
    - $J^{-1}$ = Jacobian of the inverse transformation

### Quadratic Forms & Completing the Square
- Completing the square: $-ax^2 + bx = -a\left(x - \frac{b}{2a}\right)^2 + \frac{b^2}{4a}$
  - where:
    - $a > 0$ = quadratic coefficient
    - $b$ = linear coefficient
- Used to compute partition functions (Gaussian integrals) in Module 9:
  $$\int_{-\infty}^{\infty} e^{-ax^2+bx}dx = \sqrt{\frac{\pi}{a}} \cdot e^{b^2/(4a)}$$
  - where:
    - $a > 0$ = controls Gaussian width
    - $b$ = linear shift term
- Example: for energy $E(x) = 2x^2 - 4x$, complete the square to get $E(x) = 2(x-1)^2 - 2$, then $Z = \int e^{-E(x)}dx = e^2 \cdot \sqrt{\pi/2}$

---

## Mathematics — Probability & Statistics

### Probability Distributions

**Gaussian/Normal**:
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- where:
  - $\mu$ = mean (center)
  - $\sigma$ = standard deviation (spread)
  - $\sigma^2$ = variance
- Shorthand: $x \sim \mathcal{N}(\mu, \sigma^2)$
- Multivariate: $x \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
  - where:
    - $\boldsymbol{\mu} \in \mathbb{R}^d$ = mean vector
    - $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ = covariance matrix
- The most commonly assumed distribution in ADL: latent priors in VAEs (Module 6), noise in diffusion (Module 8), base distribution in flows (Module 5)

**Bernoulli**: $p(x=1) = p$, $p(x=0) = 1-p$
- where:
  - $p \in [0,1]$ = success probability
  - $x \in \{0, 1\}$ = Bernoulli outcome
- Used for binary pixel data, MLE derivation in Module 4
- Example: MLE for Bernoulli with 7 successes in 10 trials → $\hat{p} = 7/10 = 0.7$

**Categorical/Multinomial**: $p(x = k) = \pi_k$
- where:
  - $K$ = number of categories
  - $\pi_k$ = probability of class $k$
  - $\sum_{k=1}^K \pi_k = 1$ = normalization constraint
- Generalization of Bernoulli to $K$ classes — used in language modeling softmax outputs (Module 10)

**Mixture of Gaussians**:
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x\|\mu_k, \Sigma_k)$$
- where:
  - $K$ = number of mixture components
  - $\pi_k$ = mixing weight of component $k$ ($\sum \pi_k = 1$)
  - $\mu_k$ = mean of component $k$
  - $\Sigma_k$ = covariance of component $k$
- Tractable latent variable model example in Module 6

### Bayes' Theorem
$$P(z\|x) = \frac{P(x\|z) \, P(z)}{P(x)}$$
- where:
  - $P(z\|x)$ = **posterior** (distribution of latent given data)
  - $P(x\|z)$ = **likelihood** (how likely data is given latent)
  - $P(z)$ = **prior** (belief about latent before seeing data)
  - $P(x) = \int P(x\|z)P(z)dz$ = **evidence/marginal likelihood** (normalizer, usually intractable)
- The posterior $P(z\|x)$ is what VAEs approximate (Module 6) because $P(x)$ is intractable
- The approximate posterior $q_\phi(z\|x)$ is the encoder network in a VAE
  - where:
    - $\phi$ = encoder parameters

### Maximum Likelihood Estimation (MLE)
$$\theta^* = \arg\max_\theta \sum_{i=1}^N \log p_\theta(x_i)$$
- where:
  - $\theta$ = model parameters
  - $N$ = number of data points
  - $x_i$ = $i$-th data sample
  - $p_\theta(x_i)$ = probability assigned by the model to $x_i$
- Procedure: write log-likelihood → take derivative → set to zero → solve
- Example (Gaussian MLE): given data $\{x_1, ..., x_n\}$, the MLE estimates are $\hat{\mu} = \frac{1}{n}\sum x_i$ (sample mean) and $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \hat{\mu})^2$ (sample variance)
- MLE is equivalent to minimizing $KL(p_{\text{data}} \| p_\theta)$
  - where:
    - $p_{\text{data}}$ = true data distribution
    - $p_\theta$ = model distribution
  - This connection motivates the training of AR models (Module 4) and EBMs (Module 9)

### KL Divergence
$$KL(P \| Q) = \mathbb{E}_P\left[\log \frac{P(x)}{Q(x)}\right] = \sum_x P(x) \log\frac{P(x)}{Q(x)}$$
- where:
  - $P$ = reference/"true" distribution
  - $Q$ = approximate/model distribution
  - $\mathbb{E}_P$ = expectation under $P$
- **Asymmetric**: $KL(P\|Q) \neq KL(Q\|P)$ in general. Always $\geq 0$, equals 0 iff $P=Q$

**Closed form for two Gaussians**:
$$KL\big(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$
- where:
  - $(\mu_1, \sigma_1^2)$ = mean and variance of $P$
  - $(\mu_2, \sigma_2^2)$ = mean and variance of $Q$

**Special case — VAE KL vs standard normal** (where $\mu_2=0, \sigma_2=1$):
$$KL\big(q_\phi(z\|x) \| \mathcal{N}(0,1)\big) = -\frac{1}{2}\sum_{j=1}^{J}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
- where:
  - $J$ = latent-space dimensionality
  - $\mu_j$ = mean of latent dimension $j$ predicted by the encoder
  - $\sigma_j^2$ = variance of latent dimension $j$ predicted by the encoder
- Used in ELBO (Module 6), GAN theory (Module 7 — JSD is built from two KL terms), AR model training (Module 4)

### Jensen-Shannon Divergence (JSD)
$$JSD(P \| Q) = \frac{1}{2}KL(P \| M) + \frac{1}{2}KL(Q \| M)$$
- where:
  - $M = \frac{1}{2}(P + Q)$ = average of the two distributions
  - $P$ = real/reference data distribution
  - $Q$ = generated/model distribution (GAN context)
- **Symmetric** (unlike KL) and bounded: $0 \leq JSD \leq \log 2$
- The original GAN objective at optimal discriminator minimizes JSD between real and generated distributions (Module 7)

### Chain Rule of Probability
$$p(x_1, x_2, ..., x_D) = \prod_{d=1}^{D} p(x_d \| x_1, ..., x_{d-1}) = \prod_{d=1}^{D} p(x_d \| x_{<d})$$
- where:
  - $D$ = total number of dimensions/variables
  - $x_d$ = $d$-th variable
  - $x_{<d}$ = all variables before position $d$
- This is the entire foundation of autoregressive models (Module 4): model each conditional $p(x_d \| x_{<d})$ with a neural network
- Example: for a 3-pixel image, $p(x_1, x_2, x_3) = p(x_1) \cdot p(x_2\|x_1) \cdot p(x_3\|x_1,x_2)$

### Importance Sampling
$$\mathbb{E}_p[f(x)] = \mathbb{E}_q\left[\frac{p(x)}{q(x)} f(x)\right] \approx \frac{1}{K}\sum_{k=1}^{K} \frac{p(x_k)}{q(x_k)} f(x_k)$$
- where:
  - $p(x)$ = target distribution (hard to sample from)
  - $q(x)$ = proposal distribution (easy to sample from)
  - $\frac{p(x)}{q(x)}$ = importance weight
  - $K$ = number of samples
- Used in VAE training (Module 6) — importance-weighted autoencoders (IWAE) use multiple importance samples to tighten the ELBO bound
- Key intuition: if $q$ is close to $p$, variance is low → better estimates

### Conditional Independence & Markov Property
- $X \perp Y \| Z$ means $X$ and $Y$ are independent given $Z$: $p(X, Y \| Z) = p(X\|Z) \cdot p(Y\|Z)$
- **Markov property**: future depends only on present, not past — $p(x_t \| x_{t-1}, x_{t-2}, ...) = p(x_t \| x_{t-1})$
- Used in diffusion models (Module 8): forward/reverse processes are Markov chains
- RBMs (Module 9) exploit conditional independence: all hidden units are independent given visible units, and vice versa → enables efficient parallel Gibbs sampling

---

## Mathematics — Calculus & Optimization

### Gradients & Chain Rule
$$\nabla_x f(x) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]$$
- where:
  - $\nabla_x f$ = gradient vector (direction of steepest ascent)
  - $\frac{\partial f}{\partial x_i}$ = partial derivative of $f$ w.r.t. input dimension $i$
- **Chain rule**: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$
  - where:
    - $L$ = loss
    - $y$ = intermediate variable
    - $w$ = parameter to update
- This is the backbone of backpropagation — gradients flow backward through the computational graph
- Example: for $L = (y - \hat{y})^2$ and $\hat{y} = wx + b$:
  - $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w} = 2(\hat{y} - y) \cdot x$

### Gradient Descent Variants
- **SGD**: $\theta \leftarrow \theta - \eta \nabla_\theta L$
  - where:
    - $\theta$ = model parameters
    - $\eta$ = learning rate (step size)
    - $\nabla_\theta L$ = gradient of loss w.r.t. parameters
- **Adam**: adaptive learning rates per parameter using first moment $m_t$ (mean of gradients) and second moment $v_t$ (mean of squared gradients)
  - $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
  - where:
    - $g_t = \nabla_\theta L_t$ = gradient at step $t$
    - $\beta_1 \approx 0.9$ = first-moment decay rate
    - $\beta_2 \approx 0.999$ = second-moment decay rate
  - Update: $\theta \leftarrow \theta - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
    - where:
      - $\hat{m}_t$ = bias-corrected first-moment estimate
      - $\hat{v}_t$ = bias-corrected second-moment estimate
- Adam is the default optimizer for most ADL models; typical $\eta = 0.001$

### Gaussian Integrals
$$\int_{-\infty}^{\infty} e^{-ax^2} dx = \sqrt{\frac{\pi}{a}} \quad \text{for } a > 0$$
- where:
  - $a$ = width control (larger $a$ means narrower Gaussian)
- Generalizes to: $\int e^{-ax^2 + bx} dx = \sqrt{\frac{\pi}{a}} \cdot e^{b^2/(4a)}$ via completing the square
  - where:
    - $b$ = linear shift term
- For unnormalized Gaussian $e^{-(x-\mu)^2/(2\sigma^2)}$: the integral equals $\sigma\sqrt{2\pi}$ (this is why the Gaussian PDF has $\frac{1}{\sigma\sqrt{2\pi}}$ as normalizer)
- This is not just theory — you need this to compute partition functions by hand for EBM exam problems (Module 9)

---

## Mathematics — Information Theory

### Entropy
$$H(X) = -\sum_x p(x) \log p(x)$$
- where:
  - $p(x)$ = probability of outcome $x$
  - $\sum_x$ = sum over all possible outcomes
- Measures uncertainty/randomness: maximum for uniform distribution, minimum (0) for deterministic
- Example: fair coin → $H = -2 \times 0.5 \log_2 0.5 = 1$ bit; biased coin ($p=0.9$) → $H \approx 0.47$ bits

### Cross-Entropy
$$H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + KL(P \| Q)$$
- where:
  - $P$ = true distribution (labels)
  - $Q$ = predicted distribution (model output)
  - $H(P)$ = entropy of true distribution (constant during training)
- Since $H(P)$ is constant, minimizing cross-entropy is equivalent to minimizing KL divergence
- Cross-entropy loss is the standard loss for classification and is used as reconstruction loss for binary data in autoencoders (Module 3)
- Example: true label = $[1, 0, 0]$, prediction = $[0.7, 0.2, 0.1]$ → $CE = -(1 \cdot \log 0.7 + 0 + 0) \approx 0.357$

---

## Machine Learning Fundamentals

### Supervised Learning
- **Classification**: predict discrete labels — logistic regression, SVM, decision trees, neural networks
- **Regression**: predict continuous values — linear regression, neural networks
- **Evaluation metrics**: accuracy, precision, recall, F1-score, AUC-ROC
- You should be comfortable training a basic classifier and evaluating its performance

### Unsupervised Learning
- **Clustering**: K-Means (assign points to nearest centroid, update centroids), DBSCAN, hierarchical
- **Dimensionality reduction**: PCA, t-SNE — reduce high-dimensional data to 2D/3D for visualization or compression
- **Density estimation**: modeling $p(x)$ — the core goal of generative models in this course
  - where:
    - $p(x)$ = probability density function over data $x$
- Example: K-Means on MNIST handwritten digits to discover digit clusters without labels

### Regularization
- **L1 (Lasso)**: penalty $= \lambda \sum_i \|w_i\|$
  - where:
    - $\lambda$ = regularization strength
    - $w_i$ = individual weight parameters
  - Promotes sparsity — some weights go to exactly zero
- **L2 (Ridge)**: penalty $= \lambda \sum_i w_i^2$ — penalizes large weights, prevents overfitting
- **Dropout**: randomly set fraction $p$ of neurons to zero during training — acts as ensemble
  - where:
    - $p$ = dropout probability (e.g., $p = 0.5$ drops half the neurons)
- L1 regularization is used in Sparse PCA (Module 2) and Sparse Autoencoder (Module 3)

### EM Algorithm
- **Expectation-Maximization**: iterative algorithm for MLE with latent variables
- **E-step**: compute $q(z) = p(z \| x; \theta^{\text{old}})$ — expected value of latent variables given current parameters
- **M-step**: $\theta^{\text{new}} = \arg\max_\theta \mathbb{E}_{q(z)}[\log p(x, z; \theta)]$ — maximize expected complete-data log-likelihood
  - where:
    - $z$ = latent variables
    - $x$ = observed data
    - $\theta$ = model parameters
- Used to train Probabilistic PCA and Factor Analysis (Module 2), and conceptually related to VAE training (Module 6)
- Example: fitting a Gaussian mixture model — E-step computes cluster responsibilities $\gamma_{ik} = p(z=k\|x_i)$, M-step updates means $\mu_k$, covariances $\Sigma_k$, and weights $\pi_k$

### Kernel Methods
$$\kappa(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$
- where:
  - $\kappa$ = kernel function
  - $\phi$ = (implicit) feature mapping to high-dimensional space
  - $x_i, x_j$ = input data points
- **Kernel trick**: compute inner products in high-dimensional feature space without explicitly computing $\phi$ — only need the kernel function $\kappa$
- **Common kernels**:
  - RBF/Gaussian: $\kappa(x,y) = \exp(-\gamma\|x-y\|^2)$
    - where:
      - $\gamma > 0$ = kernel width control
  - Polynomial: $\kappa(x,y) = (x^Ty + c)^d$
    - where:
      - $c$ = constant term
      - $d$ = polynomial degree
- Used in Kernel PCA (Module 2) to capture nonlinear structure in data

---

## Deep Learning Fundamentals

### Neural Network Architecture & Parameter Counting

**Fully Connected (Dense) Layer**:
- Parameters: $(D_{in} + 1) \times D_{out}$
  - where:
    - $D_{in}$ = number of input features
    - $D_{out}$ = number of output features
    - $+1$ = one bias per output neuron
- Example: layer with 784 inputs and 256 outputs → $784 \times 256 + 256 = 200{,}960$ parameters

![Fully Connected (Dense) Layer — connections and parameter counting](images/dense-layer.svg)

**Conv2D**:
- Parameters: $(K_h \times K_w \times C_{in} + 1) \times C_{out}$
  - where:
    - $K_h$ = kernel height
    - $K_w$ = kernel width
    - $C_{in}$ = number of input channels
    - $C_{out}$ = number of output filters
    - $+1$ = one bias per filter
- Output spatial size: $\lfloor \frac{H + 2p - K}{s} \rfloor + 1$
  - where:
    - $H$ = input height (same formula applies to width)
    - $p$ = padding
    - $K$ = kernel size
    - $s$ = stride
- Example: 3×3 kernel, 64 input channels, 128 output channels → $(3 \times 3 \times 64 + 1) \times 128 = 73{,}856$ parameters

![Conv2D sliding window operation](images/conv2d-operation.svg)

![Multi-channel Conv2D and parameter counting](images/conv2d-channels.svg)

![Output size calculation — stride and padding effects](images/conv2d-output-size.svg)

- Video: [2D Convolution Explained: Fundamental Operation in Computer Vision](https://www.youtube.com/watch?v=yb2tPt0QVPY)

**ConvTranspose2D** (Transposed Convolution):
- Same parameter formula as Conv2D: $(K_h \times K_w \times C_{in} + 1) \times C_{out}$
- Output spatial size (upsampling): $(H_{in}-1) \times s - 2p + K + p_{out}$
  - where:
    - $H_{in}$ = input height
    - $s$ = stride
    - $p$ = padding
    - $K$ = kernel size
    - $p_{out}$ = output padding
- Used in autoencoder decoders (Module 3) and GAN generators (Module 7) to upsample feature maps

![ConvTranspose2D — transposed convolution for upsampling](images/conv-transpose2d.svg)

**Flatten / Reshape**: 0 trainable parameters — Flatten outputs $C \times H \times W$, Reshape restores spatial dimensions

![Flatten and Reshape — tensor-to-vector-to-tensor transformation](images/flatten-reshape.svg)

You should be able to count parameters through an entire encoder-decoder network (Module 3 tests this explicitly).

### Backpropagation
- **Forward pass**: compute outputs layer by layer
- **Backward pass**: compute gradients layer by layer using chain rule
- **Computational graph**: DAG of operations — automatic differentiation frameworks (PyTorch autograd) traverse this graph backward
- You should understand how gradients flow through common operations (matrix multiply, activation, loss)

### Activation Functions

**Sigmoid**:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- where:
  - $x$ = pre-activation (weighted sum)
  - $e$ = Euler's number
- Output range: $(0, 1)$ — interprets output as probability
- Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- Used as: output activation for binary data (Module 3), discriminator output in GANs (Module 7), gate activations in LSTM/GRU, RBM conditionals (Module 9)
- Issue: vanishing gradients when $\|x\|$ is large (saturation regions where derivative → 0)

**Tanh**:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- where:
  - $x$ = pre-activation input
  - $e$ = Euler's number
- Output range: $(-1, 1)$ — zero-centered unlike sigmoid
- Derivative: $\tanh'(x) = 1 - \tanh^2(x)$
- Used as: output activation for normalized data (Module 3), gated activations in WaveNet/PixelCNN (Module 4), planar flow activation (Module 5)
- Example: for Planar Flow, $h'(a) = 1 - \tanh^2(a)$ appears directly in the Jacobian determinant formula

**ReLU**:
$$\text{ReLU}(x) = \max(0, x)$$
- where:
  - $x$ = pre-activation input
- Output range: $[0, \infty)$
- Derivative: $\text{ReLU}'(x) = \begin{cases} 0 & x < 0 \\ 1 & x > 0 \end{cases}$ (undefined at $x=0$, typically set to 0)
- Default choice for hidden layers — no vanishing gradient for positive inputs
- Used in DCGAN generator hidden layers (Module 7), encoder/decoder hidden layers (Module 3)
- Issue: "dying ReLU" — neurons stuck at zero if they always receive negative input

**LeakyReLU**:
$$\text{LeakyReLU}(x) = \max(\alpha x, x) = \begin{cases} x & x \geq 0 \\ \alpha x & x < 0 \end{cases}$$
- where:
  - $x$ = pre-activation input
  - $\alpha \approx 0.01$ = small positive slope for negative inputs
- Fixes dying ReLU by allowing small gradient ($\alpha$) for negative inputs
- Used specifically in DCGAN discriminator throughout all layers (Module 7)

**ELU / GELU**: smooth alternatives to ReLU
- ELU: $f(x) = \begin{cases} x & x \geq 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}$
  - where:
    - $\alpha$ = saturation value for negative inputs
  - Smooth, allows negative outputs, self-normalizing properties
- GELU: $f(x) = x \cdot \Phi(x)$
  - where:
    - $\Phi(x)$ = CDF of standard Gaussian
  - Used in modern Transformers (Module 10)

**Softmax**:
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
- where:
  - $z \in \mathbb{R}^K$ = logits (raw scores)
  - $K$ = number of classes
  - $z_i$ = logit for class $i$
  - $\sigma(z)_i$ = predicted probability for class $i$
- All outputs sum to 1: $\sum_i \sigma(z)_i = 1$
- Used in attention mechanism (Module 10), AR model outputs (Module 4), language model predictions (Module 10)
- Example: logits $[2.0, 1.0, 0.1]$ → softmax ≈ $[0.659, 0.242, 0.099]$

### Loss Functions

**MSE (Mean Squared Error)**:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^n \|x_i - \hat{x}_i\|^2$$
- where:
  - $n$ = number of samples
  - $x_i$ = true value for sample $i$
  - $\hat{x}_i$ = predicted/reconstructed value for sample $i$
- Default for continuous reconstruction — used in autoencoders (Module 3), VQ-VAE codebook loss (Module 6), DDPM noise prediction (Module 8)
- Gradient: $\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{2}{n}(\hat{x}_i - x_i)$

**Binary Cross-Entropy (BCE)**:
$$\mathcal{L}_{\text{BCE}} = -\sum_i \left[x_i \log \hat{x}_i + (1-x_i) \log(1-\hat{x}_i)\right]$$
- where:
  - $x_i \in \{0, 1\}$ = true binary label
  - $\hat{x}_i \in (0, 1)$ = predicted probability (sigmoid output)
- For binary/image data in $[0,1]$ — requires sigmoid output layer
- Used in autoencoder reconstruction (Module 3), GAN discriminator loss (Module 7)
- Example: true pixel = 1, predicted = 0.9 → loss $= -[\log(0.9) + 0] \approx 0.105$

**Negative Log-Likelihood (NLL)**:
$$\mathcal{L}_{\text{NLL}} = -\sum_{i=1}^{N} \log p_\theta(x_i)$$
- where:
  - $N$ = number of data points
  - $p_\theta(x_i)$ = probability the model assigns to observed data point $x_i$
  - $\theta$ = model parameters
- The general training objective for generative models
- For autoregressive models (Module 4): $\mathcal{L} = -\sum_i \sum_d \log p_\theta(x_{id} \| x_{i,<d})$
  - where:
    - $d$ = dimension index
    - $x_{i,<d}$ = all dimensions of sample $i$ before position $d$
- **Perplexity** = $\exp(\text{NLL per token})$ — used to evaluate language models (Module 10); lower = better

### Convolutional Neural Networks (CNNs)

**Dilated Convolutions**:
- Insert gaps (dilation) in the kernel — receptive field grows exponentially with depth while keeping parameter count fixed
- Dilation rates $1, 2, 4, 8, 16...$ → receptive field of $2^L$ with $L$ layers
  - where:
    - $L$ = number of layers with doubled dilation rate
- Used in WaveNet (Module 4) for audio generation to capture long-range dependencies

**1×1 Convolutions**:
- Channel mixing without spatial interaction — equivalent to a pointwise fully connected layer across channels
- Parameters: $(1 \times 1 \times C_{in} + 1) \times C_{out} = (C_{in} + 1) \times C_{out}$
- Used in Glow (Module 5) as invertible 1×1 convolutions for learnable channel permutations

### Recurrent Neural Networks (RNNs)

**Vanilla RNN**:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
- where:
  - $h_t \in \mathbb{R}^{d_h}$ = hidden state at time $t$
  - $h_{t-1}$ = previous hidden state
  - $x_t$ = input at time $t$
  - $W_h \in \mathbb{R}^{d_h \times d_h}$ = hidden-to-hidden weight matrix
  - $W_x \in \mathbb{R}^{d_h \times d_x}$ = input-to-hidden weight matrix
  - $b$ = bias vector
- Processes sequences step-by-step; hidden state $h_t$ summarizes history
- Problem: **vanishing/exploding gradients** for long sequences — gradient magnitude shrinks/grows exponentially with sequence length

**LSTM (Long Short-Term Memory)**:
- **Cell state** $c_t$ acts as long-term memory + three gates control information flow:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
- where:
  - $f_t$ = forget gate (what to erase from cell state)
  - $W_f$ = forget-gate weight matrix
  - $b_f$ = forget-gate bias
  - $[h_{t-1}, x_t]$ = concatenation of previous hidden state and current input

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
- where:
  - $i_t$ = input gate (what new information to store)
  - $W_i$ = input-gate weight matrix
  - $b_i$ = input-gate bias

$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
- where:
  - $\tilde{c}_t$ = candidate cell state (proposed new memory content)
  - $W_c$ = candidate-state weight matrix
  - $b_c$ = candidate-state bias

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
- where:
  - $c_t$ = updated cell state
  - $c_{t-1}$ = previous cell state
  - $\odot$ = element-wise multiplication
  - $f_t \odot c_{t-1}$ = forget contribution
  - $i_t \odot \tilde{c}_t$ = input contribution

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t)$$
- where:
  - $o_t$ = output gate (what part of cell state to expose)
  - $W_o$ = output-gate weight matrix
  - $b_o$ = output-gate bias
  - $h_t$ = hidden state emitted at time $t$
- Used in AR models for sequential generation (Module 4), VRNN (Module 11)

**GRU (Gated Recurrent Unit)**:
- Simplified LSTM with two gates (reset $r_t$, update $z_t$) — often comparable performance, fewer parameters than LSTM

**BiLSTM**: runs forward and backward LSTMs, combines their hidden states
$$\text{output}(x_t) = h_t^f + h_t^b \quad \text{(sum)} \quad \text{or} \quad [h_t^f; h_t^b] \quad \text{(concat)}$$
- where:
  - $x_t$ = input token/vector at time $t$
  - $h_t^f$ = forward LSTM hidden state (left-to-right)
  - $h_t^b$ = backward LSTM hidden state (right-to-left)
- Captures both left and right context — used in CoVe (Module 10)

### Training Techniques

**Batch Normalization**:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
- where:
  - $x$ = input activation
  - $\mu_B$ = mini-batch mean
  - $\sigma_B^2$ = mini-batch variance
  - $\epsilon$ = small numerical-stability constant
  - $\gamma$ = learned scale parameter
  - $\beta$ = learned shift parameter
  - $\hat{x}$ = normalized activation
  - $y$ = output activation after affine transform
- Stabilizes training, allows higher learning rates — used in DCGAN (Module 7), deep autoencoders (Module 3)
- Applied to all layers except generator output and discriminator input in DCGAN

**Layer Normalization**: same formula but normalizes across features (not batch dimension)
- where:
  - mean/variance are computed over all features for each individual sample
- Advantage over BatchNorm: works with variable-length sequences and small batches
- Used in Transformer blocks (Module 10)

**Instance Normalization**: normalizes each sample's spatial dimensions independently
- where:
  - mean/variance are computed per channel per sample
  - normalization is over spatial dimensions $(H, W)$
- Used in style transfer, CycleGAN (Module 7)

**Residual / Skip Connections**:
$$y = F(x) + x$$
- where:
  - $F(x)$ = output of a residual block
  - $x$ = block input (skip path)
  - $y$ = residual block output
- Gradient flows directly through the skip path ($\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$) — prevents vanishing gradient
- Enables training very deep networks — used in deep autoencoders (Module 3), U-Net (Module 8), Transformer blocks (Module 10), WaveNet (Module 4)

**Dropout**: randomly zero out neurons with probability $p$ during training
- where:
  - $p$ = dropout probability (e.g., 0.5)
  - remaining activations are scaled by $\frac{1}{1-p}$
- Acts as regularization via implicit ensemble of sub-networks
- Masking noise in Denoising Autoencoders (Module 3) is conceptually similar

---

## Dimensionality Reduction (Pre-Module 2)

### Principal Component Analysis (PCA)
- **Covariance matrix**: $C = \frac{1}{n} X^T X$
  - where:
    - $X \in \mathbb{R}^{n \times d}$ = centered data matrix ($n$ samples, $d$ features)
    - $C \in \mathbb{R}^{d \times d}$ = covariance matrix
  - $C_{ij}$ captures the linear relationship between features $i$ and $j$
- **PCA procedure**: center data ($X \leftarrow X - \bar{X}$) → compute covariance $C$ → eigendecomposition $C = V\Lambda V^T$ → project onto top-$k$ eigenvectors
- **Projection**: $Z = XW$
  - where:
    - $W \in \mathbb{R}^{d \times k}$ = matrix containing top-$k$ eigenvectors
    - $Z \in \mathbb{R}^{n \times k}$ = reduced representation
- **Reconstruction**: $\hat{X} = ZW^T$ — approximation of original data in the original $d$-dimensional space
- **Explained variance ratio**: $\frac{\lambda_i}{\sum_j \lambda_j}$
  - where:
    - $\lambda_i$ = $i$-th eigenvalue (variance captured by component $i$)
- Example: 100-dimensional data with 90% variance in first 5 components → PCA reduces to 5 dimensions with minimal information loss
- Module 2 builds on this with Randomized, Incremental, Kernel, Probabilistic, and Sparse variants

### t-SNE / UMAP
- Nonlinear dimensionality reduction for **visualization** (typically to 2D/3D)
- t-SNE preserves local structure (nearby points stay nearby), UMAP preserves both local and some global structure
- Referenced in Manifold Learning (Module 2) as comparison methods

### Manifold Hypothesis
- Real-world high-dimensional data (images, text) lies near a **low-dimensional manifold** embedded in the high-dimensional space
- Example: the space of natural face images (a few hundred meaningful parameters: pose, lighting, identity) is tiny compared to all possible pixel arrangements (millions of dimensions)
- This motivates the entire field of representation learning and generative modeling (Modules 1–9)

---

## Probabilistic Graphical Models (Pre-Module 9)

### Directed vs Undirected Models
- **Directed (Bayesian Networks)**: edges have direction, represent causal/generative relationships
  - Example: $z \to x$ in VAEs — latent variable generates observed data
- **Undirected (Markov Random Fields)**: edges are symmetric, model associations
  - Example: Ising model, Boltzmann machines (Module 9) — neighboring variables influence each other bidirectionally
- Key difference: directed models have tractable sampling but may have intractable inference; undirected models have intractable partition function $Z$

### Markov Chains
$$p(x_{t+1} \| x_t, x_{t-1}, ..., x_1) = p(x_{t+1} \| x_t)$$
- where:
  - $x_t$ = state at time $t$
  - $x_{t+1}$ = next state
- **Transition matrix** $T$: $T_{ij} = p(x_{t+1} = j \| x_t = i)$
  - where:
    - $T_{ij}$ = transition probability from state $i$ to state $j$
    - each row of $T$ sums to 1
- **Stationary distribution** $\pi$: $\pi T = \pi$
  - where:
    - $\pi$ = long-run distribution of the Markov chain
- Diffusion models (Module 8) use forward/reverse Markov chains: forward gradually adds noise (data → noise), reverse denoises (noise → data)

### MCMC Methods
- **Gibbs Sampling**: sample each variable conditioned on all others, cycling through variables
  - For RBM (Module 9): alternate between sampling $h \| v$ (all hidden given visible) and $v \| h$ (all visible given hidden) — both are easy due to conditional independence within each layer
- **Langevin Dynamics**:
  $$x_{k+1} = x_k - \frac{\eta}{2}\nabla_x E(x_k) + \sqrt{\eta}\,\epsilon_k$$
  - where:
    - $x_k$ = current sample
    - $x_{k+1}$ = next sample after one update step
    - $\eta$ = step size
    - $E(x)$ = energy function
    - $\nabla_x E(x_k)$ = energy gradient at $x_k$
    - $\epsilon_k \sim \mathcal{N}(0, I)$ = random Gaussian noise for exploration
  - As $\eta \to 0$ and $k \to \infty$, samples converge to $p(x) \propto e^{-E(x)}$
  - Used for EBM and score-based sampling (Module 9)
- **Contrastive Divergence (CD-$k$)**: approximate MCMC starting from data instead of random initialization — run only $k$ steps of Gibbs sampling (typically $k=1$)
  - Much cheaper than full MCMC, surprisingly effective for RBM training (Module 9)

---

## Natural Language Processing (Pre-Module 10)

### Tokenization
- Converting text into tokens (units the model processes)
- **Word-level**: "the cat sat" → ["the", "cat", "sat"] — simple but large vocabulary, out-of-vocabulary (OOV) problem for unseen words
- **Subword (BPE/WordPiece)**: "unhappiness" → ["un", "happi", "ness"] — handles rare words by breaking them into known subwords, used in BERT/GPT
- **Vocabulary size** $V$ determines embedding matrix size ($V \times d$) and softmax output dimension — typical: $V \approx$ 30K–50K for subword models

### Word Embeddings Concept
- Represent each word as a dense vector $w \in \mathbb{R}^d$ (typically $d$ = 100–300) instead of sparse one-hot vector $\in \mathbb{R}^V$
- Similar words have similar vectors: $\cos(w_1, w_2) = \frac{w_1 \cdot w_2}{\|w_1\|\|w_2\|}$
  - where:
    - $\cos$ = cosine similarity function
    - similarity range is from -1 (opposite) to +1 (identical direction)
- Module 10 covers word2Vec (CBOW, Skip-gram), GloVe, and contextual embeddings (CoVe, BERT, GPT)
- Example: $\cos(\text{king}, \text{queen}) > \cos(\text{king}, \text{apple})$

### N-gram Language Models
$$p(w_t \| w_1, ..., w_{t-1}) \approx p(w_t \| w_{t-n+1}, ..., w_{t-1})$$
- where:
  - $w_t$ = word at position $t$
  - $w_1, ..., w_{t-1}$ = full history before position $t$
  - $n$ = N-gram order (context length + 1)
  - $w_{t-n+1}, ..., w_{t-1}$ = truncated context used by the N-gram model
- **Bigram** ($n=2$): $p(w_t \| w_{t-1})$; **Trigram** ($n=3$): $p(w_t \| w_{t-2}, w_{t-1})$
- Limitations: fixed context window, data sparsity (many N-grams never observed), no semantic generalization — motivates neural language models (Module 10)

### Attention Mechanism
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
- where:
  - $Q \in \mathbb{R}^{n \times d_k}$ = queries ("what am I looking for?") — typically derived from the current token
  - $K \in \mathbb{R}^{m \times d_k}$ = keys ("what do I contain?") — derived from all tokens being attended to
  - $V \in \mathbb{R}^{m \times d_v}$ = values ("what content do I provide?") — the actual information to retrieve
  - $d_k$ = dimension of queries/keys, $\sqrt{d_k}$ = scaling factor to prevent softmax saturation
  - $n$ = number of query positions, $m$ = number of key/value positions
  - $QK^T \in \mathbb{R}^{n \times m}$ = attention score matrix (how relevant each key is to each query)
- **Multi-head attention**: run $h$ parallel attention heads with different learned projections, concatenate, project:
  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$
  - where:
    - $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ = output of head $i$
    - $W_i^Q, W_i^K, W_i^V$ = per-head projection matrices
    - $h$ = number of attention heads
    - $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ = output projection matrix
  - Allows attending to different types of relationships simultaneously
- Foundation of the Transformer architecture (Module 10), later used in diffusion U-Net (Module 8), Flow++ (Module 5)

### Positional Encoding
- Transformers have no inherent notion of sequence order (unlike RNNs which process sequentially)
- **Sinusoidal encoding**:
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
  - where:
    - $pos$ = token position in the sequence
    - $i$ = frequency index
    - $d$ = embedding dimension
    - $2i$ = even embedding dimension index
    - $2i+1$ = odd embedding dimension index
- Or **learned embeddings**: trainable position vectors added to token embeddings

---

## Time Series Analysis (Pre-Module 11)

### Stationarity
- A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time
- Non-stationary series need **differencing** (the "I" in ARIMA) to become stationary: $\Delta X_t = X_t - X_{t-1}$
  - where:
    - $\Delta$ = first-difference operator
    - operator may be applied $d$ times for order-$d$ differencing
- Example: stock prices are non-stationary (trending), but daily returns ($\Delta X_t$) are approximately stationary

### ARIMA Components
$$ARIMA(p, d, q): \quad \phi(B)(1-B)^d X_t = \theta(B) \epsilon_t$$
- where:
  - $p$ = AR (autoregressive) order — number of past values used
  - $d$ = differencing order — number of times the series is differenced
  - $q$ = MA (moving average) order — number of past error terms used
  - $B$ = backshift operator: $B \cdot X_t = X_{t-1}$, $B^2 \cdot X_t = X_{t-2}$
  - $\phi(B) = 1 - \phi_1 B - ... - \phi_p B^p$ = AR polynomial
  - $\theta(B) = 1 + \theta_1 B + ... + \theta_q B^q$ = MA polynomial
  - $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ = white noise error term

- **AR($p$)** expanded: $X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t$
  - where:
    - $\phi_i$ = AR coefficient for lag $i$
- **MA($q$)** expanded: $X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}$
  - where:
    - $\theta_j$ = MA coefficient for lagged error $j$
- **S-ARIMA** $SARIMA(p,d,q)(P,D,Q)_s$: adds seasonal AR/MA/differencing components with period $s$
  - where:
    - $(P,D,Q)$ = seasonal AR order, seasonal differencing order, seasonal MA order
    - $s$ = seasonal period (e.g., 12 for monthly data with yearly cycles)

### ACF and PACF
- **Autocorrelation Function (ACF)**: $\rho(k) = \frac{\text{Cov}(X_t, X_{t-k})}{\text{Var}(X_t)}$
  - where:
    - $k$ = lag
    - $\rho(k) \in [-1, 1]$ = correlation with $k$-step lagged series
  - Helps determine MA order $q$: significant ACF values cut off after lag $q$
- **Partial Autocorrelation Function (PACF)**: correlation between $X_t$ and $X_{t-k}$ after removing the effect of intermediate lags
  - Helps determine AR order $p$: significant PACF values cut off after lag $p$

---

## Game Theory (Pre-Module 7)

### Minimax Games
$$\min_G \max_D V(D, G)$$
- where:
  - $G$ = minimizer player (generator in GANs)
  - $D$ = maximizer player (discriminator in GANs)
  - $V(D, G)$ = game value / payoff
- Two-player zero-sum game: one player's gain = other's loss
- This is exactly the GAN objective (Module 7): Generator tries to minimize, Discriminator tries to maximize the same objective
- Example: in rock-paper-scissors, the minimax strategy is to play each option with probability 1/3 — no opponent strategy can exploit this

### Nash Equilibrium
- A state where **no player can improve** by unilaterally changing their strategy
- For GANs (Module 7): the Nash equilibrium is when $p_G = p_{\text{data}}$ and $D(x) = 0.5$ for all $x$
  - where:
    - $p_G$ = generator distribution
    - $p_{\text{data}}$ = real data distribution
    - $D(x)$ = discriminator output (probability that $x$ is real)
- In practice, GANs rarely converge to Nash equilibrium — leading to **mode collapse** (generator produces limited variety) and **training instability** (oscillation)

---

## Physics & Statistical Mechanics (Pre-Module 9)

### Boltzmann Distribution
$$p(x) = \frac{1}{Z} e^{-E(x)/T}$$
- where:
  - $E(x)$ = energy of state $x$
  - $T$ = temperature controlling sharpness
  - $Z = \sum_x e^{-E(x)/T}$ = partition function (normalizer)
  - $p(x)$ = normalized probability distribution over states
- In ADL (Module 9), temperature is typically absorbed: $p_\theta(x) = \frac{1}{Z(\theta)} \exp(-E_\theta(x))$
  - where:
    - $E_\theta$ = energy function parameterized by neural network with parameters $\theta$

### Partition Function
$$Z(\theta) = \int \exp(-E_\theta(x)) \, dx$$
- where:
  - $Z(\theta)$ = partition function
  - $E_\theta(x)$ = energy function parameterized by $\theta$
  - $\int (\cdot)\,dx$ = integration over all possible states $x$
- **Intractable** for most models — computing it requires integrating/summing over an exponentially large or continuous state space
- This intractability motivates score-based models (Module 9): the score $\nabla_x \log p(x) = -\nabla_x E(x)$ doesn't depend on $Z$ because $\nabla_x \log Z = 0$ (constant w.r.t. $x$)

### Langevin Dynamics
$$x_{k+1} = x_k - \frac{\eta}{2} \nabla_x E(x_k) + \sqrt{\eta}\, \epsilon_k$$
- where:
  - $x_k$ = current sample at step $k$
  - $x_{k+1}$ = next sample
  - $\eta$ = step size
  - $\nabla_x E(x_k)$ = gradient of energy at $x_k$
  - $-\nabla_x E(x_k)$ = direction toward lower energy (higher probability)
  - $\epsilon_k \sim \mathcal{N}(0, I)$ = random Gaussian noise for exploration
- Combines gradient descent on energy (exploitation — moves toward high-probability regions) with random noise (exploration — avoids getting stuck)
- As $\eta \to 0$ and $k \to \infty$, the distribution of $x_k$ converges to $p(x) \propto e^{-E(x)}$
- Used for sampling in EBMs and score-based models (Module 9)

---

## Programming & Tools

### Python
- Proficiency in Python 3.x: classes, list comprehensions, generators, decorators
- Understanding of broadcasting semantics in array operations

### NumPy
- Array operations, reshaping, slicing, broadcasting
- Random sampling: `np.random.normal(mu, sigma, size)`, `np.random.uniform(low, high, size)`
- Linear algebra: `np.linalg.eig(A)` → eigenvalues/eigenvectors, `np.linalg.svd(A)` → SVD, `np.dot(A, B)` → matrix multiply

### PyTorch
- **Tensors**: creation (`torch.randn`, `torch.zeros`), GPU transfer (`.to('cuda')`), gradients (`.requires_grad_(True)`)
- **Autograd**: automatic differentiation — `loss.backward()` computes all gradients, `optimizer.step()` updates parameters
- **nn.Module**: building models with `__init__` (define layers) + `forward` (define computation), registering parameters
- **Common layers**: `nn.Linear(in, out)`, `nn.Conv2d(C_in, C_out, K)`, `nn.ConvTranspose2d(C_in, C_out, K)`, `nn.BatchNorm2d(C)`, `nn.LayerNorm(d)`
- **DataLoader**: `DataLoader(dataset, batch_size=32, shuffle=True)` — handles batching, shuffling, parallel data loading
- **Training loop**: forward pass → loss computation → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`
- Example: you should be able to write a training loop for a simple CNN classifier from scratch

### Visualization
- **Matplotlib**: plotting loss curves, grid of generated images (`plt.imshow`), latent space visualizations (`plt.scatter`)
- Useful for debugging: plot reconstruction quality, training vs validation loss, generated samples per epoch
