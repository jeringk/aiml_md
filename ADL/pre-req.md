# Pre-requisites — Advanced Deep Learning (ADL)

Topics assumed to be learnt before starting this course. Each section includes the key concepts, formulas, and examples you should be comfortable with before encountering them in the ADL modules.

---

## Mathematics — Linear Algebra

### Vectors & Matrices
- Vector and matrix multiplication, transpose ($A^T$), inverse ($A^{-1}$), identity matrix
- **Norms**: L1 norm $\|x\|_1 = \sum|x_i|$, L2 norm $\|x\|_2 = \sqrt{\sum x_i^2}$, Frobenius norm $\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}$
- Example: given $x = [3, -4]$, L1 norm = 7, L2 norm = 5
- The Frobenius norm is used in the Contractive Autoencoder penalty (Module 3): $\left\|\frac{\partial f_\theta(x)}{\partial x}\right\|_F^2$

### Eigendecomposition
- A square matrix $A$ can be decomposed as $A = V \Lambda V^{-1}$ where $\Lambda$ is diagonal (eigenvalues) and $V$ contains eigenvectors
- **Eigenvalue equation**: $Av = \lambda v$ — eigenvector $v$ is only scaled (not rotated) by $A$
- Example: for rotation matrix, eigenvalues reveal the principal axes; for covariance matrix, eigenvalues = variance along each principal direction
- Used directly in PCA (Module 2): the top-$k$ eigenvectors of the covariance matrix $C = \frac{1}{n}X^TX$ give the principal components

### Singular Value Decomposition (SVD)
- Any matrix $A_{m \times n} = U \Sigma V^T$ where $U, V$ are orthogonal and $\Sigma$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \cdots$
- **Truncated SVD**: keep only top-$k$ singular values for low-rank approximation
- **Spectral norm** = largest singular value $\sigma_1(W)$ — used in Spectral Normalization for GANs (Module 7) to enforce Lipschitz constraint: $W' = W / \sigma_1(W)$

### Determinants & Jacobians
- Determinant measures how a linear transformation scales volume: $\det(A) = 0$ means the transformation collapses a dimension
- **Jacobian matrix** $J_{ij} = \frac{\partial f_i}{\partial x_j}$ — matrix of all first-order partial derivatives of a vector-valued function
- **Jacobian determinant** measures how a nonlinear transformation stretches/compresses probability density locally
- Example: if $f(x,y) = (2x, 3y)$, then $J = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ and $\det(J) = 6$ — the transformation scales area by 6×
- Critical for Normalizing Flows (Module 5): $\log p_X(x) = \log p_Z(z) + \log|\det J^{-1}|$

### Quadratic Forms & Completing the Square
- Completing the square: $-ax^2 + bx = -a(x - \frac{b}{2a})^2 + \frac{b^2}{4a}$
- Used to compute partition functions (Gaussian integrals) in Module 9: $\int e^{-ax^2+bx}dx = \sqrt{\frac{\pi}{a}} e^{b^2/(4a)}$
- Example: for energy $E(x) = 2x^2 - 4x$, complete the square to get $E(x) = 2(x-1)^2 - 2$, then $Z = \int e^{-E(x)}dx = e^2 \sqrt{\pi/2}$

---

## Mathematics — Probability & Statistics

### Probability Distributions
- **Gaussian/Normal**: $p(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$ — the most commonly assumed distribution in ADL (latent priors in VAEs, noise in diffusion)
- **Bernoulli**: $p(x=1) = p$, $p(x=0) = 1-p$ — used for binary pixel data, MLE derivation in Module 4
- **Categorical/Multinomial**: generalization of Bernoulli to $K$ classes — used in language modeling softmax outputs (Module 10)
- **Mixture of Gaussians**: $p(x) = \sum_k \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$ — tractable latent variable model example in Module 6
- Example: MLE for Bernoulli with 7 successes in 10 trials → $\hat{p} = 7/10 = 0.7$

### Bayes' Theorem
- $P(z|x) = \frac{P(x|z) P(z)}{P(x)}$ — posterior = (likelihood × prior) / evidence
- The **posterior** $P(z|x)$ is what VAEs approximate (Module 6) because $P(x) = \int P(x|z)P(z)dz$ is intractable
- The **approximate posterior** $q_\phi(z|x)$ is the encoder network in a VAE

### Maximum Likelihood Estimation (MLE)
- Find parameters $\theta$ maximizing $\mathcal{L}(\theta) = \sum_{i=1}^N \log p_\theta(x_i)$
- Procedure: write log-likelihood → take derivative → set to zero → solve
- Example (Gaussian MLE): given data $\{x_1, ..., x_n\}$, the MLE estimates are $\hat{\mu} = \frac{1}{n}\sum x_i$ (sample mean) and $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \hat{\mu})^2$ (sample variance)
- MLE is equivalent to minimizing KL divergence $KL(p_{\text{data}} \| p_\theta)$ — this connection motivates the training of AR models (Module 4) and EBMs (Module 9)

### KL Divergence
- $KL(P \| Q) = \mathbb{E}_P\left[\log \frac{P(x)}{Q(x)}\right] = \sum P(x) \log\frac{P(x)}{Q(x)}$
- **Asymmetric**: $KL(P\|Q) \neq KL(Q\|P)$ in general. Always $\geq 0$, equals 0 iff $P=Q$
- **Closed form for two Gaussians**: $KL(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$
- Special case (VAE KL vs standard normal): $KL(q \| \mathcal{N}(0,1)) = -\frac{1}{2}\sum(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$
- Used in ELBO (Module 6), GAN theory (Module 7 — JSD is built from two KL terms), AR model training (Module 4)

### Jensen-Shannon Divergence (JSD)
- $JSD(P \| Q) = \frac{1}{2}KL(P \| M) + \frac{1}{2}KL(Q \| M)$ where $M = \frac{1}{2}(P + Q)$
- **Symmetric** (unlike KL) and bounded: $0 \leq JSD \leq \log 2$
- The original GAN objective at optimal discriminator minimizes JSD between real and generated distributions (Module 7)

### Chain Rule of Probability
- $p(x_1, x_2, ..., x_D) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1,x_2) \cdots = \prod_{d=1}^D p(x_d | x_{<d})$
- This is the entire foundation of autoregressive models (Module 4): model each conditional with a neural network
- Example: for a 3-pixel image, $p(x_1, x_2, x_3) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1,x_2)$

### Importance Sampling
- Estimate $\mathbb{E}_{p}[f(x)]$ using samples from a different distribution $q$: $\mathbb{E}_p[f(x)] = \mathbb{E}_q\left[\frac{p(x)}{q(x)} f(x)\right]$
- Used in VAE training (Module 6) — importance-weighted autoencoders (IWAE) use multiple importance samples to tighten the ELBO bound
- Key intuition: if $q$ is close to $p$, variance is low → better estimates

### Conditional Independence & Markov Property
- $X \perp Y | Z$ means $X$ and $Y$ are independent given $Z$
- **Markov property**: future depends only on present, not past — $p(x_t | x_{t-1}, x_{t-2}, ...) = p(x_t | x_{t-1})$
- Used in diffusion models (Module 8): forward/reverse processes are Markov chains
- RBMs (Module 9) exploit conditional independence for efficient Gibbs sampling

---

## Mathematics — Calculus & Optimization

### Gradients & Chain Rule
- $\nabla_x f(x) = \left[\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n}\right]$ — direction of steepest ascent
- **Chain rule**: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$ — the backbone of backpropagation
- Example: for $L = (y - \hat{y})^2$ and $\hat{y} = wx + b$, $\frac{\partial L}{\partial w} = 2(y - \hat{y})(-x)$

### Gradient Descent Variants
- **SGD**: $\theta \leftarrow \theta - \eta \nabla_\theta L$ — update in direction of negative gradient
- **Adam**: adaptive learning rates per parameter using first/second moment estimates — the default optimizer for most ADL models
- Example: learning rate $\eta = 0.001$ is typical for Adam; too large → divergence, too small → slow convergence

### Gaussian Integrals
- $\int_{-\infty}^{\infty} e^{-ax^2} dx = \sqrt{\frac{\pi}{a}}$ for $a > 0$
- Generalizes to: $\int e^{-ax^2 + bx} dx = \sqrt{\frac{\pi}{a}} e^{b^2/(4a)}$ via completing the square
- This is not just theory — you need this to compute partition functions by hand for EBM exam problems (Module 9)

---

## Mathematics — Information Theory

### Entropy
- $H(X) = -\sum_x p(x) \log p(x)$ — measures uncertainty/randomness of a distribution
- Maximum entropy for discrete uniform distribution, minimum (0) for deterministic
- Example: fair coin → $H = -2 \cdot 0.5 \log_2 0.5 = 1$ bit; biased coin ($p=0.9$) → $H \approx 0.47$ bits

### Cross-Entropy
- $H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + KL(P \| Q)$
- Cross-entropy loss is the standard loss for classification (softmax output) and is used as reconstruction loss for binary data in autoencoders (Module 3)
- Example: true label = [1, 0, 0], prediction = [0.7, 0.2, 0.1] → $CE = -\log(0.7) \approx 0.357$

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
- Example: K-Means on MNIST handwritten digits to discover digit clusters without labels

### Regularization
- **L1 (Lasso)**: $\lambda \sum |w_i|$ — promotes sparsity (some weights go to exactly zero)
- **L2 (Ridge)**: $\lambda \sum w_i^2$ — penalizes large weights, prevents overfitting
- **Dropout**: randomly set fraction of neurons to zero during training — acts as ensemble
- L1 regularization is used in Sparse PCA (Module 2) and Sparse Autoencoder (Module 3)

### EM Algorithm
- **Expectation-Maximization**: iterative algorithm for MLE with latent variables
- **E-step**: compute expected value of latent variables given current parameters
- **M-step**: maximize likelihood w.r.t. parameters given expected latent variables
- Used to train Probabilistic PCA and Factor Analysis (Module 2), and conceptually related to VAE training (Module 6)
- Example: fitting a Gaussian mixture model — E-step computes cluster responsibilities, M-step updates means/covariances/weights

### Kernel Methods
- **Kernel trick**: compute inner products in high-dimensional feature space without explicit transformation — $\kappa(x_i, x_j) = \phi(x_i)^T \phi(x_j)$
- **Common kernels**: RBF/Gaussian $\kappa(x,y) = \exp(-\gamma\|x-y\|^2)$, Polynomial $\kappa(x,y) = (x^Ty + c)^d$
- Used in Kernel PCA (Module 2) to capture nonlinear structure in data

---

## Deep Learning Fundamentals

### Neural Network Architecture & Parameter Counting
- **MLP**: stacks of fully connected layers — parameters per layer = $(D_{in} + 1) \times D_{out}$ (the +1 is for bias)
- Example: layer with 784 inputs and 256 outputs → $784 \times 256 + 256 = 200{,}960$ parameters
- You should be able to count parameters through an entire encoder-decoder network, including Conv2D, ConvTranspose2D, Dense, Flatten, and Reshape layers (Module 3 tests this explicitly)

### Backpropagation
- Forward pass: compute outputs layer by layer; backward pass: compute gradients layer by layer using chain rule
- **Computational graph**: DAG of operations — automatic differentiation frameworks (PyTorch autograd) traverse this graph
- You should understand how gradients flow through common operations (matrix multiply, activation, loss)

### Activation Functions

**Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Output range: $(0, 1)$ — interprets as probability
- Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- Used as: output activation for binary data (Module 3), discriminator output in GANs (Module 7), gate activations in LSTM/GRU, RBM conditionals (Module 9)
- Issue: vanishing gradients when $|x|$ is large (saturation)

**Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- Output range: $(-1, 1)$ — zero-centered unlike sigmoid
- Derivative: $\tanh'(x) = 1 - \tanh^2(x)$
- Used as: output activation for normalized data (Module 3), gated activations in WaveNet/PixelCNN (Module 4), planar flow activation (Module 5)
- Example: for Planar Flow, $h'(a) = 1 - \tanh^2(a)$ appears directly in the Jacobian determinant formula

**ReLU**: $\text{ReLU}(x) = \max(0, x)$
- Output range: $[0, \infty)$; derivative = 0 for $x < 0$, 1 for $x > 0$
- Default choice for hidden layers — no vanishing gradient for positive inputs
- Used in DCGAN generator hidden layers (Module 7), encoder/decoder hidden layers (Module 3)
- Issue: "dying ReLU" — neurons stuck at zero if they always get negative input

**LeakyReLU**: $\text{LeakyReLU}(x) = \max(\alpha x, x)$ where $\alpha \approx 0.01$
- Fixes dying ReLU by allowing small gradient for negative inputs
- Used specifically in DCGAN discriminator (Module 7)

**ELU / GELU**: smooth alternatives to ReLU
- ELU: $\alpha(e^x - 1)$ for $x < 0$ — smooth, allows negative outputs
- GELU: $x \cdot \Phi(x)$ where $\Phi$ is Gaussian CDF — used in Transformers (Module 10)

**Softmax**: $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
- Converts logits to probability distribution over $K$ classes (all outputs sum to 1)
- Used in attention mechanism (Module 10), AR model outputs (Module 4), language model predictions (Module 10)
- Example: logits $[2.0, 1.0, 0.1]$ → softmax ≈ $[0.659, 0.242, 0.099]$

### Loss Functions

**MSE (Mean Squared Error)**: $\frac{1}{n}\sum_{i=1}^n \|x_i - \hat{x}_i\|^2$
- Default for continuous reconstruction — used in autoencoders (Module 3), VQ-VAE codebook loss (Module 6), DDPM noise prediction (Module 8)
- Gradient: $\frac{\partial}{\partial \hat{x}} = \frac{2}{n}(\hat{x} - x)$

**Binary Cross-Entropy (BCE)**: $-\sum[x \log \hat{x} + (1-x) \log(1-\hat{x})]$
- For binary/image data in $[0,1]$ — requires sigmoid output
- Used in autoencoder reconstruction (Module 3), GAN discriminator loss (Module 7)
- Example: true pixel = 1, predicted = 0.9 → $-[\log(0.9) + 0] \approx 0.105$

**Negative Log-Likelihood (NLL)**: $-\sum_i \log p_\theta(x_i)$
- The general training objective for generative models
- For autoregressive models (Module 4): $-\sum_i \sum_d \log p_\theta(x_{id} | x_{i,<d})$
- Perplexity = $\exp(\text{NLL per token})$ — used to evaluate language models (Module 10)

### Convolutional Neural Networks (CNNs)

**Conv2D**:
- Parameters per layer: $(K_h \times K_w \times C_{in} + 1) \times C_{out}$
- Output spatial size: $\lfloor \frac{H + 2p - K}{s} \rfloor + 1$
- Example: 3×3 kernel, 64 input channels, 128 output channels → $(3 \times 3 \times 64 + 1) \times 128 = 73{,}856$ parameters

**ConvTranspose2D** (Transposed Convolution):
- Used in decoder/generator networks to **upsample** feature maps
- Same parameter formula as Conv2D, but output spatial size is larger: $(H_{in}-1) \times s - 2p + K + p_{out}$
- Used in autoencoder decoders (Module 3) and GAN generators (Module 7)

**Dilated Convolutions**:
- Insert gaps (dilation) in the kernel — receptive field grows exponentially with depth while keeping parameter count fixed
- Dilation rates 1, 2, 4, 8, 16… → receptive field of $2^L$ with $L$ layers
- Used in WaveNet (Module 4) for audio generation

**1×1 Convolutions**:
- Channel mixing without spatial interaction — equivalent to a pointwise fully connected layer across channels
- Used in Glow (Module 5) as invertible 1×1 convolutions for learnable channel permutations

### Recurrent Neural Networks (RNNs)

**Vanilla RNN**: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
- Processes sequences step-by-step, hidden state $h_t$ summarizes history
- Problem: **vanishing/exploding gradients** for long sequences

**LSTM (Long Short-Term Memory)**:
- **Cell state** $c_t$ + three gates: forget ($f_t$), input ($i_t$), output ($o_t$)
- $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ — what to forget
- $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ — what new info to store
- $c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c)$ — update cell
- $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$, $h_t = o_t \odot \tanh(c_t)$
- Used in AR models for sequential generation (Module 4), VRNN (Module 11)

**GRU (Gated Recurrent Unit)**:
- Simplified LSTM with two gates (reset, update) — often comparable performance
- Fewer parameters than LSTM

**BiLSTM**: runs forward and backward LSTMs, concatenates/sums their hidden states
- Captures both left and right context — used in CoVe (Module 10)
- Example: $\text{CoVe}(x_t) = h_t^f + h_t^b$ where $h^f$ = forward, $h^b$ = backward hidden state

### Training Techniques

**Batch Normalization**: normalize activations across the mini-batch to have zero mean and unit variance, then scale and shift with learned parameters $\gamma, \beta$
- Stabilizes training, allows higher learning rates — used in DCGAN (Module 7), deep autoencoders (Module 3)
- Applied to all layers except generator output and discriminator input in DCGAN

**Layer Normalization**: normalize across features (not batch) — used in Transformer blocks (Module 10)
- Advantage over BatchNorm: works with variable-length sequences and small batches

**Instance Normalization**: normalize each sample's spatial dimensions independently — used in style transfer, CycleGAN (Module 7)

**Residual / Skip Connections**: $y = F(x) + x$ — gradient flows directly through the skip path
- Enables training very deep networks — used in deep autoencoders (Module 3), U-Net architecture (Module 8), Transformer blocks (Module 10), WaveNet (Module 4)

**Dropout**: randomly zero out neurons with probability $p$ during training — acts as regularization via implicit ensemble
- Masking noise in Denoising Autoencoders (Module 3) is conceptually similar

---

## Dimensionality Reduction (Pre-Module 2)

### Principal Component Analysis (PCA)
- **Covariance matrix**: $C = \frac{1}{n} X^T X$ — captures variance and linear relationships between features
- **PCA procedure**: center data → compute covariance matrix → eigendecomposition → project onto top-$k$ eigenvectors
- **Projection**: $Z = XW$ where $W \in \mathbb{R}^{d \times k}$ contains top-$k$ eigenvectors
- **Reconstruction**: $\hat{X} = ZW^T$ — approximation of original data
- **Explained variance ratio**: $\frac{\lambda_i}{\sum_j \lambda_j}$ — fraction of total variance captured by each component
- Example: 100-dimensional data with 90% variance in first 5 components → PCA reduces to 5 dimensions with minimal information loss
- Module 2 builds on this with Randomized, Incremental, Kernel, Probabilistic, and Sparse variants

### t-SNE / UMAP
- Nonlinear dimensionality reduction for **visualization** (typically to 2D/3D)
- t-SNE preserves local structure (nearby points stay nearby), UMAP preserves both local and some global structure
- Referenced in Manifold Learning (Module 2) as comparison methods

### Manifold Hypothesis
- Real-world high-dimensional data (images, text) lies near a **low-dimensional manifold** embedded in the high-dimensional space
- Example: the space of natural face images is tiny compared to all possible pixel arrangements
- This motivates the entire field of representation learning and generative modeling (Modules 1–9)

---

## Probabilistic Graphical Models (Pre-Module 9)

### Directed vs Undirected Models
- **Directed (Bayesian Networks)**: edges have direction, represent causal/generative relationships — VAEs use directed models ($z \to x$)
- **Undirected (Markov Random Fields)**: edges are symmetric — Boltzmann machines, Ising model (Module 9)
- Key difference: directed models have tractable sampling but may have intractable inference; undirected models have intractable partition function

### Markov Chains
- Sequence of states where next state depends only on current state: $p(x_{t+1} | x_t, x_{t-1}, ...) = p(x_{t+1} | x_t)$
- **Transition matrix**: $T_{ij} = p(x_{t+1} = j | x_t = i)$
- **Stationary distribution** $\pi$: $\pi T = \pi$ — the distribution the chain converges to
- Diffusion models (Module 8) use forward/reverse Markov chains: forward gradually adds noise, reverse denoises

### MCMC Methods
- **Gibbs Sampling**: sample each variable conditioned on all others — used in RBM training (Module 9)
  - Example: for RBM, alternate between sampling $h | v$ and $v | h$ (both are easy due to conditional independence)
- **Langevin Dynamics**: gradient-based MCMC — $x_{k+1} = x_k - \frac{\eta}{2}\nabla_x E(x_k) + \sqrt{\eta}\epsilon$ — used for EBM sampling (Module 9)
- **Contrastive Divergence (CD-$k$)**: approximate MCMC starting from data instead of random initialization — used for RBM training (Module 9)

---

## Natural Language Processing (Pre-Module 10)

### Tokenization
- Converting text into tokens (units the model processes)
- **Word-level**: "the cat sat" → ["the", "cat", "sat"] — simple but large vocabulary, OOV problem
- **Subword (BPE/WordPiece)**: "unhappiness" → ["un", "happi", "ness"] — handles rare words, used in BERT/GPT
- **Vocabulary size** $V$ determines embedding matrix size and softmax output dimension

### Word Embeddings Concept
- Represent words as dense vectors $\mathbb{R}^d$ (typically $d$ = 100–300) instead of sparse one-hot vectors $\mathbb{R}^V$
- Similar words have similar vectors (cosine similarity)
- Module 10 covers word2Vec (CBOW, Skip-gram), GloVe, and contextual embeddings (CoVe, BERT, GPT)
- Example: $\cos(\text{king}, \text{queen}) > \cos(\text{king}, \text{apple})$

### N-gram Language Models
- Approximate $p(w_t | w_1, ..., w_{t-1})$ using only last $n-1$ words: $p(w_t | w_{t-n+1}, ..., w_{t-1})$
- **Bigram**: $p(w_t | w_{t-1})$; **Trigram**: $p(w_t | w_{t-2}, w_{t-1})$
- Limitations: fixed context window, data sparsity, no generalization — motivates neural language models (Module 10)

### Attention Mechanism
- **Query-Key-Value**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
- $Q$ asks "what am I looking for?", $K$ answers "what do I contain?", $V$ is "what do I provide?"
- Scaling by $\sqrt{d_k}$ prevents softmax saturation
- **Multi-head attention**: run $h$ parallel attention heads, concatenate, project — allows attending to different relationship types simultaneously
- Foundation of the Transformer architecture (Module 10), later used in diffusion U-Net (Module 8), Flow++ (Module 5)

### Positional Encoding
- Transformers have no inherent notion of sequence order (unlike RNNs)
- **Sinusoidal encoding**: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$, $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$
- Or **learned embeddings**: trainable position vectors added to token embeddings

---

## Time Series Analysis (Pre-Module 11)

### Stationarity
- A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time
- Non-stationary series need **differencing** (the "I" in ARIMA) to become stationary
- Example: stock prices are non-stationary (trending), but daily returns are approximately stationary

### ARIMA Components
- **AR($p$)**: current value depends on $p$ previous values — $X_t = \phi_1 X_{t-1} + ... + \phi_p X_{t-p} + \epsilon_t$
- **I($d$)**: differencing $d$ times to achieve stationarity — $(1-B)^d X_t$
- **MA($q$)**: current value depends on $q$ previous error terms — $X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}$
- **Backshift operator**: $B \cdot X_t = X_{t-1}$, $B^2 \cdot X_t = X_{t-2}$ — compact notation used in Module 11
- **S-ARIMA**: adds seasonal AR/MA/differencing components with period $s$

### ACF and PACF
- **Autocorrelation Function (ACF)**: correlation of series with its lagged values — helps determine MA order $q$
- **Partial Autocorrelation Function (PACF)**: correlation controlling for intermediate lags — helps determine AR order $p$

---

## Game Theory (Pre-Module 7)

### Minimax Games
- Two-player zero-sum game: one player's gain = other's loss
- **Minimax strategy**: Player 1 minimizes the maximum loss that Player 2 can inflict: $\min_G \max_D V(D, G)$
- This is exactly the GAN objective (Module 7): Generator minimizes, Discriminator maximizes the same value function
- Example: in rock-paper-scissors, the minimax strategy is to play each option with probability 1/3

### Nash Equilibrium
- A state where **no player can improve** by unilaterally changing strategy
- For GANs (Module 7): equilibrium is when $p_G = p_{\text{data}}$ and $D(x) = 0.5$ everywhere
- In practice, GANs rarely converge to Nash equilibrium — leading to **mode collapse** and **training instability**

---

## Physics & Statistical Mechanics (Pre-Module 9)

### Boltzmann Distribution
- $p(x) = \frac{1}{Z} e^{-E(x)/T}$ — probability is exponentially related to negative energy
- Lower energy states are more probable; temperature $T$ controls the sharpness
- In ADL (Module 9): $p_\theta(x) = \frac{1}{Z(\theta)} \exp(-E_\theta(x))$ — energy function is parameterized by a neural network

### Partition Function
- $Z(\theta) = \int \exp(-E_\theta(x)) dx$ — normalization constant ensuring probabilities sum to 1
- **Intractable** for most models — computing it requires integrating over all possible states
- This intractability motivates score-based models (Module 9): the score $\nabla_x \log p(x) = -\nabla_x E(x)$ doesn't depend on $Z$

### Langevin Dynamics
- $x_{k+1} = x_k - \frac{\eta}{2} \nabla_x E(x_k) + \sqrt{\eta} \epsilon_k$ where $\epsilon_k \sim \mathcal{N}(0, I)$
- Combines gradient descent on energy (moves toward high-probability regions) with random noise (for exploration)
- As step size $\eta \to 0$ and steps $\to \infty$, samples converge to $p(x) \propto e^{-E(x)}$
- Used for sampling in EBMs and score-based models (Module 9)

---

## Programming & Tools

### Python
- Proficiency in Python 3.x: classes, list comprehensions, generators, decorators
- Understanding of broadcasting semantics in array operations

### NumPy
- Array operations, reshaping, slicing, broadcasting
- Random sampling: `np.random.normal()`, `np.random.uniform()`
- Linear algebra: `np.linalg.eig()`, `np.linalg.svd()`, `np.dot()`

### PyTorch
- **Tensors**: creation, GPU transfer (`.to('cuda')`), gradients (`.requires_grad`)
- **Autograd**: automatic differentiation, `loss.backward()`, `optimizer.step()`
- **nn.Module**: building models with `__init__` + `forward`, registering parameters
- **Layers**: `nn.Linear`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.BatchNorm2d`, `nn.LayerNorm`
- **DataLoader**: batching, shuffling, parallel data loading
- **Training loop**: forward pass → loss computation → backward pass → optimizer step → zero gradients
- Example: you should be able to write a training loop for a simple CNN classifier from scratch

### Visualization
- **Matplotlib**: plotting loss curves, grid of generated images, latent space visualizations
- Useful for debugging: plot reconstruction quality, training/validation loss, generated samples per epoch
