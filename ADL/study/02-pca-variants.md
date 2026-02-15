# Module 2 — PCA Variants

## Topics

- [[#2.2 Randomized PCA|Randomized PCA]]
- [[#2.3 Incremental PCA|Incremental PCA]]
- [[#2.4 Kernel PCA|Kernel PCA]]
- [[#2.5 Probabilistic PCA|Probabilistic PCA]]
- [[#2.6 Sparse PCA|Sparse PCA]]
- [[#2.7 Canonical Correlation Analysis (CCA)|Canonical Correlation Analysis (CCA)]]
- [[#2.8 Locally Linear Embedding (LLE)|Locally Linear Embedding (LLE)]]
- [[#2.9 Independent Component Analysis (ICA)|Independent Component Analysis]]
- [[#2.10 Factor Analysis|Factor Analysis]]
- [[#2.11 Manifold Learning|Manifold learning]]

---

## 2.1 Classical PCA Recap

**Principal Component Analysis** finds orthogonal directions of maximum variance.

Given data matrix $X \in \mathbb{R}^{n \times d}$, PCA computes the eigendecomposition of the covariance matrix:

$$C = \frac{1}{n} X^T X = V \Lambda V^T$$

The top $k$ eigenvectors form the projection matrix $W \in \mathbb{R}^{d \times k}$, and the reduced representation is:

$$Z = X W$$

---

## 2.2 Randomized PCA

- Uses **randomized algorithms** to approximate the top-$k$ singular vectors
- Much faster than full SVD for large datasets: $O(ndk)$ vs $O(nd \min(n,d))$
- Algorithm:
  1. Generate random matrix $\Omega \in \mathbb{R}^{d \times k}$
  2. Form $Y = X \Omega$
  3. Compute QR decomposition: $Y = QR$
  4. Project: $B = Q^T X$
  5. Compute SVD of small matrix $B$
- Provides a good approximation when eigenvalue spectrum decays quickly

---

## 2.3 Incremental PCA

- Processes data in **mini-batches** — does not require full dataset in memory
- Useful for **large datasets** or **streaming data**
- Updates the principal components incrementally as new batches arrive
- Trades slight accuracy for memory efficiency

---

## 2.4 Kernel PCA

- Applies PCA in a **high-dimensional feature space** using the kernel trick
- Computes eigendecomposition of the **kernel matrix** $K$ instead of covariance matrix
- Common kernels: RBF (Gaussian), polynomial, sigmoid

$$K_{ij} = \kappa(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

- Can capture **nonlinear** structure in data
- Limitation: requires $O(n^2)$ kernel matrix computation and storage

---

## 2.5 Probabilistic PCA

- Formulates PCA as a **latent variable model**:

$$x = Wz + \mu + \epsilon$$

where $z \sim \mathcal{N}(0, I)$ and $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

- Parameters $W, \mu, \sigma^2$ are learned via **maximum likelihood** or **EM algorithm**
- Advantages:
  - Handles **missing data** naturally
  - Provides a **probabilistic framework** for PCA
  - Can be used for **density estimation**
  - Connection to **Factor Analysis** and **VAEs**

---

## 2.6 Sparse PCA

- Adds **sparsity constraint** (L1 penalty) to the loading vectors
- Results in principal components that are linear combinations of only a **few original features**
- Improves **interpretability** at the cost of explained variance
- Formulation:

$$\min_{W, Z} \|X - ZW^T\|_F^2 + \lambda \|W\|_1$$

---

## 2.7 Canonical Correlation Analysis (CCA)

- Finds **linear combinations** of two sets of variables that are **maximally correlated**
- Given two views $X_1$ and $X_2$, find $w_1, w_2$ such that:

$$\max_{w_1, w_2} \text{corr}(X_1 w_1, X_2 w_2)$$

- Applications: multi-view learning, cross-modal retrieval
- Deep CCA: uses neural networks instead of linear projections

---

## 2.8 Locally Linear Embedding (LLE)

- **Nonlinear dimensionality reduction** that preserves local neighborhood structure
- Algorithm:
  1. Find $k$ nearest neighbors for each point
  2. Compute reconstruction weights: $x_i \approx \sum_j w_{ij} x_j$
  3. Find low-dimensional embedding $Y$ that preserves these weights:

$$\min_Y \sum_i \left\| y_i - \sum_j w_{ij} y_j \right\|^2$$

- Assumes data lies on a **smooth manifold**
- Non-parametric: cannot embed new points directly

---

## 2.9 Independent Component Analysis (ICA)

- Finds **statistically independent** components (not just uncorrelated like PCA)
- Assumes data is a **linear mixture** of independent sources:

$$x = As$$

where $s$ are independent non-Gaussian sources
- Goal: recover $s = Wx$ where $W = A^{-1}$
- Key assumption: at most one source is Gaussian
- Applications: blind source separation (cocktail party problem), EEG/fMRI analysis
- Algorithms: FastICA, Infomax

---

## 2.10 Factor Analysis

- Similar to Probabilistic PCA but with **diagonal** (not isotropic) noise:

$$x = Wz + \mu + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \Psi)$$

where $\Psi = \text{diag}(\psi_1, \ldots, \psi_d)$

- Each observed variable has its own noise variance
- Better model when variables have different noise levels
- Solved via EM algorithm

---

## 2.11 Manifold Learning

- Assumes high-dimensional data lies on a **low-dimensional manifold**
- Goal: discover this manifold and provide a low-dimensional representation

### Key Methods
| Method | Key Idea |
|--------|----------|
| **Isomap** | Preserves geodesic distances |
| **LLE** | Preserves local linear relationships |
| **Laplacian Eigenmaps** | Preserves local distances via graph Laplacian |
| **t-SNE** | Preserves local structure; used for visualization |
| **UMAP** | Fast, preserves both local and global structure |

### Manifold Hypothesis
> Real-world high-dimensional data (images, text) concentrates near a low-dimensional manifold. Learning this manifold is key to representation learning.

---

## Key Takeaways

- PCA has many variants optimized for different scenarios (speed, nonlinearity, sparsity, streaming)
- **Kernel PCA** and **manifold learning** methods handle nonlinear structure
- **Probabilistic PCA** and **Factor Analysis** provide generative probabilistic models
- These methods form the foundation for understanding deeper generative models (autoencoders, VAEs)

---

## References

- T2: Goodfellow et al., Ch. 13 — Linear Factor Models
- R1: Géron, Ch. 8 — Dimensionality Reduction
