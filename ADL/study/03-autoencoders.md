# Module 3 — Autoencoders

## Topics

- Type of activation and loss functions
- Undercomplete vs. Overcomplete autoencoders
- Relationship with PCA
- Regularization
  - Denoising autoencoder
  - Sparse autoencoder
  - Contractive Autoencoders
- Effect of Depth
- Application of Autoencoders

---

## 3.1 Autoencoder Architecture

An autoencoder learns to **compress** and **reconstruct** data through an encoder-decoder framework:

$$\text{Encoder: } z = f_\theta(x), \quad \text{Decoder: } \hat{x} = g_\phi(z)$$

$$\min_{\theta, \phi} \mathcal{L}(x, g_\phi(f_\theta(x)))$$

### Components
- **Encoder** $f_\theta$: maps input $x$ to latent code $z$
- **Bottleneck**: the latent representation $z$ (lower dimensional)
- **Decoder** $g_\phi$: reconstructs $\hat{x}$ from $z$

---

## 3.2 Activation and Loss Functions

### Activation Functions
| Layer | Common Choices |
|-------|---------------|
| Hidden (encoder/decoder) | ReLU, LeakyReLU, ELU, GELU |
| Output (binary data) | Sigmoid |
| Output (continuous data) | Linear (identity) |
| Output (normalized data) | Tanh |

### Loss Functions
| Data Type | Loss Function | Formula |
|-----------|--------------|---------|
| Continuous | MSE (Mean Squared Error) | $\mathcal{L} = \frac{1}{n}\sum_i \|x_i - \hat{x}_i\|^2$ |
| Binary / images [0,1] | Binary Cross-Entropy | $\mathcal{L} = -\sum_i [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$ |
| Normalized | Cosine similarity loss | $\mathcal{L} = 1 - \frac{x \cdot \hat{x}}{\|x\|\|\hat{x}\|}$ |

---

## 3.3 Undercomplete vs. Overcomplete Autoencoders

### Undercomplete Autoencoder
- Bottleneck dimension $\dim(z) < \dim(x)$
- **Forces compression** — learns most important features
- Without regularization, a linear undercomplete AE learns the same subspace as PCA

### Overcomplete Autoencoder
- Bottleneck dimension $\dim(z) \geq \dim(x)$
- Risk: can learn the **identity function** (trivial solution)
- Must use **regularization** to learn useful representations
- Allows more expressive latent space

---

## 3.4 Relationship with PCA

| Property | Linear AE | PCA |
|----------|-----------|-----|
| Objective | Minimize reconstruction error | Maximize variance |
| Solution space | Same subspace as PCA | Principal components |
| Nonlinearity | No (linear activation) | No |
| Orthogonal components | Not guaranteed | Yes |

**Key result**: A linear autoencoder with MSE loss and $k$-dimensional bottleneck learns the same subspace as the top-$k$ PCA components (though the basis vectors may differ by rotation).

**Deep nonlinear autoencoders** can learn nonlinear manifolds that PCA cannot capture.

---

## 3.5 Regularization

### 3.5.1 Denoising Autoencoder (DAE)

- Corrupts input $\tilde{x} = x + \text{noise}$ and trains to reconstruct clean $x$
- Loss: $\mathcal{L} = \|x - g_\phi(f_\theta(\tilde{x}))\|^2$
- Noise types: Gaussian noise, masking noise (dropout), salt-and-pepper
- Forces the model to learn **robust features** rather than identity
- Learns to project corrupted points back onto the data manifold
- Connection to **score matching**: DAE implicitly learns $\nabla_x \log p(x)$

### 3.5.2 Sparse Autoencoder

- Adds sparsity penalty on activations of the hidden layer:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \sum_j |h_j|$$

or using KL divergence against a target sparsity $\rho$:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \beta \sum_j KL(\rho \| \hat{\rho}_j)$$

where $\hat{\rho}_j = \frac{1}{n}\sum_i h_j(x_i)$ is the average activation

- Encourages only a few neurons to be active for any given input
- Learns **overcomplete but sparse** representations

### 3.5.3 Contractive Autoencoder (CAE)

- Penalizes the **Frobenius norm of the Jacobian** of the encoder:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \left\| \frac{\partial f_\theta(x)}{\partial x} \right\|_F^2$$

- Makes the representation **insensitive to small input perturbations**
- Encourages the encoder to learn a **locally flat mapping** — contracts the representation around the data manifold
- Relation to DAE: both encourage robustness, but CAE does it analytically

---

## 3.6 Effect of Depth

- **Deeper autoencoders** can learn more complex, hierarchical representations
- Benefits:
  - Exponentially more efficient compression
  - Better generalization
  - Hierarchical feature extraction (edges → textures → objects)
- Challenges:
  - Harder to train (vanishing gradients)
  - Need careful initialization (greedy layer-wise pretraining was historically used)
  - Modern solution: batch normalization, skip connections, better optimizers

---

## 3.7 Applications of Autoencoders

| Application | How |
|-------------|-----|
| **Dimensionality reduction** | Use bottleneck as low-dim representation |
| **Anomaly detection** | High reconstruction error → anomaly |
| **Denoising** | Train DAE, use for noise removal |
| **Feature learning** | Use encoder output as features for downstream tasks |
| **Image compression** | Learned compression better than JPEG for specific domains |
| **Data generation** | VAEs (Module 6) extend AEs for generation |
| **Pretraining** | Pretrain encoder, fine-tune for supervised tasks |
| **Recommendation systems** | Reconstruct user-item interaction matrices |

---

## Key Takeaways

- Autoencoders learn compressed representations through reconstruction
- **Undercomplete** AEs force compression; **overcomplete** AEs need regularization
- Linear AEs ≈ PCA; nonlinear AEs capture richer structure
- Regularization (denoising, sparse, contractive) prevents trivial solutions and improves representations
- Foundation for **VAEs** (Module 6) which add probabilistic interpretation

---

## References

- T2: Goodfellow et al., Ch. 14 — Autoencoders
- T1: Prince, Ch. 17 — Autoencoders
- R1: Géron, Ch. 17 — Autoencoders and GANs
