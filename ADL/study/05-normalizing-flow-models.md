# Module 5 — Normalizing Flow Models

## Topics

- [[#5.1 Difference with AR Models|Difference with AR Models]]
- [[#5.2 Foundations of 1-D Flow|Foundations of 1-D Flow]]
- [[#5.3 2-D Flow|2-D Flow]]
- [[#5.4 N-Dimensional Flows|N-dimensional flows]]
  - [[#5.4.1 AR and Inverse AR Flows|AR and inverse AR flows]]
  - [[#5.4.2 NICE / RealNVP|NICE / RealNVP]]
  - [[#5.4.3 Glow, Flow++|Glow, Flow++]]
- [[#5.5 Dequantization|Dequantization]]
- [[#5.6 Applications|Applications]]

---

## 5.1 Difference with AR Models

| Property | Autoregressive Models | Normalizing Flows |
|----------|----------------------|-------------------|
| **Likelihood** | Exact (chain rule) | Exact (change of variables) |
| **Latent space** | No explicit latent space | Explicit latent space $z$ |
| **Generation** | Sequential (slow) | Single pass (fast) |
| **Inference** | Single pass (fast) | Depends on architecture |
| **Invertibility** | Not required | Required (bijective) |

- Both compute **exact log-likelihoods**, but via different mechanisms
- Flows provide a **bidirectional mapping** between data and latent space

---

## 5.2 Foundations of 1-D Flow

### Change of Variables (1D)

Given a bijective mapping $x = f(z)$ where $z \sim p_Z(z)$:

$$p_X(x) = p_Z(f^{-1}(x)) \left\| \frac{df^{-1}}{dx} \right\|$$

Or equivalently:

$$p_X(x) = p_Z(z) \left\| \frac{dz}{dx} \right\| = p_Z(z) \left\| \frac{dx}{dz} \right\|^{-1}$$

- The absolute value of the derivative accounts for how $f$ **stretches or compresses** probability density
- $z$ typically drawn from a simple base distribution (e.g., standard Gaussian)

---

## 5.3 2-D Flow

### Change of Variables (2D)

For bijective $f: \mathbb{R}^2 \to \mathbb{R}^2$:

$$p_X(x) = p_Z(f^{-1}(x)) \left\| \det \frac{\partial f^{-1}}{\partial x} \right\|$$

- The **Jacobian determinant** replaces the 1D derivative
- Measures how the transformation changes area (volume in higher dims)

---

## 5.4 N-Dimensional Flows

### General Change of Variables Formula

$$\log p_X(x) = \log p_Z(z) + \log \left\| \det \frac{\partial f^{-1}}{\partial x} \right\|$$

where $z = f^{-1}(x)$

### Composition of Flows

Chain multiple simple transformations:

$$x = f_K \circ f_{K-1} \circ \cdots \circ f_1(z)$$

$$\log p_X(x) = \log p_Z(z) + \sum_{k=1}^{K} \log \left\| \det \frac{\partial f_k^{-1}}{\partial f_k} \right\|$$

### Key Challenge
Computing $\det(J)$ for general Jacobian is $O(D^3)$ — need architectures with **tractable Jacobians**.

### Planar Flow

A simple flow that warps a distribution with hyperplanes:

$$\mathbf{x} = \mathbf{z} + \mathbf{u} \cdot h(\mathbf{w}^\top \mathbf{z} + b)$$

where $\mathbf{u}, \mathbf{w} \in \mathbb{R}^D$, $b \in \mathbb{R}$, $h(\cdot)$ is an activation (e.g., tanh).

**Jacobian determinant** (closed form):

$$\det\left(\frac{\partial \mathbf{x}}{\partial \mathbf{z}}\right) = 1 + h'(\mathbf{w}^\top \mathbf{z} + b) \cdot \mathbf{u}^\top \mathbf{w}$$

**Log probability** (change of variables):

$$\log p_x(\mathbf{x}) = \log p_z(\mathbf{z}) - \log \left\| 1 + h'(\mathbf{w}^\top \mathbf{z} + b) \cdot \mathbf{u}^\top \mathbf{w} \right\|$$

For $\mathbf{z} \sim \mathcal{N}(0, I)$: $\log p_z(\mathbf{z}) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\|\mathbf{z}\|^2$

For tanh activation: $h'(a) = 1 - \tanh^2(a)$

---

### 5.4.1 AR and Inverse AR Flows

**Autoregressive Flows (AF)**:
- Each dimension depends on previous dimensions via autoregressive structure
- Jacobian is **triangular** → determinant is product of diagonal:

$$\det(J) = \prod_{d=1}^{D} \frac{\partial f_d}{\partial z_d}$$

- **Fast density evaluation** (parallel), **slow sampling** (sequential)

**Inverse Autoregressive Flows (IAF)**:
- Inverse of AF: fast sampling, slow density evaluation
- Useful when fast sampling is needed (e.g., VAE decoder)

| | Density Evaluation | Sampling |
|--|-------------------|----------|
| **AF** | Fast (parallel) | Slow (sequential) |
| **IAF** | Slow (sequential) | Fast (parallel) |

### 5.4.2 NICE / RealNVP

**NICE (Non-linear Independent Components Estimation)**:
- Uses **additive coupling layers**:
  - Split $z$ into $(z_{1:d}, z_{d+1:D})$
  - $y_{1:d} = z_{1:d}$ (unchanged)
  - $y_{d+1:D} = z_{d+1:D} + m(z_{1:d})$ where $m$ is a neural network
- Jacobian determinant = 1 (volume-preserving)
- Easy to invert: $z_{d+1:D} = y_{d+1:D} - m(y_{1:d})$

**RealNVP (Real-valued Non-Volume Preserving)**:
- Extends NICE with **affine coupling layers**:
  - $y_{1:d} = z_{1:d}$
  - $y_{d+1:D} = z_{d+1:D} \odot \exp(s(z_{1:d})) + t(z_{1:d})$
- Log-determinant: $\sum_j s_j(z_{1:d})$
- Multi-scale architecture with squeeze and split operations

### 5.4.3 Glow, Flow++

**Glow**:
- Extends RealNVP with three key improvements:
  1. **Actnorm**: data-dependent initialization of scale and bias
  2. **Invertible 1×1 convolutions**: replaces fixed permutations → learnable channel mixing
  3. **Affine coupling layers** (same as RealNVP)
- Enables high-resolution image synthesis and interpolation

**Flow++**:
- Improvements over Glow:
  1. **Variational dequantization** (instead of uniform)
  2. **Logistic mixture CDF coupling layers** (more expressive)
  3. **Self-attention in coupling layers**

---

## 5.5 Dequantization

- Problem: discrete data (e.g., integer pixel values) with continuous flow models
- Naive approach: place Dirac deltas on integers → infinite density
- **Uniform dequantization**: add uniform noise $u \sim U[0, 1)$ to each pixel → $x \to x + u$

$$\log p_{\text{model}}(x) \geq \mathbb{E}_{u \sim U[0,1)} [\log p_\theta(x + u)]$$

- **Variational dequantization** (Flow++): learn the dequantization noise distribution with a flow

---

## 5.6 Applications

| Application | Details |
|-------------|---------|
| **Image generation** | Glow generates realistic faces; interpolation in latent space |
| **Super-resolution** | SRFlow: conditional normalizing flow for super-resolution |
| **Text/audio synthesis** | WaveGlow: flow-based model for speech synthesis |
| **Point cloud generation** | PointFlow: continuous normalizing flows for 3D point clouds |
| **Density estimation** | Exact log-likelihoods for anomaly detection |
| **Variational inference** | Flows as flexible approximate posteriors in VAEs |

---

## Key Takeaways

- Normalizing flows provide **exact likelihood** via the change of variables formula
- Key challenge: designing transformations with **tractable Jacobian determinants**
- **Coupling layers** (NICE, RealNVP, Glow) enable efficient and invertible transformations
- Flows support both **fast density evaluation and generation** (architecture-dependent)
- **Dequantization** is necessary for applying continuous flows to discrete data

---

## References

- T1: Prince, Ch. 16 — Normalizing Flows
