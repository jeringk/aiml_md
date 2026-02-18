# Module 9 — Energy and Score-Based Models

## Topics

- [[#9.1 Parametrizing Probability Distributions|Parametrizing probability distributions]]
- [[#9.2 Energy-Based Generative Modeling|Energy-based generative modeling]]
- [[#9.3 Classical Energy-Based Models|Ising Model, Product of Experts, Restricted Boltzmann machine]]
- [[#9.4 Deep Boltzmann Machines (DBM)|Deep Boltzmann Machines]]
- [[#9.5 Training and Sampling from EBMs|Training and sampling from EBMs]]
- [[#9.6 Score-Based Models|Score-based Models]]

---

## 9.1 Parametrizing Probability Distributions

### Unnormalized Models

$$p_\theta(x) = \frac{1}{Z(\theta)} \tilde{p}_\theta(x)$$

where $Z(\theta) = \int \tilde{p}_\theta(x) dx$ is the **partition function** (normalization constant)

- Problem: $Z(\theta)$ is typically **intractable** to compute
- Cannot directly evaluate $p_\theta(x)$ or compute MLE gradients easily

### Energy-Based Parameterization

$$p_\theta(x) = \frac{1}{Z(\theta)} \exp(-E_\theta(x))$$

- $E_\theta(x)$: **energy function** — lower energy = higher probability
- Any positive function can be an unnormalized density
- Exponential ensures positivity: $\tilde{p}_\theta(x) = \exp(-E_\theta(x)) > 0$

### Computing Partition Functions (Gaussian Integrals)

For quadratic energy functions, the partition function reduces to a Gaussian integral:

$$\int_{-\infty}^{\infty} e^{-ax^2 + bx}\, dx = \sqrt{\frac{\pi}{a}}\, e^{b^2/(4a)} \quad (a > 0)$$

**Technique — Completing the Square:**

$$-ax^2 + bx = -a\left(x - \frac{b}{2a}\right)^2 + \frac{b^2}{4a}$$

Then factor out the constant and use the standard Gaussian integral $\int e^{-u^2}\, du = \sqrt{\pi}$:

$$Z = e^{b^2/(4a)} \int_{-\infty}^{\infty} e^{-a(x - b/2a)^2}\, dx = e^{b^2/(4a)} \cdot \sqrt{\frac{\pi}{a}}$$

For unnormalized Gaussian form $e^{-(x-\mu)^2/(2\sigma^2)}$: $Z = \sigma\sqrt{2\pi}$

---

## 9.2 Energy-Based Generative Modeling

### Why EBMs?

- **Flexibility**: $E_\theta(x)$ can be any neural network — no architectural constraints
  - No invertibility (unlike flows)
  - No encoder-decoder structure (unlike VAEs)
  - No adversarial training (unlike GANs)
- Model only needs to assign relative energies — no normalization needed for comparisons
- Connects to physics (statistical mechanics) and probabilistic graphical models

### Challenges
1. **Partition function** $Z(\theta)$ is intractable
2. **Sampling** from $p_\theta(x)$ is difficult
3. **Training** requires approximating gradients of $\log Z(\theta)$

---

## 9.3 Classical Energy-Based Models

### 9.3.1 Ising Model
- Binary variables on a lattice: $x_i \in \{-1, +1\}$
- Energy: $E(x) = -\sum_{(i,j)} J_{ij} x_i x_j - \sum_i h_i x_i$
  - $J_{ij}$: coupling between neighbors
  - $h_i$: external field
- Phase transitions between ordered and disordered states

### 9.3.2 Product of Experts (PoE)
- Combine multiple expert distributions:

$$p(x) = \frac{1}{Z} \prod_{m=1}^{M} f_m(x)$$

- Each expert $f_m$ is "soft" constraint — product enforces all constraints simultaneously
- Energy: $E(x) = -\sum_m \log f_m(x)$

### 9.3.3 Restricted Boltzmann Machine (RBM)
- Bipartite graph: **visible** units $v$ and **hidden** units $h$
- No connections within the same layer
- Energy:

$$E(v, h) = -b^T v - c^T h - v^T W h$$

- Marginal distribution: $p(v) = \frac{1}{Z} \sum_h \exp(-E(v, h))$
- **Key property**: conditional independence makes Gibbs sampling efficient:

$$p(h_j = 1 \| v) = \sigma(c_j + W_j^T v)$$
$$p(v_i = 1 \| h) = \sigma(b_i + W_i h)$$

- Training: **Contrastive Divergence (CD-k)**

---

## 9.4 Deep Boltzmann Machines (DBM)

- Multiple hidden layers with **symmetric** connections (undirected)
- All layers interact bidirectionally
- Energy:

$$E(v, h^{(1)}, h^{(2)}) = -v^T W^{(1)} h^{(1)} - h^{(1)^T} W^{(2)} h^{(2)} - b^T v - c^{(1)^T} h^{(1)} - c^{(2)^T} h^{(2)}$$

- Richer representations than RBMs but harder to train
- Training requires **variational inference** for the positive phase
- Layer-wise pretraining (as stack of RBMs) helps initialization

### DBM vs. Deep Belief Networks (DBN)
| Property | DBM | DBN |
|----------|-----|-----|
| Connections | All undirected | Top 2 layers undirected, rest directed |
| Inference | Variational (approximate) | Exact for top-down pass |
| Training | Harder | Greedy layer-wise |

---

## 9.5 Training and Sampling from EBMs

### Training: Maximum Likelihood

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \mathbb{E}_{x^- \sim p_\theta}[-\nabla_\theta E_\theta(x^-)]$$

$$= \underbrace{-\nabla_\theta E_\theta(x)}_{\text{positive phase}} + \underbrace{\mathbb{E}_{x^- \sim p_\theta}[\nabla_\theta E_\theta(x^-)]}_{\text{negative phase}}$$

- **Positive phase**: push down energy of training data
- **Negative phase**: push up energy of model samples — requires **sampling from model**

### Sampling: MCMC Methods

**Langevin Dynamics**:
$$x_{k+1} = x_k - \frac{\eta}{2} \nabla_x E_\theta(x_k) + \sqrt{\eta} \, \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I)$$

- Gradient descent on energy + noise for exploration
- Converges to $p_\theta(x)$ as $\eta \to 0$ and $k \to \infty$

**Contrastive Divergence (CD-k)**:
1. Initialize chain at data point $x^0 = x_{\text{data}}$
2. Run $k$ steps of Gibbs sampling
3. Use the resulting sample as negative phase sample
- $k = 1$ works surprisingly well in practice (CD-1)

**Persistent Contrastive Divergence (PCD)**:
- Maintain persistent Markov chains across parameter updates
- Better mixing than CD-k

### Alternative Training Methods
| Method | Idea |
|--------|------|
| **Score matching** | Match $\nabla_x \log p_\theta$ instead of $p_\theta$ — avoids $Z$ |
| **Noise Contrastive Estimation (NCE)** | Discriminate data from noise distribution |
| **Contrastive learning** | Learn relative energies without normalization |

---

## 9.6 Score-Based Models

### Score Function

$$s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

- The score is the gradient of log-density w.r.t. input
- **Does not depend on $Z(\theta)$** — partition function cancels in the gradient

### Score Matching

Train $s_\theta(x)$ to match the data score $\nabla_x \log p_{\text{data}}(x)$:

$$\mathcal{L} = \mathbb{E}_{p_{\text{data}}} \left[ \frac{1}{2} \|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2 \right]$$

Equivalent objective (does not require knowing $\nabla_x \log p_{\text{data}}$):

$$\mathcal{L} = \mathbb{E}_{p_{\text{data}}} \left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \|s_\theta(x)\|^2 \right]$$

### Denoising Score Matching (DSM)

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x \sim p_{\text{data}}, \tilde{x} \sim q_\sigma(\tilde{x}\|x)} \left[ \|s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}\|x)\|^2 \right]$$

For Gaussian noise $q_\sigma(\tilde{x}\|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$:

$$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}\|x) = \frac{x - \tilde{x}}{\sigma^2}$$

- Connection to DDPM: the noise prediction network $\epsilon_\theta$ is essentially a **score model**!

### Sampling via Langevin Dynamics

Given the learned score $s_\theta(x)$:

$$x_{k+1} = x_k + \frac{\eta}{2} s_\theta(x_k) + \sqrt{\eta} \, \epsilon_k$$

### Noise Conditional Score Networks (NCSN)
- Train score models at **multiple noise levels** $\sigma_1 > \sigma_2 > \cdots > \sigma_L$
- **Annealed Langevin dynamics**: run Langevin at each noise level progressively
- Solves the problem of inaccurate scores in low-density regions

### Score SDE (Continuous Formulation)
- Unifies DDPM and score-based models under **Stochastic Differential Equations**:

$$dx = f(x, t)dt + g(t)dw$$

- Reverse SDE:

$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

---

## Key Takeaways

- EBMs define distributions via energy functions — **flexible but hard to train**
- Training requires estimating the partition function gradient via **MCMC sampling**
- **Score-based models** avoid the partition function by learning $\nabla_x \log p(x)$
- Denoising score matching connects score-based models to **diffusion models**
- Score SDE provides a **unified framework** for diffusion and score-based generation

---

## References

- T2: Goodfellow et al., Ch. 16–20 — Monte Carlo Methods, Boltzmann Machines
- T1: Prince, Ch. 18 — Diffusion Models (score-based connection)
