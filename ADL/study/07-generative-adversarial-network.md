# Module 7 — Generative Adversarial Network

## Topics

- [[#7.1 Principles|Principles]]
- [[#7.2 Minimax Optimization|Minimax optimization]]
- [[#7.3 DCGAN (Deep Convolutional GAN)|DCGAN]]
- [[#7.4 Variants|Variants]]
  - [[#7.4.1 Wasserstein GAN (WGAN)|Wasserstein GAN]]
  - [[#7.4.2 Conditional GAN (cGAN)|Conditional GAN]]
  - [[#7.4.3 Cycle GAN|Cycle GAN]]
  - [[#7.4.4 Style GAN|Style GAN]]
- [[#7.5 Applications of GAN|Applications of GAN]]

---

## 7.1 Principles

- Two networks competing in a **game**:
  - **Generator** $G$: maps noise $z \sim p_z$ to fake data $G(z)$
  - **Discriminator** $D$: classifies inputs as real or fake
- Generator tries to **fool** the discriminator; discriminator tries to **distinguish** real from generated
- At equilibrium, $G$ produces data indistinguishable from real data, and $D$ outputs 0.5 for all inputs

### GAN Framework

$$z \sim p_z(z) \xrightarrow{G_\theta} G_\theta(z) \xrightarrow{D_\phi} D_\phi(G_\theta(z)) \in [0, 1]$$

---

## 7.2 Minimax Optimization

### Objective Function

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### Optimal Discriminator

For fixed $G$, the optimal discriminator is:

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$$

### Global Optimum

When $p_G = p_{\text{data}}$, $D^* = \frac{1}{2}$ everywhere, and:

$$V(D^*, G^*) = -\log 4$$

### Jensen-Shannon Divergence (JSD)

The GAN objective at optimal discriminator is equivalent to minimizing the JSD:

$$\text{JSD}(P \| Q) = \frac{1}{2}\text{KL}(P \| M) + \frac{1}{2}\text{KL}(Q \| M) \quad \text{where } M = \frac{1}{2}(P + Q)$$

- JSD is **symmetric**: $\text{JSD}(P \| Q) = \text{JSD}(Q \| P)$ (because $M = \frac{1}{2}(P+Q) = \frac{1}{2}(Q+P)$ and the two KL terms simply swap)
- $0 \leq \text{JSD} \leq \log 2$
- Unlike KL divergence, JSD is always **finite** and **well-defined**

### Training Algorithm
1. **Update $D$**: maximize $V$ w.r.t. $\phi$ for $k$ steps
2. **Update $G$**: minimize $V$ w.r.t. $\theta$ for 1 step
3. Repeat

### Non-saturating Loss (practical)
Instead of $\min_G \log(1 - D(G(z)))$ (saturates early), use:

$$\max_G \mathbb{E}_{z}[\log D(G(z))]$$

### Training Challenges
- **Mode collapse**: generator produces limited variety
- **Training instability**: oscillation, discriminator too strong/weak
- **Vanishing gradients**: when discriminator is too good
- No convergence guarantees

---

## 7.3 DCGAN (Deep Convolutional GAN)

Architecture guidelines that stabilized GAN training:

| Guideline | Details |
|-----------|---------|
| Replace pooling with strided convolutions | Discriminator: strided conv; Generator: transposed conv |
| Batch normalization | In both G and D (except G output and D input) |
| Remove fully connected layers | Use global average pooling in D |
| Generator activations | ReLU (hidden), Tanh (output) |
| Discriminator activations | LeakyReLU throughout |

- Established the standard architecture for image GANs
- Showed that learned features are semantically meaningful (vector arithmetic in latent space)

---

## 7.4 Variants

### 7.4.1 Wasserstein GAN (WGAN)

- Replaces JS divergence with **Wasserstein-1 distance** (Earth Mover's Distance):

$$W(p_{\text{data}}, p_G) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_G}[f(x)]$$

#### Wasserstein Distance (Closed Form for Gaussians)

For two 1D Gaussians:

$$W_2(\mathcal{N}(\mu_1, \sigma_1^2), \mathcal{N}(\mu_2, \sigma_2^2)) = \sqrt{(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2}$$

#### Lipschitz Continuity

A function $f$ is **$K$-Lipschitz** if: $|f(x) - f(y)| \leq K|x - y|$ for all $x, y$

- **Lipschitz constant** = $\sup_x |f'(x)|$ (supremum of the absolute derivative)
- WGAN requires critic to be **1-Lipschitz** ($K \leq 1$)
- If $\sup |f'(x)| = \infty$ → function is **not Lipschitz** (e.g., $f(x) = e^{x^2}$, since $f'(x) = 2xe^{x^2} \to \infty$)
- On bounded domain: $f(x) = x^2$ on $[-1,1]$ has $K = \sup|2x| = 2$ → 2-Lipschitz

- Objective (using Kantorovich-Rubinstein duality):

$$\min_G \max_{D \in \mathcal{F}_L} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

where $\mathcal{F}_L$ = set of 1-Lipschitz functions

- **No sigmoid** on discriminator output (now called "critic")
- **Weight clipping** to enforce Lipschitz constraint (WGAN)
  - Issues: capacity underuse, exploding/vanishing gradients
- **Gradient penalty** (WGAN-GP): $\lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$
  - $\hat{x}$ is interpolation between real and fake samples
  - Replaces weight clipping — enforces Lipschitz without limiting capacity

#### Spectral Normalization (SNGAN)

- Normalize weight matrices by their **spectral norm** (largest singular value $\sigma_1(W)$):

$$W' = \frac{W}{\sigma_1(W)}$$

- **Power iteration method** to compute $\sigma_1(W)$:
  1. Initialize $u_0$ (random unit vector)
  2. For each iteration $k$:
     - $v_{k} = \frac{W^\top u_{k-1}}{\|W^\top u_{k-1}\|}$
     - $u_{k} = \frac{W v_{k}}{\|W v_{k}\|}$
  3. $\sigma_1 \approx u_k^\top W v_k = \|W v_k\|$ (after convergence)

- Applied to **discriminator only** — constrains Lipschitz constant for stable training
- **Not applied to generator** — would limit expressiveness and sample quality
- Each weight after normalization: $w' = w / \sigma_1(W)$

### 7.4.2 Conditional GAN (cGAN)

- Conditions both G and D on additional information $y$ (label, text, image):

$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x, y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z, y), y))]$$

- Applications: class-conditional generation, image-to-image translation (Pix2Pix)

### 7.4.3 Cycle GAN

- **Unpaired image-to-image translation**: no paired training data needed
- Two generators: $G_{A \to B}$ and $G_{B \to A}$
- Two discriminators: $D_A$ and $D_B$
- **Cycle consistency loss**:

$$\mathcal{L}_{\text{cyc}} = \mathbb{E}_x[\|G_{B \to A}(G_{A \to B}(x)) - x\|_1] + \mathbb{E}_y[\|G_{A \to B}(G_{B \to A}(y)) - y\|_1]$$

- Total loss = adversarial loss + $\lambda \cdot$ cycle consistency loss
- Applications: style transfer, season transfer, horse↔zebra

### 7.4.4 Style GAN

- **Progressive growing**: train at increasing resolutions (4×4 → 8×8 → ... → 1024×1024)
- **Mapping network**: $z \to w$ via 8-layer MLP (disentangles latent space)
- **Adaptive Instance Normalization (AdaIN)**: injects style $w$ at each layer:

$$\text{AdaIN}(x_i, y) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

- **Noise injection**: adds stochastic variation (hair, freckles)
- **Style mixing**: uses different $w$ at different layers → coarse/fine control
- StyleGAN2: removes artifacts, weight demodulation, path length regularization
- StyleGAN3: alias-free generation with continuous equivariance

---

## 7.5 Applications of GAN

| Application | Examples |
|-------------|---------|
| **Image synthesis** | Faces (StyleGAN), scenes (BigGAN) |
| **Image-to-image translation** | Pix2Pix, CycleGAN |
| **Super-resolution** | SRGAN, ESRGAN |
| **Text-to-image** | StackGAN, AttnGAN |
| **Video generation** | MoCoGAN, DVD-GAN |
| **Data augmentation** | Medical imaging, few-shot learning |
| **Image editing** | Inpainting, attribute manipulation |
| **Domain adaptation** | Transfer across visual domains |
| **3D generation** | NeRF-based GANs, 3D-aware synthesis |

---

## Key Takeaways

- GANs learn via **adversarial training** — generator vs. discriminator
- Original GAN suffers from **mode collapse** and **training instability**
- **WGAN** uses Wasserstein distance for stable training
- **Conditional GANs** enable controlled generation
- **CycleGAN** enables unpaired image translation via cycle consistency
- **StyleGAN** achieves state-of-the-art image synthesis with style-based architecture

---

## References

- T1: Prince, Ch. 15 — GANs
- T2: Goodfellow et al., Ch. 20.10.4 — GANs
- R1: Géron, Ch. 17 — GANs
- R2: Chollet, Ch. 12 — Generative Deep Learning
