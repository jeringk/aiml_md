# Module 8 — Denoising Diffusion Models

## Topics

- [Diffusion Probabilistic Models](#81-diffusion-probabilistic-models)
- [Denoising Diffusion Probabilistic Models (DDPM)](#82-denoising-diffusion-probabilistic-models-ddpm)
- [Denoising Diffusion Implicit Model (DDIM)](#83-denoising-diffusion-implicit-model-ddim)
- [Applications](#84-applications)

---

## 8.1 Diffusion Probabilistic Models

### Core Idea
- Define a **forward process** that gradually adds noise to data until it becomes pure noise
- Learn a **reverse process** that gradually removes noise to generate data
- Both processes are **Markov chains** with learned Gaussian transitions

### Forward Process (Noise Addition)

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I)$$

- $\beta_t$ is the noise schedule ($\beta_1 < \beta_2 < \cdots < \beta_T$)
- After $T$ steps: $x_T \approx \mathcal{N}(0, I)$

### Direct Sampling at Any Step $t$

Using $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I)$$

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

## 8.2 Denoising Diffusion Probabilistic Models (DDPM)

### Reverse Process (Learned Denoising)

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

- Neural network predicts the mean $\mu_\theta(x_t, t)$
- In practice, network predicts the **noise** $\epsilon_\theta(x_t, t)$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

### Training Objective

Simplified loss (predicting the noise):

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

where $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$

### Training Algorithm
1. Sample $x_0$ from data, $t \sim \text{Uniform}\{1, \ldots, T\}$, $\epsilon \sim \mathcal{N}(0, I)$
2. Compute $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$
3. Optimize $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$

### Sampling Algorithm
1. Sample $x_T \sim \mathcal{N}(0, I)$
2. For $t = T, T-1, \ldots, 1$:
   - $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$
   - where $z \sim \mathcal{N}(0, I)$ for $t > 1$, else $z = 0$

### Architecture: U-Net
- U-Net with residual blocks and attention layers
- **Time embedding**: sinusoidal positional encoding of $t$, added to each block
- Skip connections between encoder and decoder paths

### Connection to ELBO

$$\log p_\theta(x_0) \geq \mathbb{E}_q \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} | x_0)} \right]$$

The VLB decomposes into:
- $L_T$: prior matching term
- $L_{t-1}$: denoising matching terms (for $1 < t \leq T$)
- $L_0$: reconstruction term

---

## 8.3 Denoising Diffusion Implicit Model (DDIM)

- **Non-Markovian** forward process that allows **deterministic** sampling
- Same training objective as DDPM but different sampling procedure
- **Key advantage**: can skip steps — sample with $S \ll T$ steps

### DDIM Sampling Update

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } x_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon_t$$

- When $\sigma_t = 0$: **deterministic** sampling (same latent → same image)
- When $\sigma_t = \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)}} \sqrt{\beta_t}$: recovers DDPM

### Advantages over DDPM
| Property | DDPM | DDIM |
|----------|------|------|
| **Steps needed** | $T$ (e.g., 1000) | $S \ll T$ (e.g., 50–100) |
| **Deterministic** | No (stochastic) | Yes (when $\sigma = 0$) |
| **Interpolation** | Not meaningful | Meaningful latent interpolation |
| **Speed** | Slow | 10–50× faster |

---

## 8.4 Applications

| Application | Details |
|-------------|---------|
| **Image generation** | DALL·E 2, Stable Diffusion, Imagen |
| **Image editing** | SDEdit, InstructPix2Pix |
| **Inpainting** | Fill in missing regions |
| **Super-resolution** | Conditional diffusion for upscaling |
| **Text-to-image** | Classifier-free guidance with CLIP/T5 |
| **Video generation** | Video diffusion models |
| **3D generation** | DreamFusion, Point-E |
| **Audio synthesis** | DiffWave, WaveGrad |
| **Molecular generation** | Drug discovery and protein design |

### Guidance Techniques
- **Classifier guidance**: use gradients from a pre-trained classifier to steer generation
- **Classifier-free guidance**: train with and without conditioning; interpolate at inference:

$$\hat{\epsilon}_\theta(x_t, c) = (1 + w) \epsilon_\theta(x_t, c) - w \epsilon_\theta(x_t, \varnothing)$$

where $w > 0$ is guidance scale

---

## Key Takeaways

- Diffusion models gradually add then remove noise — **simple training, high-quality samples**
- DDPM uses $\epsilon$-prediction with MSE loss — surprisingly simple objective
- DDIM enables **fast deterministic sampling** by skipping diffusion steps
- Currently **state-of-the-art** for image generation quality
- **Classifier-free guidance** is the dominant technique for conditional generation

---

## References

- T1: Prince, Ch. 18 — Diffusion Models
