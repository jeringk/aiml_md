# Module 6 — Variational Inferencing

## Topics

- [[#6.1 Latent Variable Models|Latent Variable Models]]
- [[#6.2 Training Latent Variable Models|Training Latent Variable Models]]
  - [[#6.2.1 Exact Likelihood|6.2.1 Exact Likelihood]]
  - [[#6.2.2 Sampling; Prior and Importance Sampling|6.2.2 Sampling; Prior, Importance]]
  - [[#6.2.3 Importance Weighted Autoencoder (IWAE)|6.2.3 Importance Weighted AE]]
- [[#6.3 Variational / Evidence Lower Bound (ELBO)|Variational / Evidence Lower Bound]]
- [[#6.4 Optimizing VLB / ELBO|Optimizing VLB / ELBO]]
- [[#6.5 VAE Variants|VAE Variants: VQ-VAE, AR_VAE, Beta VAE]]
- [[#6.6 Variational Dequantization|Variational Dequantization]]

---

## 6.1 Latent Variable Models

- Introduce **latent variables** $z$ to model complex data distributions
- Generative process:
  1. Sample $z \sim p(z)$ (prior — typically $\mathcal{N}(0, I)$)
  2. Sample $x \sim p_\theta(x|z)$ (decoder / likelihood)
- Marginal likelihood:

$$p_\theta(x) = \int p_\theta(x|z) p(z) \, dz$$

- where:
  - $x$ = observed data sample
  - $z$ = latent variable
  - $p(z)$ = prior over latents
  - $p_\theta(x|z)$ = decoder likelihood parameterized by $\theta$
  - $\theta$ = decoder/generative model parameters

- This integral is typically **intractable** — cannot be computed in closed form
- Latent variables capture **hidden factors of variation** in the data

---

## 6.2 Training Latent Variable Models

### 6.2.1 Exact Likelihood

- For simple models (e.g., mixture of Gaussians), the marginal can be computed:

$$p_\theta(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

- where:
  - $K$ = number of mixture components
  - $\pi_k$ = mixing coefficient for component $k$ (with $\sum_k \pi_k = 1$)
  - $\mu_k$ = mean of component $k$
  - $\Sigma_k$ = covariance of component $k$
  - $\mathcal{N}(x|\mu_k,\Sigma_k)$ = Gaussian density evaluated at $x$

- For deep generative models with continuous $z$, the integral is intractable
- Cannot directly maximize $\log p_\theta(x)$

### 6.2.2 Sampling; Prior and Importance Sampling

**Naive Monte Carlo** (sampling from prior):

$$p_\theta(x) = \mathbb{E}_{z \sim p(z)}[p_\theta(x|z)] \approx \frac{1}{K} \sum_{k=1}^{K} p_\theta(x|z_k), \quad z_k \sim p(z)$$

- where:
  - $\mathbb{E}_{z \sim p(z)}[\cdot]$ = expectation under the prior
  - $K$ = number of Monte Carlo samples
  - $z_k$ = $k$-th sampled latent variable
  - $p_\theta(x|z_k)$ = likelihood contribution for sample $z_k$

- Problem: most samples from $p(z)$ contribute little to $p_\theta(x|z)$ — **high variance**

**Importance Sampling** (using proposal distribution $q(z)$):

$$p_\theta(x) = \mathbb{E}_{z \sim q(z)} \left[ \frac{p_\theta(x|z) p(z)}{q(z)} \right] \approx \frac{1}{K} \sum_{k=1}^{K} \frac{p_\theta(x|z_k) p(z_k)}{q(z_k)}$$

- where:
  - $q(z)$ = proposal distribution
  - $\frac{p_\theta(x|z)p(z)}{q(z)}$ = importance weight contribution
  - $q(z_k)$ = proposal density at sample $z_k$
  - $p(z_k)$ = prior density at sample $z_k$

- Better when $q(z)$ is close to the true posterior $p_\theta(z|x)$

### 6.2.3 Importance Weighted Autoencoder (IWAE)

- Uses multiple importance samples to tighten the bound:

$$\log p_\theta(x) \geq \mathbb{E}_{z_1, \ldots, z_K \sim q_\phi(z|x)} \left[ \log \frac{1}{K} \sum_{k=1}^{K} \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)} \right]$$

- where:
  - $\phi$ = inference/encoder network parameters
  - $q_\phi(z|x)$ = variational posterior (proposal conditioned on $x$)
  - $p_\theta(x,z_k)$ = joint density of data and latent sample
  - $K$ = number of importance samples

- As $K \to \infty$, the bound becomes tight
- Better log-likelihood estimates than standard VAE
- Trade-off: more compute per gradient step

---

## 6.3 Variational / Evidence Lower Bound (ELBO)

### Derivation

$$\log p_\theta(x) = \log \int p_\theta(x, z) \, dz$$

- where:
  - $p_\theta(x,z)$ = joint generative distribution, obtained as $p_\theta(x|z)p(z)$

Introduce approximate posterior $q_\phi(z|x)$:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]}_{\text{ELBO } \mathcal{L}(\theta, \phi; x)} + \underbrace{KL(q_\phi(z|x) \| p_\theta(z|x))}_{\geq 0}$$

- where:
  - $\mathcal{L}(\theta,\phi;x)$ = ELBO objective for one sample $x$
  - $KL(\cdot\|\cdot)$ = Kullback-Leibler divergence
  - $p_\theta(z|x)$ = true posterior under the model

Therefore:

$$\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{KL(q_\phi(z|x) \| p(z))}_{\text{Regularization}}$$

- where:
  - reconstruction term measures expected data fit
  - KL term regularizes latent codes toward prior $p(z)$

### Interpretation
- **Reconstruction term**: how well the decoder reconstructs $x$ from sampled $z$
- **KL term**: keeps the approximate posterior close to the prior
- ELBO = Evidence Lower Bound = Variational Lower Bound (VLB)
- Gap = $KL(q_\phi(z|x) \| p_\theta(z|x))$ — tighter when $q_\phi$ is closer to true posterior

---

## 6.4 Optimizing VLB / ELBO

### Encoder Outputs in a VAE

- The encoder predicts parameters of a Gaussian posterior:

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)))$$

- where:
  - $\mu_\phi(x)$ = **mean layer** output (center/location of latent distribution)
  - $\log \sigma_\phi^2(x)$ = **log-variance layer** output (spread/uncertainty of each latent dimension)
- In implementation, we predict $\log \sigma^2$ instead of $\sigma^2$ directly:
  - ensures positive variance via $\sigma^2 = \exp(\log \sigma^2)$
  - improves numerical stability during optimization

### The Reparameterization Trick

- Cannot backprop through sampling $z \sim q_\phi(z|x)$
- Reparameterize: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$
- Practical form with log-variance:

$$z = \mu_\phi(x) + \exp\left(\frac{1}{2}\log \sigma_\phi^2(x)\right)\odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

- Intuition: sampling noise is isolated in $\epsilon$, so $z$ becomes differentiable w.r.t. encoder outputs
- Now gradients flow through $\mu_\phi$ and $\sigma_\phi$

### VAE Training

$$\max_{\theta, \phi} \mathcal{L}(\theta, \phi) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} [\log p_\theta(x | \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)] - KL(q_\phi(z|x) \| p(z)) \right]$$

- where:
  - $p_{\text{data}}$ = empirical data distribution
  - $\mu_\phi(x)$ and $\sigma_\phi(x)$ = encoder outputs (mean and std)
  - $\epsilon$ = auxiliary noise
  - $\odot$ = element-wise product

### KL Divergence (Gaussian case, closed form)

$$KL(q_\phi(z|x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

- where:
  - $J$ = latent dimensionality
  - $\mu_j$ and $\sigma_j$ = mean and std of latent dimension $j$ in $q_\phi(z|x)$
  - $p(z)=\mathcal{N}(0,I)$

### KL Divergence (General Gaussians)

For two arbitrary univariate Gaussians:

$$\text{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

- where:
  - $(\mu_1,\sigma_1^2)$ are mean/variance of the first Gaussian
  - $(\mu_2,\sigma_2^2)$ are mean/variance of the second Gaussian

- KL divergence is **asymmetric**: $\text{KL}(P \| Q) \neq \text{KL}(Q \| P)$ in general
- The VAE formula above is a special case where $\mu_2 = 0$, $\sigma_2 = 1$

### Posterior Collapse
- Problem: decoder ignores $z$ and models $p(x)$ directly
- KL term goes to zero — latent space is unused
- Solutions: KL annealing, free bits, aggressive training schedule

---

## 6.5 VAE Variants

### VQ-VAE (Vector Quantized VAE)
- Replaces continuous latent space with **discrete** codebook vectors
- Encoder output is mapped to **nearest codebook entry**: $z_q = \text{argmin}_{e_k} \|z_e - e_k\|$
- Loss = reconstruction + codebook loss + commitment loss:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \|sg[z_e] - e\|^2 + \beta \|z_e - sg[e]\|^2$$

- where:
  - $x$ = input
  - $\hat{x}$ = reconstruction
  - $z_e$ = encoder output before quantization
  - $e$ = selected codebook vector
  - $\beta$ = commitment-loss weight
  - $sg$ = stop-gradient
- No KL divergence term — uses a uniform prior over codebook
- Often paired with **PixelCNN** as a prior for generation
- Excellent for audio (speech), images, and video

### AR_VAE (Autoregressive VAE)
- Uses an autoregressive decoder $p_\theta(x|z) = \prod_d p_\theta(x_d | x_{<d}, z)$
- Combines VAE's latent structure with AR's expressive power
- Challenge: risk of posterior collapse (decoder can ignore $z$)

### Beta-VAE ($\beta$-VAE)
- Modifies ELBO with a weight $\beta > 1$ on the KL term:

$$\mathcal{L}_\beta = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot KL(q_\phi(z|x) \| p(z))$$

- where:
  - $\beta$ = scalar that controls KL regularization strength (typically $\beta > 1$ for stronger disentanglement)

- Higher $\beta$ → more pressure for **disentangled representations**
- Each latent dimension encodes a single factor of variation
- Trade-off: better disentanglement vs. worse reconstruction quality

---

## 6.6 Variational Dequantization

- Problem: applying continuous VAEs to discrete data (e.g., pixel values 0–255)
- Solution: learn a variational distribution over the dequantization noise
- Instead of uniform noise: $q(u|x)$ is a flexible distribution (e.g., a flow)
- Provides a tighter lower bound on the log-likelihood
- Used in Flow++ and other advanced models

---

## Key Takeaways

- VAEs introduce a latent variable model trained via the **ELBO**
- The **reparameterization trick** enables end-to-end gradient-based training
- ELBO = reconstruction quality − KL regularization
- **VQ-VAE** uses discrete latents; **β-VAE** encourages disentanglement
- Posterior collapse is a key challenge — addressed by annealing and architectural choices

---

## References

- T1: Prince, Ch. 17 — VAEs
- T2: Goodfellow et al., Ch. 20.10 — Variational Autoencoders
- R2: Chollet, Ch. 12 — Generative Deep Learning
