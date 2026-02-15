# Module 11 — Time Series Modeling and Generation

## Topics

- [[#11.1 Advanced VAE Techniques for Time Series|Advanced VAE techniques]] | [[#11.2 Advanced GAN Techniques for Time Series|Advanced GAN techniques]]
- [[#11.3 Classical Time Series Models|Generation of time-series data (ARIMA, S-ARIMA etc.)]]

---

## 11.1 Advanced VAE Techniques for Time Series

### Temporal VAE
- Extend standard VAE with **temporal structure** in the latent space
- Latent transitions: $z_t = f(z_{t-1}) + \epsilon$
- Encoder and decoder use RNN/LSTM/Transformer architectures

### Key Approaches
| Model | Description |
|-------|-------------|
| **VRNN** | VAE + RNN — latent variables at each time step conditioned on RNN hidden state |
| **SRNN** | Structured inference networks for temporal data |
| **GP-VAE** | Gaussian Process prior on latent trajectories for smooth temporal dynamics |
| **TimeVAE** | Generates synthetic time series using interpretable temporal components |

### VRNN (Variational RNN)
- Prior, encoder, decoder all conditioned on RNN hidden state $h_t$:
  - Prior: $p_\theta(z_t | h_{t-1})$
  - Encoder: $q_\phi(z_t | x_t, h_{t-1})$
  - Decoder: $p_\theta(x_t | z_t, h_{t-1})$

---

## 11.2 Advanced GAN Techniques for Time Series

### TimeGAN
- Combines **autoencoder**, **adversarial training**, and **supervised loss**
- Four components: embedding, recovery, generator, discriminator
- Supervised loss preserves **temporal dynamics**
- Learns in a **learned embedding space** (not raw feature space)

### RCGAN (Recurrent Conditional GAN)
- Generator and discriminator are both **RNNs**
- Conditional generation based on labels or partial sequences
- Well-suited for medical time series

### Other Approaches
| Model | Key Idea |
|-------|----------|
| **C-RNN-GAN** | Music generation with GAN + LSTM |
| **COT-GAN** | Causal optimal transport for temporal data |
| **SigWGAN** | Wasserstein GAN with signature features for time series |

---

## 11.3 Classical Time Series Models

### ARIMA (AutoRegressive Integrated Moving Average)

$$ARIMA(p, d, q): \quad \phi(B)(1-B)^d X_t = \theta(B) \epsilon_t$$

- $p$: AR order, $d$: differencing order, $q$: MA order
- $B$: backshift operator ($B X_t = X_{t-1}$)
- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ (AR polynomial)
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ (MA polynomial)

### S-ARIMA (Seasonal ARIMA)

$$SARIMA(p,d,q)(P,D,Q)_s$$

- Adds seasonal components with period $s$
- Seasonal AR: $\Phi(B^s)$, Seasonal MA: $\Theta(B^s)$, Seasonal differencing: $(1-B^s)^D$

### Deep Learning vs. Classical
| Aspect | ARIMA / S-ARIMA | Deep Generative |
|--------|----------------|-----------------|
| Interpretability | High | Low |
| Multivariate | Limited | Natural |
| Nonlinearity | No | Yes |
| Data requirements | Small | Large |
| Generation quality | Limited | High |

---

## Key Takeaways

- **TimeGAN** combines autoencoding + adversarial + supervised losses for temporal fidelity
- **VRNNs** extend VAEs with recurrent temporal structure
- Classical models (ARIMA, S-ARIMA) provide interpretable baselines
- Deep generative models excel at multivariate, nonlinear time series generation

---

## References

- Yoon et al., "TimeGAN" (NeurIPS 2019)
- Chung et al., "VRNN" (NeurIPS 2015)
