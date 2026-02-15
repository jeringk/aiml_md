# Advanced Deep Learning (ADL)

**Course Code:** AIMLCZG513  
**Semester:** 3  
**Institution:** BITS Pilani (WILP)

---

## Contents

- [`pre-req.md`](pre-req.md) — Pre-requisite topics assumed before this course
- [`study/`](study/) — Topic-wise study notes and summaries
- [`questions/`](questions/) — Past papers and practice questions

---

## Textbooks

| Code | Book |
|------|------|
| **T1** | *Understanding Deep Learning*, The MIT Press, 2023 — Simon J.D. Prince |
| **T2** | *Deep Learning* (Adaptive Computation and Machine Learning series), Hardcover – 18 Nov 2016 — Aaron Courville, Ian Goodfellow, Yoshua Bengio |

## Reference Books & Other Resources

| Code | Resource |
|------|----------|
| **R1** | *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems* — Aurélien Géron |
| **R2** | *Deep Learning with Python* — François Chollet |
| — | Research Papers, Blogs |

---

## Modular Content Structure

### [1. Introduction](study/01-introduction.md)
- Unsupervised, semi-supervised, self-supervised learning
- Representation learning
- Generative Modeling

### [2. PCA Variants](study/02-pca-variants.md)
- Randomized PCA
- Incremental PCA
- Kernel PCA
- Probabilistic PCA
- Sparse PCA
- Canonical Correlation Analysis (CCA)
- Locally Linear Embedding (LLE)
- Independent Component Analysis
- Factor Analysis
- Manifold learning

### [3. Autoencoders](study/03-autoencoders.md)
- Type of activation and loss functions
- Undercomplete vs. Overcomplete autoencoders
- Relationship with PCA
- Regularization
  - Denoising autoencoder
  - Sparse autoencoder
  - Contractive Autoencoders
- Effect of Depth
- Application of Autoencoders

### [4. Autoregressive Models](study/04-autoregressive-models.md)
- Motivation
- Simple generative models: histograms
- Parameterized distributions and maximum likelihood
- Recurrent Neural Nets
- Masking-based Models
  - Masked AEs for Distribution Estimation (MADE)
  - Masked Convolutions
    - WaveNet
    - PixelCNN and Variants
- Applications in super-resolution, colorization
- Speed Up Strategies

### [5. Normalizing Flow Models](study/05-normalizing-flow-models.md)
- Difference with AR Models
- Foundations of 1-D Flow
- 2-D Flow
- N-dimensional flows
  - AR and inverse AR flows
  - NICE / RealNVP
  - Glow, Flow++
- Dequantization
- Applications in super-resolution, text/audio synthesis, point cloud generation

### [6. Variational Inferencing](study/06-variational-inferencing.md)
- Latent Variable Models
- Training Latent Variable Models
  - 6.2.1 Exact Likelihood
  - 6.2.2 Sampling; Prior, Importance
  - 6.2.3 Importance Weighted AE
- Variational / Evidence Lower Bound
- Optimizing VLB / ELBO
- VAE Variants: VQ-VAE, AR_VAE, Beta VAE
- Variational Dequantization

### [7. Generative Adversarial Network](study/07-generative-adversarial-network.md)
- Principles
- Minimax optimization
- DCGAN
- Variants
  - Wasserstein GAN
  - Conditional GAN
  - Cycle GAN
  - Style GAN
- Applications of GAN

### [8. Denoising Diffusion Models](study/08-denoising-diffusion-models.md)
- Diffusion Probabilistic Models
- Denoising Diffusion Probabilistic Models
- Denoising Diffusion Implicit Model
- Applications

### [9. Energy and Score-Based Models](study/09-energy-score-based-models.md)
- Parametrizing probability distributions
- Energy-based generative modeling
- Ising Model, Product of Experts, Restricted Boltzmann machine
- Deep Boltzmann Machines
- Training and sampling from EBMs
- Score-based Models

### [10. Language Modeling](study/10-language-modeling.md)
- Motivation and Introduction
- Introduction to Language Models
- A digression into Transformer, word2Vec, BERT, …, GPT

### [11. Time Series Modeling and Generation](study/11-time-series-modeling.md)
- Advanced VAE and GAN techniques for modelling of time-series data
- Generation of time-series data (ARIMA, S-ARIMA etc.)
