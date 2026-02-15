# Module 1 — Introduction

## Topics

- [[#1.1 Unsupervised, Semi-supervised, Self-supervised Learning|Unsupervised, semi-supervised, self-supervised learning]]
- [[#1.2 Representation Learning|Representation learning]]
- [[#1.3 Generative Modeling|Generative Modeling]]

---

## 1.1 Unsupervised, Semi-supervised, Self-supervised Learning

### Unsupervised Learning
- Learning patterns from **unlabeled data** — no target labels provided
- Goals: discover hidden structure, cluster data, reduce dimensionality
- Examples: clustering (K-Means), dimensionality reduction (PCA), density estimation

### Semi-supervised Learning
- Uses a **small amount of labeled data + large amount of unlabeled data**
- Exploits the structure in unlabeled data to improve supervised learning
- Assumptions: smoothness, cluster, manifold
- Useful when labeling is expensive (medical imaging, NLP)

### Self-supervised Learning
- A form of unsupervised learning where the model generates its own labels from the data
- **Pretext tasks**: predict part of the input from other parts
  - E.g., predicting next word (GPT), masked word (BERT), rotation angle, colorization
- Learns rich representations that transfer well to downstream tasks
- Bridge between unsupervised and supervised learning

---

## 1.2 Representation Learning

- Learning **useful features/representations** automatically from raw data
- Replaces hand-crafted feature engineering
- Goal: learn transformations that make subsequent learning tasks easier
- Key idea: learn $z = f(x)$ where $z$ captures meaningful factors of variation

### Why Representation Learning?
- Raw data (pixels, waveforms) is high-dimensional and redundant
- Good representations are **compact**, **disentangled**, and **informative**
- Enables transfer learning and multi-task learning

### Types of Representations
- **Distributed representations**: each feature captures a factor of variation
- **Disentangled representations**: individual dimensions correspond to independent generative factors
- **Hierarchical representations**: multiple levels of abstraction (as in deep networks)

---

## 1.3 Generative Modeling

- **Discriminative models**: learn $P(y|x)$ — decision boundary
- **Generative models**: learn $P(x)$ or $P(x, y)$ — the data distribution

### Why Generative Models?
- Generate new realistic samples (images, text, audio)
- Understand the underlying data distribution
- Anomaly detection, data augmentation, missing data imputation
- Density estimation

### Taxonomy of Generative Models

$$\text{Generative Models} \begin{cases} \text{Explicit density} \begin{cases} \text{Tractable}: \text{AR models, Flow models} \\ \text{Approximate}: \text{VAE, Boltzmann machines} \end{cases} \\ \text{Implicit density}: \text{GANs} \end{cases}$$

### Key Generative Model Families (covered in later modules)
1. **Autoencoders** (Module 3)
2. **Autoregressive Models** (Module 4)
3. **Normalizing Flows** (Module 5)
4. **VAEs** (Module 6)
5. **GANs** (Module 7)
6. **Diffusion Models** (Module 8)
7. **Energy / Score-Based Models** (Module 9)

---

## Key Takeaways

- Deep learning enables powerful **representation learning** from raw data
- **Generative models** learn the data distribution and can generate new samples
- The ADL course covers the major families of deep generative models
- Self-supervised learning is a rapidly growing paradigm that bridges unsupervised and supervised approaches

---

## References

- T1: Prince, Ch. 1 — Introduction
- T2: Goodfellow et al., Ch. 1 — Introduction, Ch. 15 — Representation Learning
