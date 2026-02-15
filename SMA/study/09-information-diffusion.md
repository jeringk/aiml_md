# Module 9 — Community and Interactions: Information Diffusion

## Topics

- [[#9.1 Data Mining Essentials|Data Mining Essentials]]
- [[#9.2 Information Diffusion in Social Media|Information Diffusion in Social Media]]

---

## 9.1 Data Mining Essentials

- **Data mining**: extracting patterns, knowledge, and insights from large datasets
- Foundation for analyzing social media data at scale

### Key Data Mining Tasks

| Task | Description | Social Media Application |
|------|-------------|--------------------------|
| **Classification** | Assign labels to instances | Spam detection, bot classification |
| **Clustering** | Group similar items without labels | User segmentation, topic grouping |
| **Association rules** | Find co-occurring patterns | "Users who share X also share Y" |
| **Anomaly detection** | Identify unusual patterns | Fake account detection, unusual activity |
| **Regression** | Predict continuous values | Engagement prediction, virality score |

### Feature Engineering for Social Media

- **Content features**: text (BoW, TF-IDF, embeddings), images, hashtags
- **Network features**: degree, centrality, community membership
- **Behavioral features**: posting frequency, time of activity, engagement rate
- **Temporal features**: recency, periodicity, burstiness

### Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Balanced datasets |
| **Precision** | $\frac{TP}{TP + FP}$ | Minimize false positives |
| **Recall** | $\frac{TP}{TP + FN}$ | Minimize false negatives |
| **F1-Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Balance precision and recall |
| **AUC-ROC** | Area under ROC curve | Ranking quality |

---

## 9.2 Information Diffusion in Social Media

- **Information diffusion**: how information, ideas, and behaviors spread through social networks
- Central question: **What spreads? How? Why? How fast?**

### Types of Diffusion

| Type | Description | Example |
|------|-------------|---------|
| **Information diffusion** | Spreading of news, memes, content | Viral tweets, news sharing |
| **Innovation diffusion** | Adoption of new products/ideas | Technology adoption |
| **Influence diffusion** | Behavioral change due to peers | Opinion change due to friends |

### Diffusion Models

#### Independent Cascade (IC) Model

1. Each newly activated node gets **one chance** to activate each inactive neighbor
2. Activation succeeds with probability $p_{uv}$ (edge-specific)
3. Process continues until no more activations occur

- **Memoryless**: each attempt is independent
- Once activated, a node stays active forever

#### Linear Threshold (LT) Model

1. Each node $v$ has a **threshold** $\theta_v$ (sampled uniformly from $[0, 1]$)
2. Each edge has an influence weight $w_{uv}$ with $\sum_{u \in N(v)} w_{uv} \leq 1$
3. Node $v$ activates when the total influence from active neighbors exceeds its threshold:

$$\sum_{u \in N_{\text{active}}(v)} w_{uv} \geq \theta_v$$

- **Cumulative influence**: all active neighbors contribute

### Properties of Diffusion

- **Cascade size**: number of nodes eventually activated
- **Cascade depth**: longest chain of activations
- **Cascade speed**: how quickly activation spreads
- **Virality**: typically measured by cascade size and structure

### Influence Maximization

- **Problem**: find a seed set $S$ of size $k$ that maximizes the expected number of activated nodes

$$S^* = \arg\max_{|S| = k} \sigma(S)$$

- $\sigma(S)$: expected spread of seed set $S$
- NP-hard problem under both IC and LT models
- **Greedy algorithm** (Kempe et al., 2003): achieves $(1 - 1/e)$-approximation
  - Iteratively add the node with the highest marginal gain
  - Requires Monte Carlo simulation for $\sigma(S)$ estimation

### Epidemiological Models

| Model | Description |
|-------|-------------|
| **SI** | Susceptible → Infected (no recovery) |
| **SIS** | Susceptible → Infected → Susceptible (can be reinfected) |
| **SIR** | Susceptible → Infected → Recovered (permanent immunity) |

- **Basic reproduction number** $R_0$: average number of secondary infections
  - $R_0 > 1$: epidemic spreads
  - $R_0 < 1$: epidemic dies out

---

## Key Takeaways

- Data mining provides the foundational techniques for analyzing social media at scale
- Information diffusion models (IC, LT) capture how content and influence propagate
- Influence maximization is a key optimization problem — greedy approaches provide good approximations
- Epidemiological models (SI, SIS, SIR) offer analogies for understanding viral spreading

---

## References

- T1: Zafarani et al., Ch. 7 — Information Diffusion in Social Media
