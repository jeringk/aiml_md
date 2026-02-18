# Module 6 — Machine Learning & Traditional Analytical Techniques: Network Models

## Topics

- [[#6.1 Why Network Models?|Why Network Models?]]
- [[#6.2 Random Graphs (Erdős–Rényi)|Random Graphs (Erdős–Rényi)]]
- [[#6.3 Small-World Model (Watts–Strogatz)|Small-World Model (Watts–Strogatz)]]
- [[#6.4 Preferential Attachment (Barabási–Albert)|Preferential Attachment (Barabási–Albert)]]
- [[#6.5 Comparing Models|Comparing Models]]

---

## 6.1 Why Network Models?

- **Network models** generate synthetic networks with known properties
- Purposes:
  - **Understand** mechanisms that produce observed network properties
  - **Null models**: compare real networks against random baselines
  - **Simulation**: test algorithms on controlled network structures
  - **Prediction**: model network growth and evolution

---

## 6.2 Random Graphs (Erdős–Rényi)

### Two Variants

| Model | Description |
|-------|-------------|
| **$G(n, m)$** \| $n$ nodes, $m$ edges chosen uniformly at random |
| **$G(n, p)$** \| $n$ nodes, each pair connected independently with probability $p$ |

### Properties of $G(n, p)$

- **Expected edges**: $\mathbb{E}[\|E\|] = \binom{n}{2} p$
- **Expected degree**: $\langle k \rangle = (n - 1)p$
- **Degree distribution**: Binomial → Poisson for large $n$:

$$P(k) \approx e^{-\langle k \rangle} \frac{\langle k \rangle^k}{k!}$$

- **Clustering coefficient**: $C = p = \frac{\langle k \rangle}{n - 1}$ → vanishes for sparse graphs
- **Average path length**: $\langle l \rangle \approx \frac{\ln n}{\ln \langle k \rangle}$ (short — logarithmic)

### Phase Transitions

- **Giant component** emerges when $\langle k \rangle > 1$ (i.e., $p > \frac{1}{n}$)
- Connectivity: graph is almost surely connected when $p > \frac{\ln n}{n}$

### Limitations for Social Networks

- **No power-law degree**: real networks have hubs, ER does not
- **Low clustering**: ER clustering vanishes, real networks have high clustering
- No community structure

---

## 6.3 Small-World Model (Watts–Strogatz)

### Construction

1. Start with a **ring lattice**: $n$ nodes, each connected to $k$ nearest neighbors
2. **Rewire** each edge with probability $p$:
   - $p = 0$: regular lattice (high clustering, long paths)
   - $p = 1$: random graph (low clustering, short paths)
   - $0 < p \ll 1$: **small-world** (high clustering AND short paths)

### Properties

| Property | Regular ($p=0$) | Small-World ($p$ small) | Random ($p=1$) |
|----------|-----------------|-------------------------|-----------------|
| Clustering | High | High | Low |
| Average path length | Long ($O(n)$) \| Short ($O(\log n)$) \| Short ($O(\log n)$) |

### Key Insight

- A **few random shortcuts** dramatically reduce path lengths while preserving local clustering
- Explains the "six degrees of separation" phenomenon

### Limitations

- Still no power-law degree distribution
- No community structure
- Rewiring is not a growth mechanism

---

## 6.4 Preferential Attachment (Barabási–Albert)

### Construction

1. Start with $m_0$ connected nodes
2. At each time step, add a new node with $m$ edges
3. New node connects to existing node $i$ with probability proportional to degree:

$$\Pi(k_i) = \frac{k_i}{\sum_j k_j}$$

- **"Rich get richer"** — high-degree nodes attract more connections

### Properties

- **Power-law degree distribution**: $P(k) \sim k^{-3}$
- Produces **scale-free** networks with hubs
- **Short average path length**: $\langle l \rangle \sim \frac{\ln n}{\ln \ln n}$ (ultra-small world)
- **Low clustering**: lower than real social networks

### Why Scale-Free?

- Many real networks show power-law degree distributions:
  - Social networks (follower counts)
  - Citation networks
  - Web link structure
- Preferential attachment is one explanation (but not the only one)

### Limitations

- Clustering is too low compared to real networks
- Exponent is fixed at $\gamma = 3$
- No community structure

---

## 6.5 Comparing Models

| Property | Erdős–Rényi | Watts–Strogatz | Barabási–Albert | Real Social Networks |
|----------|-------------|----------------|------------------|----------------------|
| Degree distribution | Poisson | Narrow | Power-law | Power-law |
| Clustering | Low | **High** | Low | **High** |
| Avg path length | **Short** | **Short** | **Short** | **Short** |
| Hubs | No | No | **Yes** | **Yes** |
| Growth mechanism | No | No | **Yes** | **Yes** |
| Community structure | No | No | No | **Yes** |

### Beyond Basic Models

- **Configuration model**: generate networks with arbitrary degree distribution
- **Stochastic block model**: generates networks with community structure
- **LFR benchmark**: community detection benchmarking with overlapping communities
- **Kronecker graphs**: self-similar graph generation

---

## Key Takeaways

- ER random graphs are simple but fail to capture power-law degrees and high clustering
- Watts-Strogatz explains the small-world phenomenon: high clustering + short paths
- Barabási-Albert explains scale-free networks via preferential attachment
- No single model captures all properties of real social networks
- Understanding these models is essential for benchmarking and null hypothesis testing

---

## References

- T1: Zafarani et al., Ch. 4 — Network Models
