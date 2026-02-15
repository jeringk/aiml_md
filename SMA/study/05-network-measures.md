# Module 5 — Machine Learning & Traditional Analytical Techniques: Network Measures

## Topics

- [[#5.1 Centrality Measures|Centrality Measures]]
- [[#5.2 Degree Centrality|Degree Centrality]]
- [[#5.3 Betweenness Centrality|Betweenness Centrality]]
- [[#5.4 Closeness Centrality|Closeness Centrality]]
- [[#5.5 Eigenvector Centrality and PageRank|Eigenvector Centrality and PageRank]]
- [[#5.6 Other Network Measures|Other Network Measures]]

---

## 5.1 Centrality Measures

- **Centrality**: quantifies how "important" a node is in a network
- Different definitions of importance lead to different centrality measures
- In social media: **who are the most influential users? Which nodes are bridges?**

| Centrality | Measures | Social Media Interpretation |
|------------|----------|-----------------------------|
| **Degree** | Number of connections | Popularity, activity |
| **Betweenness** | How often a node lies on shortest paths | Brokerage, information control |
| **Closeness** | Average distance to all other nodes | Speed of information reach |
| **Eigenvector / PageRank** | Importance of neighbors | Influential connections |

---

## 5.2 Degree Centrality

$$C_D(v) = \frac{d(v)}{n - 1}$$

- Normalized by maximum possible degree ($n - 1$)
- In directed graphs:
  - **In-degree centrality**: popularity (how many follow you)
  - **Out-degree centrality**: activity (how many you follow)
- Simple but effective — highly correlated with influence in many networks
- Limitation: doesn't consider **position** in the network, only local structure

### Degree Distribution

- Social networks often follow **power-law** (scale-free):

$$P(k) \propto k^{-\gamma} \quad (\gamma \text{ typically between 2 and 3})$$

- A few nodes have very high degree (**hubs**), most have low degree

---

## 5.3 Betweenness Centrality

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$: total number of shortest paths from $s$ to $t$
- $\sigma_{st}(v)$: number of shortest paths from $s$ to $t$ that pass through $v$

### Interpretation

- High betweenness → node acts as a **bridge** or **gatekeeper**
- Removing high betweenness nodes can **disconnect** the network
- In social media: users who bridge different communities

### Normalized Betweenness

$$C_B'(v) = \frac{C_B(v)}{(n-1)(n-2)/2} \quad \text{(undirected)}$$

### Computation

- Brandes' algorithm: $O(nm)$ for unweighted, $O(nm + n^2 \log n)$ for weighted

---

## 5.4 Closeness Centrality

$$C_C(v) = \frac{n - 1}{\sum_{u \neq v} d(v, u)}$$

Where $d(v, u)$ is the shortest path distance from $v$ to $u$

### Interpretation

- High closeness → node can **reach all others quickly**
- In social media: users who can spread information efficiently
- Problem: undefined for disconnected graphs (infinite distances)
  - Solution: use **harmonic centrality**: $C_H(v) = \sum_{u \neq v} \frac{1}{d(v, u)}$

---

## 5.5 Eigenvector Centrality and PageRank

### Eigenvector Centrality

- A node is important if its **neighbors are important** (recursive definition)

$$C_E(v) = \frac{1}{\lambda} \sum_{u \in N(v)} C_E(u) \quad \Rightarrow \quad Ax = \lambda x$$

- $x$ is the eigenvector corresponding to the **largest eigenvalue** $\lambda_1$ of adjacency matrix $A$
- Computed iteratively via **power iteration**

### PageRank

- Variant of eigenvector centrality designed for directed graphs (originally for the web)

$$\text{PR}(v) = \frac{1 - d}{n} + d \sum_{u \in N_{\text{in}}(v)} \frac{\text{PR}(u)}{d_{\text{out}}(u)}$$

Where:
- $d$: damping factor (typically 0.85)
- Models a **random surfer** who follows links with probability $d$ and jumps to a random page with probability $1 - d$

### Application in Social Media

- Ranking users by influence (Twitter, citation networks)
- Content recommendation
- Identifying authoritative sources

---

## 5.6 Other Network Measures

### Network-Level Measures

| Measure | Formula | Description |
|---------|---------|-------------|
| **Average degree** | $\langle k \rangle = \frac{2|E|}{|V|}$ | Mean connections per node |
| **Average path length** | $\langle l \rangle = \frac{1}{n(n-1)} \sum_{i \neq j} d(i,j)$ | Mean shortest path |
| **Diameter** | $\max_{i,j} d(i,j)$ | Longest shortest path |
| **Density** | $\frac{2|E|}{|V|(|V|-1)}$ | Fraction of possible edges |
| **Transitivity** | $\frac{3 \times \text{triangles}}{\text{triples}}$ | Global clustering |
| **Reciprocity** | Fraction of mutual edges | Mutual relationships (directed) |
| **Assortativity** | Correlation of degrees at ends of edges | Do similar-degree nodes connect? |

### Small-World Property

- Social networks exhibit:
  - **Short average path length** (six degrees of separation)
  - **High clustering coefficient**
- Watts-Strogatz model captures this phenomenon

---

## Key Takeaways

- Centrality measures quantify node importance from different perspectives
- Degree = popularity, Betweenness = brokerage, Closeness = reachability, PageRank = recursive importance
- Social networks are typically scale-free (power-law degree) and small-world (short paths, high clustering)
- Network measures help identify influential users, bridges, and structural patterns

---

## References

- T1: Zafarani et al., Ch. 3 — Network Measures
