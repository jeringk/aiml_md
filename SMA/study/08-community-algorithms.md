# Module 8 — Community and Interactions: Community Detection Algorithms

## Topics

- [[#8.1 Girvan-Newman Algorithm|Girvan-Newman Algorithm]]
- [[#8.2 Brute-Force Search Algorithm|Brute-Force Search Algorithm]]
- [[#8.3 Community Evaluation|Community Evaluation]]
- [[#8.4 Review of Sessions 1–7|Review of Sessions 1–7]]

---

## 8.1 Girvan-Newman Algorithm

- A **divisive** (top-down) hierarchical algorithm that removes edges to uncover community structure
- Based on **edge betweenness centrality**: edges with high betweenness are likely bridges between communities

### Algorithm

1. Calculate **edge betweenness** for all edges in the network
2. **Remove** the edge with the highest betweenness
3. **Recalculate** edge betweenness for all remaining edges
4. Repeat steps 2–3 until no edges remain

### Edge Betweenness

$$EB(e) = \sum_{s \neq t} \frac{\sigma_{st}(e)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$: number of shortest paths from $s$ to $t$
- $\sigma_{st}(e)$: number of those paths that pass through edge $e$

### Output

- Produces a **dendrogram** — hierarchical decomposition of the network
- Cut the dendrogram at the level that **maximizes modularity** $Q$

### Example Walkthrough

```
Step 1: Compute edge betweenness for all edges
Step 2: Remove edge with highest betweenness (bridge between groups)
Step 3: Recompute - components may split
Step 4: Continue until desired number of communities or max Q
```

### Complexity

- Recomputing betweenness at each step: $O(m^2 n)$ for the full algorithm
- Not scalable for very large networks

---

## 8.2 Brute-Force Search Algorithm

- **Exhaustive approach**: try all possible partitions and pick the best one

### For Graph Bisection

1. Enumerate all possible ways to split $n$ nodes into two groups
2. Evaluate each partition using a quality metric (e.g., modularity, cut size)
3. Select the partition that optimizes the metric

### Complexity

- Number of ways to partition $n$ nodes into 2 groups: $\binom{n}{n/2} \sim 2^n$
- For $k$ communities: grows even faster (Bell numbers)
- **Exponential** — only feasible for very small networks

### Why Study Brute-Force?

- Establishes the **optimal baseline** for comparison
- Demonstrates why heuristic-based algorithms (Louvain, spectral, etc.) are necessary
- Useful for **validation** on small test networks

---

## 8.3 Community Evaluation

- How do we measure the **quality** of detected communities?

### Internal Evaluation (No Ground Truth)

| Metric | Description |
|--------|-------------|
| **Modularity ($Q$)** | Fraction of edges within communities vs. expected by chance |
| **Conductance** | Ratio of external edges to total edges for a community |
| **Internal density** | Fraction of possible internal edges that exist |
| **Cut ratio** | Fraction of possible external edges that exist |
| **Coverage** | Fraction of total edges that are within communities |
| **Performance** | Fraction of correctly classified node pairs |

### External Evaluation (With Ground Truth)

| Metric | Description |
|--------|-------------|
| **Normalized Mutual Information (NMI)** | Information-theoretic measure of overlap between detected and ground truth communities |
| **Adjusted Rand Index (ARI)** | Measures agreement between two partitions, adjusted for chance |
| **F1-score / Purity** | Precision/recall-based measures |
| **Omega Index** | Extends ARI for overlapping communities |

### NMI Formula

$$\text{NMI}(X, Y) = \frac{2 \cdot I(X; Y)}{H(X) + H(Y)}$$

Where $I(X; Y)$ is the mutual information and $H$ is the entropy

- $\text{NMI} = 1$: perfect match
- $\text{NMI} = 0$: independent (no agreement)

### Practical Guidelines

- Use **modularity** when no ground truth is available
- Use **NMI** or **ARI** when ground truth exists
- Report multiple metrics — no single metric captures all aspects of quality
- Compare against **random baselines** (e.g., random partition) and **null models**

---

## 8.4 Review of Sessions 1–7

### Summary of Key Concepts So Far

| Session | Topic | Key Concepts |
|---------|-------|-------------|
| 1 | Social Media Mining Intro | Importance, characteristics, platforms, challenges |
| 2 | NLP - Sentiment Analysis | Document/sentence/aspect-level sentiment, lexicon-based vs. ML |
| 3 | NLP - IE & Summarization | NER, text summarization, GenAI applications |
| 4 | Graph Essentials | Graph types, representations, properties |
| 5 | Network Measures | Centrality (degree, betweenness, closeness, PageRank) |
| 6 | Network Models | ER, Watts-Strogatz, Barabási-Albert |
| 7 | Community Detection | Member-based, group-based, modularity, evolution |

---

## Key Takeaways

- Girvan-Newman uses edge betweenness to divisively detect communities — effective but $O(m^2 n)$
- Brute-force is exponential — motivates the need for heuristic algorithms
- Community evaluation uses modularity (no ground truth) or NMI/ARI (with ground truth)
- Multiple evaluation metrics should be used together for robust assessment

---

## References

- T1: Zafarani et al., Ch. 6 — Community Analysis
