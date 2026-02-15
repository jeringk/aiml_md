# Module 7 — Community and Interactions: Community Detection

## Topics

- [[#7.1 Member-Based Community Detection|Member-Based Community Detection]]
- [[#7.2 Group-Based Community Detection|Group-Based Community Detection]]
- [[#7.3 Community Evolution|Community Evolution]]
- [[#7.4 Leveraging GenAI for Community Analysis|Leveraging GenAI for Community Analysis]]

---

## 7.1 Member-Based Community Detection

- **Community**: a group of nodes that are **densely connected internally** and **sparsely connected externally**
- Member-based approaches evaluate community membership based on **individual node properties**

### Node-Centric Approaches

#### Degree-Based

- Nodes belonging to a community should have a **minimum internal degree**
- **Strong community**: every node has more connections inside than outside
- **Weak community**: total internal degree > total external degree

$$\forall v \in C: \quad d_{\text{int}}(v) > d_{\text{ext}}(v) \quad \text{(strong)}$$

#### Reachability-Based

- Community members should be reachable from each other within a few hops
- **$k$-clique community**: maximal sets of nodes where every pair has a path of length ≤ $k$ through intermediate nodes within the group
- **$k$-clique percolation** (Palla et al.):
  - Find all cliques of size $k$
  - Two $k$-cliques are adjacent if they share $k-1$ nodes
  - Communities = connected components in the clique overlap graph

#### Similarity-Based

- Group nodes with **similar attributes or behavior** together
- **Structural equivalence**: nodes with the same neighbors
- **Regular equivalence**: nodes with similar roles (even if different neighbors)
- Use cosine similarity, Jaccard coefficient on neighborhoods

---

## 7.2 Group-Based Community Detection

- Evaluate the community as a **whole group** rather than individual nodes

### Modularity-Based

- **Modularity** $Q$: measures the quality of a partition into communities

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- $A_{ij}$: adjacency matrix
- $k_i, k_j$: degrees of nodes $i, j$
- $m$: total number of edges
- $\delta(c_i, c_j) = 1$ if $i$ and $j$ are in the same community
- $\frac{k_i k_j}{2m}$: expected number of edges under a random null model

- $Q \in [-0.5, 1]$: higher is better. Values > 0.3 indicate significant community structure

### Modularity Optimization

- **Greedy agglomerative** (Clauset-Newman-Moore):
  - Start with each node as its own community
  - Repeatedly merge the pair of communities that gives the largest gain in $Q$
  - $O(n \log^2 n)$ for sparse graphs
- **Louvain algorithm**:
  - Phase 1: greedily move nodes to maximize $Q$
  - Phase 2: aggregate communities into super-nodes
  - Repeat until no improvement
  - Very fast, widely used
- **Leiden algorithm**: improved version of Louvain, guarantees connected communities

### Graph Partitioning

- **Spectral clustering**: use eigenvectors of the **graph Laplacian**
  - $L = D - A$ (unnormalized Laplacian) or $L_{\text{sym}} = D^{-1/2} L D^{-1/2}$
  - Second smallest eigenvector (Fiedler vector) provides the optimal bisection
  - For $k$ communities, use the $k$ smallest eigenvectors → K-means clustering

---

## 7.3 Community Evolution

- Social media communities are **dynamic** — they change over time
- Community evolution events:

| Event | Description |
|-------|-------------|
| **Growth** | Community gains new members |
| **Shrinkage** | Community loses members |
| **Merging** | Two or more communities combine |
| **Splitting** | A community divides into smaller ones |
| **Birth** | A new community forms |
| **Death** | A community dissolves |
| **Continuation** | Community persists with minor changes |

### Tracking Community Evolution

1. Detect communities at each time snapshot
2. Match communities across time steps using **Jaccard similarity** or other overlap measures
3. Classify evolution events based on matched communities

### Challenges

- Choosing time granularity
- Instability of community detection algorithms across snapshots
- Distinguishing real evolution from algorithmic noise

---

## 7.4 Leveraging GenAI for Community Analysis

- **LLMs** can enhance community analysis in several ways:

### Applications

| Application | How GenAI Helps |
|-------------|-----------------|
| **Community labeling** | Generate descriptive labels for detected communities based on member attributes/content |
| **Community summary** | Summarize typical topics, sentiment, interests of community members |
| **Anomaly detection** | Identify unusual community patterns via LLM reasoning |
| **Community characterization** | Describe community dynamics and evolution in natural language |
| **Cross-modal analysis** | Combine text, network, and profile data for richer community understanding |

### Example Workflow

1. Detect communities using traditional algorithms (Louvain, Leiden)
2. Extract representative content (posts, profiles) from each community
3. Use LLM to generate community summaries and labels
4. Use LLM for comparative analysis across communities

---

## Key Takeaways

- Communities are densely connected subgroups — core concept in social network analysis
- Member-based methods assess individual nodes; group-based methods evaluate entire communities
- Modularity is the standard quality measure; Louvain/Leiden are fast, widely-used optimizers
- Spectral clustering provides a principled mathematical approach
- Communities evolve over time — tracking these changes reveals social dynamics
- GenAI enhances community analysis with labeling, summarization, and characterization

---

## References

- T1: Zafarani et al., Ch. 6 — Community Analysis
