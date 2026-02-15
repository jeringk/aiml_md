# Module 4 — Machine Learning & Traditional Analytical Techniques: Graph Essentials

## Topics

- [[#4.1 Introduction to Graphs|Introduction to Graphs]]
- [[#4.2 Graph Representations|Graph Representations]]
- [[#4.3 Types of Graphs|Types of Graphs]]
- [[#4.4 Graph Properties|Graph Properties]]
- [[#4.5 Special Graphs and Substructures|Special Graphs and Substructures]]

---

## 4.1 Introduction to Graphs

- A **graph** $G = (V, E)$ consists of:
  - $V$: set of **vertices** (nodes) — represent entities (users, pages, etc.)
  - $E$: set of **edges** (links) — represent relationships (friendships, follows, mentions)
- Social media networks are naturally modeled as graphs
- Graph analysis provides insights into structure, influence, and community

### Why Graphs for Social Media?

- Social networks are inherently relational
- Information propagation follows network paths
- Community structure reveals user groupings
- Centrality identifies influential users

---

## 4.2 Graph Representations

### Adjacency Matrix

- $A \in \mathbb{R}^{n \times n}$, where $A_{ij} = 1$ if edge $(i, j)$ exists, 0 otherwise
- For undirected graphs: $A$ is **symmetric** ($A_{ij} = A_{ji}$)
- For weighted graphs: $A_{ij} = w_{ij}$ (weight of the edge)
- Space: $O(n^2)$ — inefficient for sparse graphs

### Adjacency List

- For each node, store a list of its neighbors
- Space: $O(n + m)$ where $m = |E|$
- More efficient for sparse graphs (typical in social networks)

### Edge List

- List of all edges as pairs $(u, v)$
- Simple but less efficient for neighbor queries

### Incidence Matrix

- $B \in \mathbb{R}^{n \times m}$: rows = nodes, columns = edges
- $B_{ie} = 1$ if node $i$ is incident to edge $e$

---

## 4.3 Types of Graphs

| Type | Description | Social Media Example |
|------|-------------|----------------------|
| **Undirected** | Edges have no direction | Facebook friendships |
| **Directed** | Edges have direction (arcs) | Twitter follows, retweets |
| **Weighted** | Edges have numerical weights | Interaction frequency |
| **Unweighted** | All edges are equal | Binary friendship |
| **Bipartite** | Nodes split into two disjoint sets; edges only between sets | Users–Products, Reviewers–Reviews |
| **Multigraph** | Multiple edges between same pair of nodes | Different types of interactions |
| **Signed** | Edges have positive or negative signs | Trust/distrust, agree/disagree |
| **Temporal / Dynamic** | Edges have timestamps, graph changes over time | Evolving social networks |

---

## 4.4 Graph Properties

### Degree

- **Degree** $d(v)$: number of edges incident to node $v$
- In directed graphs:
  - **In-degree** $d_{\text{in}}(v)$: number of incoming edges
  - **Out-degree** $d_{\text{out}}(v)$: number of outgoing edges
- **Degree distribution** $P(k)$: fraction of nodes with degree $k$
  - Social networks often follow a **power-law distribution** (scale-free)

### Paths and Connectivity

- **Path**: sequence of nodes connected by edges
- **Shortest path** (geodesic): path with minimum number of edges
- **Diameter**: maximum shortest path between any two nodes
- **Connected component**: maximal set of nodes where every pair is connected

### Density

$$\text{Density} = \frac{2|E|}{|V|(|V|-1)} \quad \text{(undirected)}$$

- Most social networks are **sparse** (density ≪ 1)

### Clustering Coefficient

- Measures how much a node's neighbors are connected to each other
- **Local clustering coefficient** of node $v$:

$$C(v) = \frac{2 \cdot |\text{edges among neighbors of } v|}{d(v) \cdot (d(v) - 1)}$$

- **Global clustering coefficient** (aka transitivity):

$$C = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$$

- Social networks typically have **high clustering** (friends of friends are often friends)

---

## 4.5 Special Graphs and Substructures

### Complete Graph ($K_n$)
- Every pair of nodes is connected
- $|E| = \frac{n(n-1)}{2}$

### Clique
- A **subset** of nodes where every pair is connected (complete subgraph)
- Finding maximum clique is NP-hard

### Tree
- Connected graph with no cycles
- $|E| = |V| - 1$

### Star Graph
- One central node connected to all others

### Ego Network
- A node (ego) and all its direct neighbors (alters), plus edges among alters
- Common in social network analysis — represents a user's local network

---

## Key Takeaways

- Graphs are the fundamental data structure for modeling social networks
- Key representations: adjacency matrix, adjacency list, edge list
- Social networks are typically sparse, scale-free, and exhibit high clustering
- Graph properties (degree, paths, clustering) reveal structural patterns in social media
- Understanding graph basics is prerequisite for network measures, models, and community detection

---

## References

- T1: Zafarani et al., Ch. 2 — Graph Essentials
