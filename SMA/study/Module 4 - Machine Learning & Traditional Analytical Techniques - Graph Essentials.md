# Module 4: Graph Essentials

## Lecture Topics Covered
- Graph Essentials
- Traversal Algorithms & Shortest Path (Dijkstra's)
- Principle of Repeated Improvement (Hubs and Authorities)
- Matrix form and Convergence

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
- Space: $O(n + m)$ where $m = \|E\|$
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

$$\text{Density} = \frac{2\|E\|}{\|V\|(\|V\|-1)} \quad \text{(undirected)}$$

- Most social networks are **sparse** (density ≪ 1)

#### Numerical Example (Density)
For an undirected graph with $V = \{A, B, C, D\}$ (so $\|V\| = 4$) and edges $\{(A,B), (B,C), (C,D)\}$ (so $\|E\| = 3$):
- Maximum possible edges $= \frac{4(4-1)}{2} = 6$.
- $\text{Density} = \frac{2(3)}{4(3)} = \frac{6}{12} = 0.5$.
- This means 50% of all possible connections actually exist.

### Clustering Coefficient

- Measures how much a node's neighbors are connected to each other
- **Local clustering coefficient** of node $v$:

$$C(v) = \frac{2 \cdot \|\text{edges among neighbors of } v\|}{d(v) \cdot (d(v) - 1)}$$

- **Global clustering coefficient** (aka transitivity):

$$C = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$$

- Social networks typically have **high clustering** (friends of friends are often friends)

#### Numerical Example (Clustering Coefficient)
Consider an undirected graph with nodes $A, B, C, D$ and edges:
$(A,B), (A,C), (B,C), (C,D)$ (Nodes A,B,C form a triangle. D is only attached to C).

**1. Local Clustering $C(v)$:**
- For node $C$: has $d(C) = 3$ neighbors ($A, B, D$).
- Maximum possible edges between its 3 neighbors $= \frac{3 \times 2}{2} = 3$.
- Actual edges between its neighbors: Only $(A,B)$ exists (so 1 edge).
- $C(C) = \frac{2 \times 1}{3 \times 2} = \frac{1}{3} \approx 0.333$.

**2. Global Clustering (Transitivity):**
- Number of triangles $= 1$ (the $A-B-C$ triangle).
- Number of connected triples (paths of length 2):
  - Centered at A: $B-A-C$ (1)
  - Centered at B: $A-B-C$ (1)
  - Centered at C: $A-C-B, A-C-D, B-C-D$ (3)
  - Centered at D: None (0)
  - Total triples $= 1 + 1 + 3 + 0 = 5$.
- $C = \frac{3 \times 1}{5} = 0.6$.

---

## 4.5 Special Graphs and Substructures

### Complete Graph ($K_n$)
- Every pair of nodes is connected
- $\|E\| = \frac{n(n-1)}{2}$

### Clique
- A **subset** of nodes where every pair is connected (complete subgraph)
- Finding maximum clique is NP-hard

### Tree
- Connected graph with no cycles
- $\|E\| = \|V\| - 1$

### Star Graph
- One central node connected to all others

### Ego Network
- A node (ego) and all its direct neighbors (alters), plus edges among alters
- Common in social network analysis — represents a user's local network

---

---

## 4.6 Traversal Algorithms

- Traversal algorithms are techniques to visit or explore nodes and edges systematically in a graph.
- These form the foundation for searching, finding paths, and measuring distances in social networks.

### Breadth-First Search (BFS)
- Explores the network level by level, starting from a source node.
- Validates the "six degrees of separation" or small-world phenomenon by finding the shortest path (minimum number of hops) in unweighted graphs.
- Steps:
  1. Visit the root node and add it to a queue.
  2. Dequeue a node, visit all its unvisited neighbors, and enqueue them.
  3. Repeat until the queue is empty.
- Complexity: $O(\|V\| + \|E\|)$
- Application: Finding friends of friends, discovering nodes within a certain distance.

### Depth-First Search (DFS)
- Explores as far as possible along each branch before backtracking.
- Uses a stack (or recursion) instead of a queue.
- Steps:
  1. Visit the root node.
  2. Recursively visit the first unvisited neighbor.
  3. Backtrack when a node has no unvisited neighbors.
- Complexity: $O(\|V\| + \|E\|)$
- Application: Detecting cycles, finding connected components.

### Shortest Path: Dijkstra's Algorithm
- Finds the shortest paths between a given source node and all other nodes in a **weighted** graph.
- Cannot handle graphs with negative weight edges.
- Steps:
  1. Initialize distances from source to all nodes as infinite, and distance to the source itself as 0.
  2. Create a priority queue (or unvisited set) and add all nodes.
  3. Extract the node with the minimum distance.
  4. For the current node, consider all its unvisited neighbors and calculate their tentative distances through the current node.
  5. If the newly calculated distance is less than the current assigned value, update the shortest distance.
  6. Mark the current node as visited. A visited node will never be checked again.
  7. Repeat until the target node is visited or the queue is empty.
- Complexity: $O((\|V\| + \|E\|) \log \|V\|)$ using a minimum priority queue.
- Application: Calculating minimum transport costs, finding the fastest routes between locations on graph networks.

---

## 4.7 Hubs and Authorities (HITS Algorithm)

- **Hyperlink-Induced Topic Search (HITS)**: An algorithm introduced by Jon Kleinberg to evaluate the importance of web pages (or social media nodes).
- It is based on the **Principle of Repeated Improvement**.
- Every node has two scores measuring different types of importance:
  - **Hub Score ($h$)**: A good hub is a node that points to many good authorities.
  - **Authority Score ($a$)**: A good authority is a node that is pointed to by many good hubs.

### Mathematical Formulation
- These two scores are mutually dependent and updated iteratively:
  $$a^{(k)}_i = \sum_{j: j \to i} h^{(k-1)}_j$$
  $$h^{(k)}_i = \sum_{j: i \to j} a^{(k)}_j$$
- After every iteration, the scores are **normalized** so they do not grow infinitely.

### Matrix Form and Convergence
- Let $A$ be the adjacency matrix. The updates can be written as:
  $$a^{(k)} = A^T h^{(k-1)}$$
  $$h^{(k)} = A a^{(k)}$$
- Substituting one into the other:
  $$a^{(k)} = A^T A a^{(k-1)}$$
  $$h^{(k)} = A A^T h^{(k-1)}$$
- **Convergence**: As the iterations continue ($k \to \infty$), the authority vector converges to the principal eigenvector of $A^T A$, and the hub vector converges to the principal eigenvector of $A A^T$.
- **Interpretation in Social Media**: Hubs could be curation accounts (e.g., news aggregators or link directories) and Authorities could be the primary sources of original content.

### Numerical Example (HITS - 1 Iteration)
Consider a directed network of 3 nodes:
- Links: $1 \to 2$, $1 \to 3$, $2 \to 3$
- Initialize all hub ($h$) and authority ($a$) scores to 1. $h_0 = [1, 1, 1]^T$, $a_0 = [1, 1, 1]^T$.

**Iteration 1:**
1. **Update Authorities** (sum of hubs pointing to it):
   - $a_1(1) = 0$ (no incoming links)
   - $a_1(2) = h_0(1) = 1$ (only node 1 points to 2)
   - $a_1(3) = h_0(1) + h_0(2) = 1 + 1 = 2$
   - $a_1$ vector $= [0, 1, 2]^T$

2. **Update Hubs** (sum of *updated* authorities it points to):
   - $h_1(1) = a_1(2) + a_1(3) = 1 + 2 = 3$
   - $h_1(2) = a_1(3) = 2$
   - $h_1(3) = 0$ (node 3 points nowhere)
   - $h_1$ vector $= [3, 2, 0]^T$

3. **Normalize (Optional for strict step-by-step but standard practice):**
   - Typically, divide by the sum or Euclidean norm so scores don't explode.
   - For example, if normalizing by sum:
     - $a_1 = [0, \frac{1}{3}, \frac{2}{3}]^T$
     - $h_1 = [\frac{3}{5}, \frac{2}{5}, 0]^T$
Node 3 is the best authority (most incoming), and Node 1 is the best hub (points to best authorities).

---

## Key Takeaways

- Graphs are the fundamental data structure for modeling social networks
- Key representations: adjacency matrix, adjacency list, edge list
- Social networks are typically sparse, scale-free, and exhibit high clustering
- Graph properties (degree, paths, clustering) reveal structural patterns in social media
- Traversal algorithms (BFS, DFS) are essential for graph searching and measuring distances
- HITS defines node importance mutually through Hubs and Authorities, converging via eigenvectors
- Understanding graph basics is prerequisite for network measures, models, and community detection

