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

## 7.5 Lecture 7: Graph Essentials & Community Detection Basics

### 7.5.1 Part 1: Homophily and the Origin of Communities
* **Definition:** Homophily is the tendency of individuals to associate and bond with similar others, often summarized as "Birds of a Feather Flock Together".
* **Mechanism:** Similar nodes tend to attract each other, while dissimilar nodes tend to move away from each other.
* **Result:** This behavior naturally causes the formation of a community structure in a social network.
* **Categories of Homophily:** It occurs based on attributes such as Age, Sex and Gender, Class (Education, occupation), Religion/Race/Ethnicity, Interests, and Organizational role.

---

### 7.5.2 Part 2: Communities in a Network
* Identifying communities provides insight into the inherent network structure.
* However, community detection is an "ill-defined problem" because what constitutes a 'community' is not concretely defined.
* It is often hard to reliably define a ground-truth annotation for communities, and there is no single standard measure to assess performance.

#### Practical Applications
Community detection is used across various domains:
* **Recommendation Systems:** Improving recommendation quality by separating like-minded people.
* **Information & Marketing:** Controlling information diffusion and designing better target marketing strategies.
* **Healthcare:** Restricting epidemic propagation by isolating and immunizing vulnerable populations.
* **Security:** Better anomaly detection, criminology applications, and detecting terrorist groups.

---

### 7.5.3 Part 3: Types of Communities

#### Disjoint Communities
* Also referred to as "flat communities".
* Each node in the network can belong to at most one community.
* This differs from disconnected components because nodes in two different communities can still have connecting edges, which are referred to as bridges.
* *Example:* Full-time employees of an organization.

#### Overlapping Communities
* Members can belong to more than one community at a time, and communities can even share edges.
* This is a more realistic and generic community structure, though harder to find mathematically than flat communities.
* *Example:* Various distinct groups that a single person might belong to in a social network.

#### Hierarchical Communities
* The outcome of merging two or more flat or overlapping communities.
* They can be linked to other hierarchical, overlapping, or flat communities at different levels.
* *Example:* Various city-level communities merging to form a larger state-level community.

#### Local Communities
* Shows a community structure strictly from a local perspective without focusing on the overall global structure.
* *Example:* A citation network formed specifically by research groups inside a single university.

---

### 7.5.4 Part 4: Node-Centric Community Detection
These methods use the specific properties of individual nodes to find community structures.

#### Cliques
* A subgraph where every vertex-pair is adjacent (has a diameter of 1).
* **Issues:** Finding cliques is NP-complete, the constraint is too strict, and large cliques are rarely present in real social networks.

#### K-Cliques & Relaxations
* **K-Clique:** The maximal subset of vertices where the shortest distance between any two nodes is $\le K$. 
    * *Note:* A node *not* present in the K-clique can actually contribute to forming the shortest distance path connecting the nodes.
* **K-Clan:** A stricter version of a K-clique. It requires that the shortest distance $\le K$ must be formed *only* using nodes present in the set under inspection. It maintains the maximality condition.
* **K-Club:** A K-clan minus the maximality condition. Every K-clan is a K-club as well as a K-clique.

#### Degree-Based Relaxations
* **K-Plex:** A subset $S$ of vertices is a K-plex if every vertex in the induced subgraph has a degree of at least $|S|-K$.
* **K-Core:** A subgraph where each node has a degree $\ge K$.
    * It is generated by recursively removing nodes of degree $< K$ until no such nodes remain.
    * Checking if a network is a K-core is computationally easy, but finding the *maximal* K-core is NP-complete.

---

### 7.5.5 Part 5: Graph Partitioning and Cuts

When communities interact, most interactions are within the group, whereas interactions between groups are few. Community detection can be framed as a minimum cut problem.
* **Cut:** A partition of vertices of a graph into two disjoint sets.
* **Minimum Cut:** Finding a partition such that the number of edges between the two sets is minimized.
* *Issue with Min Cut:* It often returns highly imbalanced partitions (e.g., isolating a single node).

To solve the imbalance, the objective function is changed to consider community size:
* **Ratio Cut:** Considers the raw number of nodes in each community ($|C_i|$).
  $$Ratio Cut(\pi)=\frac{1}{k}\sum_{i=1}^{k}\frac{cut(C_i,\overline{C}_i)}{|C_i|}$$
* **Normalized Cut:** Considers the sum of degrees (volume) in each community ($vol(C_i)$).
  $$Normalized Cut(\pi)=\frac{1}{k}\sum_{i=1}^{k}\frac{cut(C_i,\overline{C_i})}{vol(C_i)}$$

---

### 7.5.6 Part 6: Girvan-Newman Algorithm (Edge Betweenness)
* **Concept:** The strength of a tie can be measured by edge betweenness.
* **Edge Betweenness:** Defined as the number of shortest paths that pass along a specific edge.
* Edges with higher betweenness tend to act as the primary bridges between two communities.
* **Divisive Clustering Mechanism:** The algorithm works by progressively calculating and removing the edges with the highest betweenness score to slowly separate the graph into isolated communities.

---

## 7.6 Lecture 8: Graph Essentials & Community Detection (Modularity)

### 7.6.1 Part 1: Recap of Previous Concepts
Before diving into modularity, the lecture briefly recaps several foundational graph concepts:
* **Types of Communities:** Disjoint, overlapping, hierarchical, and local.
* **Node-Centric Community Detection:** Cliques, K-cliques, K-clan, K-club, K-plex, and K-core.
* **Graph Cuts:** Ratio cut and Normalized cut.
* **Girvan-Newman Algorithm:** Edge betweenness-based community detection.

---

### 7.6.2 Part 2: Introduction to Modularity

Node-centric methods are often not very useful when dealing with extremely large networks. To solve this, we use a network-centric metric called **Modularity** (derived from the word 'module') to determine the overall quality of a community structure.

#### The Core Principle: Actual vs. Expected Edges
Modularity is based on the principle of comparing the *actual* number of edges in a subgraph to its *expected* number of edges.
* The expected number of edges is calculated by assuming a **null model**.
* In this null model, each vertex is randomly connected to other vertices, completely ignoring the community structure.
* However, certain structural properties of the original network, specifically the **degree distribution**, are preserved in the null model.

#### Mathematical Formulas for Modularity
**General Formula:**
$$Q=\frac{1}{2m}\sum_{ij}\left(A_{ij}-\frac{k_i k_j}{2m}\right)\delta(c_i,c_j)$$

* $A_{ij}$: Adjacency matrix element (1 if an edge exists between $i$ and $j$, else 0).
* $k_i, k_j$: Degrees of nodes $i$ and $j$.
* $m$: Total number of edges in the network.
* $\delta(c_i, c_j)$: Kronecker delta function (1 if nodes $i$ and $j$ are in the *same* community, else 0).

**Simplified Formula (Aggregated by Community):**
$$Q=\sum_c\left[\frac{l_c}{m}-\left(\frac{d_c}{2m}\right)^2\right]$$

* $l_c$: Number of internal edges strictly inside community $c$.
* $d_c$: Sum of the degrees of all nodes inside community $c$.
* $m$: Total edges in the entire network.

#### Interpreting Modularity Values ($Q$)
Modularity can be positive, negative, or zero. A positive modularity indicates the presence of a strong community structure, meaning dense intra-module connections and sparse inter-module connections.

| Modularity ($Q$) | Interpretation |
| :--- | :--- |
| **0** | No community structure (random) |
| **0.3 - 0.5** | Moderate community structure |
| **> 0.5** | Strong community structure |

---

### 7.6.3 Part 3: Step-by-Step Calculation Examples

#### Example 1: Single Community Contribution
* **Given:** Total edges in network $m = 10$. For Community $C$: internal edges $l_c = 4$, sum of degrees $d_c = 8$.
* **Calculation:**
  $Q_c = \frac{4}{10} - (\frac{8}{20})^2$
  $Q_c = 0.4 - (0.4)^2$
  $Q_c = 0.4 - 0.16 = \mathbf{0.24}$

#### Example 2: Total Network Modularity
* **Given:** Total edges $m = 14$.
  * Community A: $l_A = 5$, $d_A = 12$.
  * Community B: $l_B = 6$, $d_B = 16$.
* **Calculation:**
  1. $Q_A = \frac{5}{14} - (\frac{12}{28})^2 = 0.357 - 0.184 = 0.173$
  2. $Q_B = \frac{6}{14} - (\frac{16}{28})^2 = 0.429 - 0.327 = 0.102$
  3. **Total $Q$** $= Q_A + Q_B = 0.173 + 0.102 = \mathbf{0.275}$

---

### 7.6.4 Part 4: Modularity Maximization Algorithms

Different community assignments lead to different modularity values. Finding an assignment that maximizes the overall network modularity is a standard way to detect communities. Two popular algorithms are the Fast Greedy Algorithm and the Louvain Method.

#### Fast Greedy Algorithm (Clauset et al., 2004)
* Starts with an initial community assignment where nodes are separate.
* Iteratively and greedily merges communities together.
* At each step, it chooses the merge that results in the maximum increase in overall modularity.

#### Louvain Method
* A highly efficient, multi-pass heuristic.
* **Phase 1: Modularity Optimization.** Local optimization of modularity is performed by moving nodes between neighboring communities.
* **Phase 2: Community Aggregation.** Nodes in the same community are aggregated into a single "super-node" to build a new network.
* These passes repeat iteratively until modularity can no longer be increased.

---

### 7.6.5 Part 5: Limitations of Modularity Maximization

Despite its popularity, relying solely on modularity maximization has two major drawbacks:
1. **Resolution Limit:** Well-connected, smaller communities tend to be forced into merges with larger communities, even if the resulting merged community isn't very dense. It fails to detect small communities that are well-separated (densely connected internally but only having a single inter-community edge with the rest of the network).
2. **Degeneracy of Solutions:** In many networks, there can be an exponential number of vastly different community structures that all yield the exact same (maximum) modularity value, making it hard to find the "true" structure.

---

## Key Takeaways

- Communities are densely connected subgroups — core concept in social network analysis
- Member-based methods assess individual nodes; group-based methods evaluate entire communities
- Modularity is the standard quality measure; Louvain/Leiden are fast, widely-used optimizers
- Spectral clustering provides a principled mathematical approach
- Communities evolve over time — tracking these changes reveals social dynamics
- GenAI enhances community analysis with labeling, summarization, and characterization

