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

### Running Example Graph

To understand centrality measures, we will consistently use a simple **undirected** network with 5 nodes ($V = \{A, B, C, D, E\}$) and the following 5 edges:
- $A - B$, $A - C$, $B - C$ (a triangle)
- $C - D$, $D - E$ (a tail)

**Visualizing the Network:**  
```text
  A
 / \
B---C---D---E
```

- Total nodes ($n$) = $5$
- Maximum possible degree = $n - 1 = 4$

---

## 5.2 Degree Centrality

$$C_D(v) = \frac{d(v)}{n - 1}$$

- Normalized by maximum possible degree ($n - 1$)
- In directed graphs:
  - **In-degree centrality**: popularity (how many follow you)
  - **Out-degree centrality**: activity (how many you follow)
- Simple but effective — highly correlated with influence in many networks
- Limitation: doesn't consider **position** in the network, only local structure

### Numerical Example (Degree Centrality)

Using our 5-node example graph (`A, B, C, D, E`):

1. **Calculate raw degrees $d(v)$:**
   - $d(A) = 2$ (connected to B, C)
   - $d(B) = 2$ (connected to A, C)
   - $d(C) = 3$ (connected to A, B, D)
   - $d(D) = 2$ (connected to C, E)
   - $d(E) = 1$ (connected to D)

2. **Calculate normalized degree centrality $C_D(v) = \frac{d(v)}{n - 1}$:**
   - $C_D(A) = 2 / 4 = 0.5$
   - $C_D(B) = 2 / 4 = 0.5$
   - $C_D(C) = 3 / 4 = 0.75$ (**Most central by degree**)
   - $C_D(D) = 2 / 4 = 0.5$
   - $C_D(E) = 1 / 4 = 0.25$

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

### Numerical Example (Betweenness Centrality)

Using our 5-node example graph. We calculate the sum of the fraction of shortest paths passing through each node.

**Shortest paths between all pairs:**
- There are $\frac{n(n-1)}{2} = 10$ unique node pairs.
- Pairs that don't need intermediate nodes (direct edges): $(A,B), (A,C), (B,C), (C,D), (D,E)$. No node lies *between* these.
- Pairs that require intermediate nodes:
  1. **A to D**: Shortest path is $A \rightarrow$ **C** $\rightarrow D$. Passes through C.
  2. **A to E**: Shortest path is $A \rightarrow$ **C** $\rightarrow$ **D** $\rightarrow E$. Passes through C and D.
  3. **B to D**: Shortest path is $B \rightarrow$ **C** $\rightarrow D$. Passes through C.
  4. **B to E**: Shortest path is $B \rightarrow$ **C** $\rightarrow$ **D** $\rightarrow E$. Passes through C and D.
  5. **C to E**: Shortest path is $C \rightarrow$ **D** $\rightarrow E$. Passes through D.

**1. Raw Betweenness $C_B(v)$:**
- $C_B(A) = 0$
- $C_B(B) = 0$
- $C_B(C) = 1 \text{ (from A-D)} + 1 \text{ (from A-E)} + 1 \text{ (from B-D)} + 1 \text{ (from B-E)} = 4$ (**Highest**)
- $C_B(D) = 1 \text{ (from A-E)} + 1 \text{ (from B-E)} + 1 \text{ (from C-E)} = 3$
- $C_B(E) = 0$

**2. Normalized Betweenness $C'_B(v) = \frac{C_B(v)}{(n-1)(n-2)/2}$:**
For $n=5$, denominator $= (4 \times 3) / 2 = 6$.
- $C'_B(C) = 4 / 6 \approx 0.667$
- $C'_B(D) = 3 / 6 = 0.5$
- Others $= 0$

---

## 5.4 Closeness Centrality

$$C_C(v) = \frac{n - 1}{\sum_{u \neq v} d(v, u)}$$

Where $d(v, u)$ is the shortest path distance from $v$ to $u$

### Interpretation

- High closeness → node can **reach all others quickly**
- In social media: users who can spread information efficiently
- Problem: undefined for disconnected graphs (infinite distances)
  - Solution: use **harmonic centrality**: $C_H(v) = \sum_{u \neq v} \frac{1}{d(v, u)}$

### Numerical Example (Closeness Centrality)

Using our 5-node example graph. Calculate the sum of shortest distances from each node to all others.

1. **Calculate sum of distances $\sum d(v,u)$:**
   - **For A**: $d(A,B)=1, d(A,C)=1, d(A,D)=2, d(A,E)=3$. Sum $= 1 + 1 + 2 + 3 = 7$.
   - **For B**: $d(B,A)=1, d(B,C)=1, d(B,D)=2, d(B,E)=3$. Sum $= 1 + 1 + 2 + 3 = 7$.
   - **For C**: $d(C,A)=1, d(C,B)=1, d(C,D)=1, d(C,E)=2$. Sum $= 1 + 1 + 1 + 2 = 5$.
   - **For D**: $d(D,A)=2, d(D,B)=2, d(D,C)=1, d(D,E)=1$. Sum $= 2 + 2 + 1 + 1 = 6$.
   - **For E**: $d(E,A)=3, d(E,B)=3, d(E,C)=2, d(E,D)=1$. Sum $= 3 + 3 + 2 + 1 = 9$.

2. **Calculate Closeness $C_C(v) = \frac{n - 1}{\sum d(v,u)}$:**
   - $C_C(A) = 4 / 7 \approx 0.571$
   - $C_C(B) = 4 / 7 \approx 0.571$
   - $C_C(C) = 4 / 5 = 0.8$ (**Most central**)
   - $C_C(D) = 4 / 6 \approx 0.667$
   - $C_C(E) = 4 / 9 \approx 0.444$

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

### Numerical Example (PageRank - 1 Iteration)

Consider a simple **directed** network with 3 pages to illustrate computation:
- $A \rightarrow B$
- $A \rightarrow C$
- $B \rightarrow C$
- $C \rightarrow A$

Let's do one iteration with a damping factor $d = 0.85$.
- Initial PageRank $\text{PR}_0(v) = \frac{1}{n} = \frac{1}{3} \approx 0.333$ for all.
- Out-degrees: $d_{out}(A) = 2$, $d_{out}(B) = 1$, $d_{out}(C) = 1$.

**Iteration 1 Calculation:**
Formula: $\text{PR}_1(v) = \frac{1 - 0.85}{3} + 0.85 \sum \frac{\text{PR}_0(u)}{d_{out}(u)}$
Base value $= \frac{0.15}{3} = 0.05$.

1. **PR(A)**: Gets a link only from C.
   $\text{PR}_1(A) = 0.05 + 0.85 \times \left( \frac{\text{PR}_0(C)}{d_{out}(C)} \right) = 0.05 + 0.85 \times \left( \frac{0.333}{1} \right) = 0.05 + 0.283 = 0.333$

2. **PR(B)**: Gets a link only from A.
   $\text{PR}_1(B) = 0.05 + 0.85 \times \left( \frac{\text{PR}_0(A)}{d_{out}(A)} \right) = 0.05 + 0.85 \times \left( \frac{0.333}{2} \right) = 0.05 + 0.141 = 0.191$

3. **PR(C)**: Gets links from both A and B.
   $\text{PR}_1(C) = 0.05 + 0.85 \times \left( \frac{\text{PR}_0(A)}{d_{out}(A)} + \frac{\text{PR}_0(B)}{d_{out}(B)} \right) = 0.05 + 0.85 \times (0.166 + 0.333) = 0.05 + 0.424 = 0.474$

Page C has the highest PageRank after 1 iteration because it receives links from multiple sources.

### Application in Social Media

- Ranking users by influence (Twitter, citation networks)
- Content recommendation
- Identifying authoritative sources

---

## 5.6 Other Network Measures

### Network-Level Measures

| Measure | Formula | Description |
|---------|---------|-------------|
| **Average degree** | $\langle k \rangle = \frac{2\|E\|}{\|V\|}$ | Mean connections per node |
| **Average path length** | $\langle l \rangle = \frac{1}{n(n-1)} \sum_{i \neq j} d(i,j)$ | Mean shortest path |
| **Diameter** | $\max_{i,j} d(i,j)$ | Longest shortest path |
| **Density** | $\frac{2\|E\|}{\|V\|(\|V\|-1)}$ | Fraction of possible edges |
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

