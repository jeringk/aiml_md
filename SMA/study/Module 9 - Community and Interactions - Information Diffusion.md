# Module 9: Information Diffusion

## Lecture Topics Covered
- Information diffusion
- Link prediction (Recap)
- Decision-Based Cascade Models
- Probabilistic Cascade Model

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

**Numerical Example (IC Model):**
- **Graph:** Node A is connected to Node B ($p_{AB} = 0.6$) and Node C ($p_{AC} = 0.3$). Node B is connected to C ($p_{BC} = 0.5$).
- **Step 0:** Start with seed set $S = \{A\}$. Active: `{A}`.
- **Step 1:** Node A (newly active) tries to activate its neighbors B and C.
  - A tries to activate B: coin flip with 60% success. Suppose it succeeds. B becomes active.
  - A tries to activate C: coin flip with 30% success. Suppose it fails. C remains inactive.
- **Step 2:** Node B (newly active) tries to activate its inactive neighbors (only C).
  - B tries to activate C with 50%. Suppose it succeeds. C becomes active.
- **Step 3:** Node C (newly active) has no inactive neighbors. Process ends. Cascade size = 3.

#### Linear Threshold (LT) Model

1. Each node $v$ has a **threshold** $\theta_v$ (sampled uniformly from $[0, 1]$)
2. Each edge has an influence weight $w_{uv}$ with $\sum_{u \in N(v)} w_{uv} \leq 1$
3. Node $v$ activates when the total influence from active neighbors exceeds its threshold:

$$\sum_{u \in N_{\text{active}}(v)} w_{uv} \geq \theta_v$$

- **Cumulative influence**: all active neighbors contribute

**Numerical Example (LT Model):**
- **Graph:** Node C is influenced by Node A ($w_{AC} = 0.4$) and Node B ($w_{BC} = 0.5$).
- **Initial State:** Node C's threshold $\theta_C$ is randomly set to $0.6$. Nodes A, B, C are inactive.
- **Step 1:** Node A gets activated (e.g., from an external campaign).
  - C checks its active neighbors: only A.
  - Cumulative influence = $w_{AC} = 0.4$.
  - $0.4 < 0.6$ ($\theta_C$). Node C stays inactive.
- **Step 2:** Node B gets activated later.
  - C checks active neighbors: A and B.
  - Cumulative influence = $w_{AC} + w_{BC} = 0.4 + 0.5 = 0.9$.
  - $0.9 \geq 0.6$ ($\theta_C$). Node C now becomes active!

### Properties of Diffusion

- **Cascade size**: number of nodes eventually activated
- **Cascade depth**: longest chain of activations
- **Cascade speed**: how quickly activation spreads
- **Virality**: typically measured by cascade size and structure

### Influence Maximization

- **Problem**: find a seed set $S$ of size $k$ that maximizes the expected number of activated nodes

$$S^* = \arg\max_{\|S\| = k} \sigma(S)$$

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

<div style="page-break-after: always;"></div>

---

## 9.3 Lecture 8: Link Prediction & Evaluation

### 9.3.1 Part 1: The Link Prediction Problem
Link prediction determines the likelihood that two nodes, currently unconnected, will form a link.

#### Missing vs. Future Link Prediction
* **Missing Link Prediction:** Assumes the graph is static but incomplete. The goal is to recover hidden or unobserved links (e.g., two people are friends in reality, but it's missing in the data).
* **Future Link Prediction:** Assumes a dynamic, evolving graph. The goal is to forecast edges that will form in the future based on temporal patterns.

#### The Big Challenge: Sparsity & Imbalance
* Real-world graphs are incredibly sparse. The number of actual edges is roughly equal to the number of nodes: $O(E)=O(V)$.
* In a 100-node network, the maximum possible edges are $\binom{100}{2}=4950$. 
* If only 100 edges exist (Positive Samples), there are 4850 non-existing edges (Negative Samples). This creates an **extreme class imbalance**.

---

### 9.3.2 Part 2: Evaluation Metrics (Confusion Matrix)
To evaluate predictions, the problem is converted into a binary classification task. 
* **True Positive (TP):** Predicted a link, and it actually formed.
* **True Negative (TN):** Predicted no link, and it did not form.
* **False Positive (FP):** Predicted a link, but it did *not* form.
* **False Negative (FN):** Predicted no link, but it *did* form.

#### Core Evaluation Formulas
* **Accuracy (ACC):** Ratio of correct predictions to total predictions.
  $$ACC=\frac{TP+TN}{TP+TN+FP+FN}$$
  * *Warning:* Accuracy is misleading in sparse graphs! A model that just predicts "no links will ever form" on a 100-node graph with 1 positive edge will still achieve $90\%$ accuracy ($TN=9$), completely missing its goal.
* **Precision (P):** Out of all predicted positive links, how many were actually positive?
  $$P=\frac{TP}{TP+FP}$$
* **Recall (R):** Out of all actual positive links, how many were correctly predicted?
  $$R=\frac{TP}{TP+FN}$$
* **True Negative Rate (TNR / Specificity):** $$TNR=\frac{TN}{TN+FP}$$
* **False Positive Rate (FPR / Fall-out):** $$FPR=\frac{FP}{FP+TN}$$

#### AUC-ROC Curve
* **AUC-ROC** maps the area under the plot comparing True Positives with False Positives.
* The score lies in the range of $[0, 1]$ and determines how strong the model is compared to a random baseline.

---

### 9.3.3 Part 3: Master Formula Sheet - Similarity Metrics

#### Local Heuristics
These calculate likelihood based on overlapping neighbors. Let $\Gamma(x)$ be the neighborhood set of node $x$, and $k_x$ be the degree of node $x$.

* **Common Neighborhood (CN):** $$S_{CN}(x,y)=|\Gamma(x)\cap\Gamma(y)|$$
* **Jaccard Similarity:** (Normalized CN)
  $$S_{J}(x,y)=\frac{|\Gamma(x)\cap\Gamma(y)|}{|\Gamma(x)\cup\Gamma(y)|}$$
* **Salton Index (Cosine Similarity):** $$S_{SI}(x,y)=\frac{|\Gamma(x)\cap\Gamma(y)|}{\sqrt{k_x\times k_y}}$$
* **Preferential Attachment (PA):** $$S_{PA}(x,y)=k_x\times k_y$$
* **Adamic Adar (AA):** (Assigns higher weights to less-connected common neighbors)
  $$S_{AA}(x,y)=\sum_{z\in\Gamma(x)\cap\Gamma(y)}\frac{1}{\log k_z}$$
* **Resource Allocation (RA):**
  $$S_{RA}(x,y)=\sum_{z\in\Gamma(x)\cap\Gamma(y)}\frac{1}{k_z}$$
* **Hub Promoted Index (HPI):** (Assigns high scores to links adjacent to hubs)
  $$S_{HPI}(x,y)=\frac{|\Gamma(x)\cap\Gamma(y)|}{\min(k_x,k_y)}$$
* **Hub Depressed Index (HDI):** (Assigns low scores to links adjacent to hubs)
  $$S_{HDI}(x,y)=\frac{|\Gamma(x)\cap\Gamma(y)|}{\max(k_x,k_y)}$$

#### Global Heuristics
Uses the entire network structure.
* **Katz Score:** ($\alpha$ is the damping factor, $A_{x,y}^p$ is number of paths of length $p$)
  $$S_{KZ}(x,y)=\sum_{p=1}^{\infty}\alpha^p\cdot A_{x,y}^p$$
* **Normalized Hitting Time:** ($\pi_y$ is PageRank stationary distribution)
  $$S_{HT}^{Norm}(x,y)=-HT_{xy}\cdot\pi_y$$

#### Probabilistic Models (Dendrograms / Hierarchical)
* **Optimal Probability** for an internal node $r$:
  $$p_r^*=\frac{E_r}{L_r R_r}$$ 
  *(Where $E_r$ = edges sharing ancestor $r$; $L_r, R_r$ = leaves in left/right subtrees).*

#### Information Diffusion (Cascade Models)
* **Decision-Based Threshold:** Node adopts strategy A if the fraction of neighbors $p$ meets:
  $$p\ge\frac{b}{a+b}$$
* **Probabilistic Epidemic (Reproductive Number):** $$R_0=q\cdot d$$
  *(Where $q$ = infection probability; $d$ = number of contacts).*

---

## 9.4 Lecture 10: Link Prediction & Information Diffusion Basics

### 9.4.1 Part 1: Introduction to Link Prediction
Link prediction is the problem of predicting the existence of a link between two entities in a network. It helps predict the state of a dynamic network at a future timestamp.

#### Key Application Areas
* **Online Social Networks:** Recommending friends to connect with or pages/users to follow.
* **E-commerce:** Recommending products or services.
* **Police/Military:** Identifying hidden terrorist groups and spotting criminals in security applications.
* **Bioinformatics:** Predicting protein-protein interactions and drug-target interactions.
* **Network Reconstruction:** Removing spurious edges and predicting missing or new links.
* **Citation Networks:** Predicting missing citations and future collaborations.

---

### 9.4.2 Part 2: Link Prediction Methods

#### Local Heuristics
These methods determine the formation of a link $(x, y)$ in the near future based on local structural similarity. They rely on nodes sharing common neighbors, such as $x$ and $y$ both being connected to a node $z$. 
* **Examples:** Common Neighbourhood, Jaccard Similarity, Preferential Attachment, Adamic Adar, Salton Index, and Hub Promoted Index.

#### Global Heuristics
Global algorithms predict missing or future links using the entire structure of the network (paths, walks, or matrix operations), not just local neighborhoods. They assign a score to node pairs, where a higher score implies a higher likelihood of a link.

**A. Katz Score**
* Measures link likelihood based on all possible paths between two nodes, where short paths matter more than long ones.
* **Formula:** $S_{KZ}(x,y) = \sum_{p=1}^{\infty} \alpha^p \cdot A_{x,y}^p$.
* $A_{x,y}^p$ is the number of paths of length $p$ between $x$ and $y$, and $\alpha$ is a damping factor that reduces the impact of longer paths.
* **Prediction Steps:** Compute scores for non-connected pairs, rank them highest to lowest, and predict links for the top-$k$ pairs or pairs exceeding a threshold.

**B. Hitting Time**
* Based on a random surfing model where a surfer starts at $x$, moves randomly to a neighbor, and repeats until reaching $y$.
* The Hitting Time ($HT_{xy}$) is the expected number of steps to reach $y$ from $x$.
* **Score:** $S_{HT}(x,y) = -HT_{xy}$. (Smaller hitting time = closer proximity = higher chance of future interaction).
* **Normalized Score:** $S_{HT}^{Norm}(x,y) = -HT_{xy} \cdot \pi_y$ (where $\pi_y$ is the stationary distribution of PageRank).

**C. Commute Time**
* Extends the random walk by traveling from $x$ to $y$, and then back from $y$ to $x$.
* **Score:** $S_{CT}(x,y) = -(HT_{xy} + HT_{yx})$.
* **Normalized Score:** $S_{CT}^{Norm}(x,y) = -(HT_{xy} \cdot \pi_y + HT_{yx} \cdot \pi_x)$.

#### Local vs. Global Heuristics Comparison
* **Local Methods:** Use neighbor-level information over short path lengths (1-2 hops). They have moderate accuracy but boast low computational cost and excellent scalability.
* **Global Methods:** Use the whole network and all paths. They have higher accuracy but high computational cost and limited scalability.

#### Probabilistic Link Prediction Methods
Unlike heuristic models, probabilistic methods model link existence as a probability and assume a generative process using statistical or Bayesian models.

**Hierarchical Networks (Dendrograms)**
* A network is hierarchical if vertices can be divided into groups, and those groups subdivided, in a logical order.
* Represented as a tree (dendrogram $D$) where the $n$ nodes of the network form the leaves, and $n-1$ internal nodes represent groups.
* Each internal node $r$ has an association probability $p_r$ determining how likely two groups are to connect given $r$ as their least common ancestor.
* **Likelihood Formula:** $\mathcal{L}(D,p_r) = \prod_{r \in D} p_r^{E_r} (1-p_r)^{L_r R_r - E_r}$.
* **Optimal Probability:** $p_r^* = \frac{E_r}{L_r R_r}$.

---

### 9.4.3 Part 3: Basics of Information Diffusion

#### Diffusion Fundamentals
* **Definition:** The process by which information is spread from one place to another through interactions.
* **Core Elements:** Sender, Receiver, and the Medium (channel).

#### Real-World Cascade Behaviors
* **Socio-political Cascades:** Arab Spring Movement, fall of the Berlin Wall, #MeToo movement.
* **Financial/Market:** Market bubbles (stocks becoming overly popular), viral marketing.
* **Other:** Healthcare disease propagation, rumor/belief spread, fake news virality.

#### Core Diffusion Phenomena
* **Information Cascade:** A phenomenon where a number of people make the same decision in a sequential fashion. Found in behavioral economics and network theory.
* **Herd Behavior:** Individuals in a group acting collectively without centralized direction (e.g., riots, strikes, religious gatherings). It is a useful marketing tool but can also turn violent.
* **Diffusion of Innovations:** The success or failure of an innovation depends on the network structure of initial adopters (e.g., farmers adopting hybrid corn or doctors adopting new drugs based on peer influence).
* **Echo Chambers:** Situations where beliefs are amplified or reinforced inside a closed system through communication and repetition. This creates fragmented communities with tunnel vision, reinforcing fake news due to unchallenged peer trust.
* **Epidemics:** Rapid spread of disease (like the 2014 West Africa Ebola Epidemic) to a large population. Epidemic models are similar to innovation models, except individuals do not *decide* to become infected.

---

## 9.5 Lecture 11: Link Prediction Recap & Information Diffusion

### 9.5.1 Part 1: Link Prediction (Recap)
Link prediction techniques are used to estimate the likelihood of a future connection between two nodes.

#### Local Heuristics
* **Concept:** These methods use local neighborhood information and assume that nodes sharing common neighbors are more likely to connect.
* **Examples:** Common Neighbors, Jaccard Coefficient, Adamic-Adar.
* **Pros/Cons:** Fast and scalable, but they ignore the global structure of the network.

#### Global Heuristics
* **Concept:** These methods use the entire network structure, considering all possible paths between nodes.
* **Examples:** Katz Index, Hitting time, Commute time.
* **Pros/Cons:** More accurate than local heuristics, but computationally expensive.

#### Probabilistic Methods
* **Concept:** Represents the network using a hierarchical structure unfolded as a dendrogram.
* **Structure:** The leaves of the dendrogram represent the network nodes, while internal nodes represent groups or communities.
* **Link Probability Estimation:** Each internal node $r$ is assigned a probability $p_r$. To find the probability of a link between nodes $u$ and $v$, find their Lowest Common Ancestor (LCA) in the dendrogram. The probability is $P(u,v) = p_{LCA(u,v)}$.

---

### 9.5.2 Part 2: Information Diffusion Terminologies

* **Contagion**: An entity that spreads across a network (virus, rumor, product, idea).
* **Adoption**: The event of infection or diffusion, also known as activation or conversion.
* **Adopters**: The final set of infected nodes.
* **Cascade**: The final propagation tree obtained by the spread of the infection.

---

### 9.5.3 Part 3: Decision-Based Cascade Models

Originated from local interaction models (Morris, 2000), these models assume nodes have the freedom to decide whether to adopt a contagion driven by a direct benefit or payoff.

#### 1. Two-Player Coordination Game (Single Choice)
The payoff of adopting a contagion is proportional to the number of neighbors that have adopted it. Players aim to coordinate on the same strategy to maximize payoffs.
* **Payoff Matrix**:
  * $u$ chooses A, $v$ chooses A $\rightarrow$ Payoff: $a$
  * $u$ chooses B, $v$ chooses B $\rightarrow$ Payoff: $b$
  * Mismatched choices (A-B or B-A) $\rightarrow$ Payoff: $0$

**The Adoption Threshold:**
If node $u$ has $d$ neighbors, and a fraction $p$ of those neighbors adopt strategy A:
* Total payoff for strategy A: $a \cdot d \cdot p$
* Total payoff for strategy B: $b \cdot d \cdot (1 - p)$
* **Decision Rule:** Node $u$ adopts strategy A if total payoff of A is $\ge$ total payoff of B, which simplifies to $p \ge \frac{b}{a+b}$.

**Cascade Formation:**
1. A seed set initially adopts A.
2. Neighbor nodes that satisfy the threshold adopt A.
3. New adopters trigger further adoptions until equilibrium.

#### 2. Multiple Choice Decision Model
Allows a node to adopt more than one strategy/behavior (AB), incurring an additional cost $c$.
* **Revised Payoff Matrix**:
  * $u$ (AB), $v$ (A) $\rightarrow$ Payoff: $a$
  * $u$ (AB), $v$ (B) $\rightarrow$ Payoff: $b$
  * $u$ (AB), $v$ (AB) $\rightarrow$ Payoff: $\max(a, b)$

**Infinite Chain Networks: Case Studies**
* **Single Choice Example ($a=3, b=2$)**: Node $u$ switches to A because payoff (3) > sticking with B (2). The cascade continues.
* **Multiple Choice - Case I ($a=3, b=2, c=1$)**: Node $u$ adopts AB because payoff ($3+2-1=4$) > A (3) > B (2). System becomes stable (no further cascade).
* **Multiple Choice - Case II ($a=5, b=3, c=1$)**: Node $u$ adopts AB (payoff $5+3-1=7$). Node $v$ also adopts AB. Cascade continues.

#### 3. Generic Model Analysis
For an infinite chain network with strategy set {A, B, AB}, analyzing a node $u$ between specific neighbor combinations:

**Case A: Node $u$ is between an 'A' node and a 'B' node**
* Payoffs: A = $a$; B = $1$ (assuming match with B yields 1); AB = $a+1-c$.
* Breakpoint Equations:
  * **B vs. A**: Prefer B if $a<1$; Prefer A if $a>1$.
  * **AB vs. B**: Prefer B if $a<c$; Prefer AB if $a>c$.
  * **A vs. AB**: Prefer AB if $c<1$; Prefer A if $c>1$.

**Case B: Node $u$ is between an 'AB' node and a 'B' node**
* Payoffs: A = $a$; B = $2$ (assuming matches with AB and B yield $1+1=2$); AB = $a+1-c$ (assuming $\max(a,1)=a$).
* Breakpoint Equations:
  * **B vs. A**: Prefer B if $a<2$; Prefer A if $a>2$.
  * **AB vs. B**: Prefer B if $a-c<1$; Prefer AB if $a-c>1$.
  * **A vs. AB**: Prefer AB if $c<1$; Prefer A if $c>1$.

---

### 9.5.4 Probabilistic Cascade Model

Instead of deterministic payoff choice, infected nodes transmit contagion with probability $q$.

This model is more suitable for uncertainty-driven spread (for example, viruses or random forwarding behavior).

#### Random Tree Assumptions

- Root node is infected
- Each node has $d$ children
- Infection from a parent-side source occurs with probability $q$

Let $p_h$ be the probability that a node at level $h$ is infected.

Recurrence:

$$
p_h = 1 - (1 - q \cdot p_{h-1})^d
$$

where:
- $p_h$: infection probability at tree level $h$
- $p_{h-1}$: infection probability at previous level
- $q$: transmission probability per parent-side incoming contact
- $d$: branching factor (number of children)

Equivalent functional form:

$$
f(x) = 1 - (1 - qx)^d
$$

where:
- $x$: input infection probability from previous level
- $f(x)$: next-level infection probability

Cascade survival condition:

$$
\lim_{h \to \infty} p_h > 0
$$

where:
- positive limit implies persistent cascade

Cascade extinction condition:

$$
\lim_{h \to \infty} p_h = 0
$$

where:
- zero limit implies dying cascade

---

### 9.5.5 Comparison of Models

| Feature | Decision-Based Model | Probabilistic Model |
| ------- | -------------------- | ------------------- |
| Type | Deterministic | Stochastic |
| Core driver | Payoff optimization | Random transmission |
| Theory base | Game theory | Probability theory |
| Human strategic choice | Explicitly modeled | Implicit/partial |
| Viral disease-like spread | Weak fit | Strong fit |
| Real-world uncertainty | Limited | Better represented |

---

### 9.5.6 Limitations of Decision-Based Model

- Requires known payoff values, often unavailable in real settings
- Assumes fully rational decisions
- Does not naturally encode randomness
- Not ideal for biological/viral spread processes

---

### 9.5.7 Exam-Focused Formula Sheet

Adoption threshold:

$$
p \ge \frac{b}{a+b}
$$

where:
- $p$: fraction of neighbors using $A$
- $a$: payoff for $(A, A)$ match
- $b$: payoff for $(B, B)$ match

Probabilistic recurrence:

$$
p_h = 1 - (1 - q p_{h-1})^d
$$

where:
- $p_h$: infection probability at level $h$
- $q$: per-contact transmission probability
- $d$: branching factor

Update function:

$$
f(x) = 1 - (1 - qx)^d
$$

where:
- $x$: previous-level infection probability
- $f(x)$: next-level infection probability

---

### 9.5.8 Intuition Summary

- Decision model: spread is strategy-driven (benefit/utility threshold)
- Probabilistic model: spread is chance-driven (transmission uncertainty)
- Use decision models for strategic adoption; use probabilistic models for viral-like processes

---

## Key Takeaways

- Data mining provides the foundational techniques for analyzing social media at scale
- Information diffusion models (IC, LT) capture how content and influence propagate
- Influence maximization is a key optimization problem — greedy approaches provide good approximations
- Epidemiological models (SI, SIS, SIR) offer analogies for understanding viral spreading

