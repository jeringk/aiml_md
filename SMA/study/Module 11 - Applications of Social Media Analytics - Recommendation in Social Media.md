# Module 11: Recommendation in Social Media

## Lecture Topics Covered
- Link Prediction Problem
- Types of Link Prediction Problems
- Local Heuristics for Link Prediction
- Global Heuristics for Link Prediction (Katz Score)
- Probabilistic Link Prediction

---
## 11.1 Classical Recommendation Algorithms

- **Recommendation systems**: predict user preferences and suggest relevant items
- Foundation for social media feeds, friend suggestions, content recommendations

### Content-Based Filtering

- Recommend items **similar** to what the user previously liked
- Build a **user profile** from features of liked items
- Similarity measures: cosine similarity, TF-IDF vectors

| Advantages | Limitations |
|------------|-------------|
| No cold-start for items (features available) | Limited to user's existing interests |
| Transparent recommendations | Cannot capture complex preferences |
| No need for other users' data | Feature engineering required |

### Collaborative Filtering (CF)

- Recommend items based on **similar users' preferences** — "users who liked X also liked Y"

#### User-Based CF

1. Find users similar to target user (cosine similarity, Pearson correlation on ratings)
2. Predict rating as weighted average of similar users' ratings:

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} \|\text{sim}(u, v)\|}$$

**Numerical Example (User-Based CF):**
- **Goal:** Predict User A's rating for Movie X ($\hat{r}_{AX}$).
- **Data:**
  - User A's average rating $\bar{r}_A = 3.0$.
  - Similar User B: $\text{sim}(A,B) = 0.8$, $\bar{r}_B = 4.0$, rating $r_{BX} = 5.0$.
  - Similar User C: $\text{sim}(A,C) = 0.5$, $\bar{r}_C = 2.5$, rating $r_{CX} = 2.0$.
- **Calculation:**
  - Numerator: $0.8 \times (5.0 - 4.0) + 0.5 \times (2.0 - 2.5) = 0.8(1.0) + 0.5(-0.5) = 0.8 - 0.25 = 0.55$.
  - Denominator: $|0.8| + |0.5| = 1.3$.
  - Prediction: $\hat{r}_{AX} = 3.0 + (0.55 / 1.3) = 3.0 + 0.423 = \mathbf{3.423}$.

#### Item-Based CF

1. Find items similar to the target item (based on co-rating patterns)
2. Predict rating from ratings of similar items

#### Matrix Factorization

- Decompose the user-item rating matrix $R \approx U \cdot V^T$
- $U \in \mathbb{R}^{m \times k}$: user latent factors
- $V \in \mathbb{R}^{n \times k}$: item latent factors
- Optimize via SGD or ALS:

$$\min_{U, V} \sum_{(u,i) \in \text{observed}} (r_{ui} - u_u^T v_i)^2 + \lambda(\|U\|^2 + \|V\|^2)$$

- **SVD++**, **ALS**, **NMF** are popular variants

### Hybrid Approaches

- Combine content-based and collaborative filtering
- Mitigate limitations of each approach
- Methods: weighted, switching, cascading, feature combination

---

## 11.2 Recommendation Using Social Context

- **Key insight**: social networks provide additional signals for recommendation
- Trust, friendship, and social influence improve predictions

### Trust-Based Recommendation

- Users trust recommendations from friends more than strangers
- Use the **trust network** (who trusts whom) to weight recommendations
- Propagate trust: if A trusts B and B trusts C, then A may partially trust C

### Social Regularization

- Add a social constraint to matrix factorization:

$$\min_{U,V} \sum_{(u,i)} (r_{ui} - u_u^T v_i)^2 + \lambda_1 \|U\|^2 + \lambda_2 \|V\|^2 + \lambda_3 \sum_{(u,v) \in E} \|u_u - u_v\|^2$$

- The last term encourages **friends to have similar latent representations**

### Social Context Features

| Signal | How It Helps |
|--------|-------------|
| **Friendships** | Friends have similar tastes (homophily) |
| **Trust** | Trusted users' opinions are weighted higher |
| **Groups** | Group membership indicates shared interests |
| **Interactions** | Likes, comments, shares indicate interest |
| **Influence** | Opinion leaders shape item popularity |

### Friend Recommendation (Link Prediction)

- Predict who should connect based on:
  - **Common neighbors**: $\frac{\|N(u) \cap N(v)\|}{\|N(u) \cup N(v)\|}$ (Jaccard)
  - **Adamic-Adar**: $\sum_{w \in N(u) \cap N(v)} \frac{1}{\log \|N(w)\|}$
  - **Preferential attachment**: $\|N(u)\| \cdot \|N(v)\|$
  - **Graph embeddings**: node2vec, GCN-based link prediction

---

## 11.3 Evaluating Recommendation

### Rating Prediction Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | $\frac{1}{n} \sum \|r_{ui} - \hat{r}_{ui}\|$ | Mean absolute error |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (r_{ui} - \hat{r}_{ui})^2}$ | Root mean squared error |

### Ranking Metrics

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K** | Fraction of relevant items in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain — considers rank position |
| **MAP** | Mean Average Precision across queries |
| **MRR** | Mean Reciprocal Rank of first relevant item |

### Beyond Accuracy

| Dimension | Description |
|-----------|-------------|
| **Diversity** | How diverse are the recommended items? |
| **Novelty** | Does the system recommend items the user didn't know about? |
| **Serendipity** | Are recommendations surprisingly relevant? |
| **Coverage** | What fraction of items/users does the system cover? |
| **Fairness** | Are recommendations unbiased across demographic groups? |

### Evaluation Protocols

- **Offline**: train/test split on historical data
- **Online**: A/B testing with real users
- **User studies**: qualitative assessment of recommendation quality

---

## Key Takeaways

- Classical methods (content-based, collaborative filtering, matrix factorization) form the foundation
- Social context (friendship, trust, influence) significantly improves recommendations in social media
- Social regularization adds network constraints to matrix factorization
- Evaluation should go beyond accuracy — diversity, novelty, and fairness matter
- Link prediction is a special case of recommendation in social networks

