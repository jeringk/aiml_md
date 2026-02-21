# Module 1 — Embeddings, Vector Search & Hybrid Retrieval

## Topics

- [[#2.1 Semantic vs Keyword Search|Semantic vs Keyword Search]]
- [[#2.2 Vector Database Architecture (HNSW, ANN)|Vector Database Architecture (HNSW, ANN)]]
- [[#2.3 BM25 + Dense Retrieval + RRF (Hybrid Search)|BM25 + Dense Retrieval + RRF (Hybrid Search)]]

---

## 2.1 Semantic vs Keyword Search

**Keyword Search (Lexical Search):**
- **Mechanism:** Looks for exact or partial word matches (e.g., matching "run" with "running" using stemming). Uses algorithms like **TF-IDF** or **BM25**.
- **Pros:** Excellent for exact matching, IDs, specific product names, or domain-specific jargon. Computationally very cheap and highly predictable.
- **Cons:** Fails at understanding synonymy (e.g., "car" vs "automobile") and polysemy (words with multiple meanings, e.g., "bank" the financial institution vs "bank" of a river). Doesn't understand the semantic *meaning* of the query.

**Semantic Search (Dense Retrieval):**
- **Mechanism:** Encodes both the query and the documents into dense, high-dimensional vectors (embeddings) using a neural network (e.g., BERT-based encoder). Searches for nearest neighbors in the vector space using distances like Cosine Similarity or Dot Product.
- **Pros:** Capable of understanding user intent, context, and synonyms. Can match queries and documents that share no common keywords but share the same underlying meaning.
- **Cons:** Struggles with out-of-vocabulary terms, specific arbitrary IDs, or exact acronyms. Computationally expensive (requires a forward pass through a neural net to embed the query in real-time).

---

## 2.2 Vector Database Architecture (HNSW, ANN)

Vector databases (e.g., Pinecone, Milvus, Weaviate, Qdrant) store and query high-dimensional data efficiently.

**Approximate Nearest Neighbor (ANN):**
- Exact nearest neighbor ($k$-NN) requires calculating the distance to *every* document in the database (Time complexity $O(N)$), which is too slow for large-scale enterprise deployments.
- ANN algorithms trade a small amount of accuracy for a massive, exponential gain in speed. They return the *approximate* closest vectors.

**HNSW (Hierarchical Navigable Small World):**
- The state-of-the-art graph-based algorithm for ANN search.
- **Structure:** It builds a multi-layered proximity graph. The bottom layer contains all nodes (vectors). Each layer above it contains a progressively smaller, exponentially decaying subset of the nodes in the layer below.
- **Search Process:**
  1. Start at the top layer (fewest nodes). Find the greedy nearest neighbor to the query.
  2. Use that node as the entry point for the next layer down.
  3. Explore local neighbors in the new layer. Find the closest.
  4. Repeat until reaching the bottom layer.
- **Complexity:** Search time is roughly $O(\log N)$.

---

## 2.3 BM25 + Dense Retrieval + RRF (Hybrid Search)

**Hybrid Search** combines the precision of Keyword Search (BM25) with the contextual understanding of Semantic Search (Dense Retrieval) to fetch the best possible context for Advanced RAG.

**BM25 (Best Matching 25):**
An advanced variation of TF-IDF. It scores a document $D$ for a query $Q$ with terms $q_1, ..., q_n$:
$$ \text{Score}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})} $$
- $f(q_i, D)$: Term frequency of word in doc.
- $k_1$: Term frequency saturation parameter.
- $b$: Length normalization parameter ($|D|$ is doc length, avgdl is average doc length).

**Reciprocal Rank Fusion (RRF):**
Since BM25 generates scores on a completely different scale (unbounded) than Cosine Similarity (bounds of -1 to 1 or 0 to 1), you cannot simply add their scores together. **RRF** combines the results by looking *only* at their ordinal rank.
$$ \text{RRF\_Score}(d) = \frac{1}{k + p_{BM25}(d)} + \frac{1}{k + p_{Dense}(d)} $$
- $p(d)$ is the position (rank) of document $d$ in the respectively retrieved list.
- $k$ is a smoothing constant, usually set to 60.

**Mathematical Example of RRF:**
- Document A is ranked #1 in BM25 and #4 in Dense.
- Document B is ranked #3 in BM25 and #2 in Dense.
- Assuming $k=60$:
  - $\text{RRF}(A) = 1/(60+1) + 1/(60+4) = 1/61 + 1/64 = 0.01639 + 0.01562 = 0.0320$
  - $\text{RRF}(B) = 1/(60+3) + 1/(60+2) = 1/63 + 1/62 = 0.01587 + 0.01612 = 0.0319$
(A and B have extremely similar RRF scores, making it mathematically stable to merge signals).

---

## References

- "Dense Passage Retrieval" (Karpukhin et al., 2020)
