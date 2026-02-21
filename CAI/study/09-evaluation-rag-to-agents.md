# Module 3 — Evaluation Metrics & Performance

## Topics

- [[#9.1 Calculation of RAG & Agent Metrics|Calculation of RAG & Agent Metrics (MRR, MAP, NDCG)]]
- [[#9.2 Strategic Evaluation (LLM-as-a-Judge)|Strategic Evaluation (LLM-as-a-Judge)]]
- [[#9.3 Benchmarks|Benchmarks]]

---

## 9.1 Calculation of RAG & Agent Metrics

To measure retrieval performance in generative systems (like RAG), mathematical metrics calculating how well the ranking algorithm places relevant documents are used.

### Unranked Metrics (Threshold-based)
1. **Precision@K:** Proportion of retrieved relevant items in the top K results.
   - Example: Top 5 contains 3 relevant docs. Precision@5 = 3/5 = 0.6.
2. **Recall@K:** Proportion of total relevant items retrieved in the top K.
   - Example: There are 10 relevant docs in the entire database. Top 5 contains 3. Recall@5 = 3/10 = 0.3.

### Ranked Metrics (Position-aware)

1. **Mean Reciprocal Rank (MRR)**
   Calculates how far down the user has to scroll to find the **first** relevant document.
   $$ \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} $$
   - Where $\text{rank}_i$ is the position of the *first relevant doc* for Query $i$.
   - **Example:** For Q1, first relevant is at rank 3 (1/3). For Q2, it's at rank 1 (1/1). MRR = $(0.33 + 1) / 2 = 0.66$.

2. **Mean Average Precision (MAP)**
   Measures precision across multiple relevance levels, considering precision at every point a relevant document is found.
   - First calculate Average Precision (AP) for a specific query: Average of the Precision@k values solely at the positions where a relevant document exists.
   - MAP is simply the mean of AP over all measured queries.

3. **Normalized Discounted Cumulative Gain (NDCG)**
   Evaluates graded relevance (e.g., 0=Irrelevant, 1=Partial, 2=Highly Relevant) accounting for rank.
   - **Discounted Cumulative Gain (DCG):** $\sum_{i=1}^{p} \frac{rel_i}{\log_2(i + 1)}$
   - **Ideal DCG (IDCG):** Calculate DCG of the exact same documents, but sorted by descending true $rel_i$ (perfect scenario).
   - **NDCG:** $\text{DCG} / \text{IDCG}$
   - Crucial concept for CAI: Placing a "highly relevant" chunk at rank 1 is exponentially better than rank 5 (penalized by log denominator) to avoid the "Lost in the Middle" LLM syndrome.

---

## 9.2 Strategic Evaluation (LLM-as-a-Judge)

Evaluating the *generative* part of an Agent's actions is remarkably difficult because text generation is open-ended (there is no exact string match baseline).

**LLM-as-a-Judge Pattern:**
- Use a highly capable frontier model (e.g., GPT-4o, Claude 3.5 Sonnet) to evaluate the output of the target pipeline.
- Provide the Judge with a rigid rubric (e.g., "Score from 1-5 for conciseness, groundedness, and lack of hallucination").
- **Known Limitations & Biases:**
  - **Position bias:** Tends to arbitrarily prefer the first answer provided in a pairwise comparison context.
  - **Verbosity bias:** LLMs often rate longer, more verbose answers higher, implicitly equating length with quality.
  - **Self-enhancement bias:** Models inherently prefer answers written in a style similar to their own native training distribution.

**RAG Assessment Triad (e.g., Trulens, RAGAS frameworks):**
1. **Context Relevance:** Does the retrieved context actually contain data necessary to answer the query?
2. **Groundedness / Faithfulness:** Is the generated answer completely supported by the retrieved context (no hallucinations)?
3. **Answer Relevance:** Does the generated answer directly address the final user query?

---

## 9.3 Benchmarks

- **MT-Bench:** A standard multi-turn benchmark designed to evaluate an LLM's capacity to engage in coherent conversation and adhere to constraints over multiple dialogue turns.
- **GAIA:** A benchmark specifically for General AI Assistants focusing on fundamentally difficult reasoning, tool-use, and multi-step planning tasks that are trivial for humans but stump modern LLMs.

---

## References

- "Judging LLM-as-a-Judge with MT-Bench" (2023)
- GAIA Benchmark
