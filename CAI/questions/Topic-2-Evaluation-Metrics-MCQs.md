# Topic 2: Evaluation Metrics & Performance - In-Depth MCQs

## Question 1 (Ranked Metrics Calculation)

An engineer is evaluating the retrieval performance of a new hybrid search algorithm across 3 test queries using the **Mean Reciprocal Rank (MRR)** metric. The ranks of the *first* relevant document returned for each query are as follows:
- Query 1: Rank 2
- Query 2: Rank 1
- Query 3: Rank 4

What is the exact calculated MRR for this system?

A) $2.33$
B) $0.583$
C) $0.75$
D) $0.428$

<div style="page-break-after: always"></div>

### Topics to know to answer Question 1
- [[../study/Module 9 - Evaluation - RAG to Agents.md#9.1 Calculation of RAG & Agent Metrics|Calculation of RAG & Agent Metrics]]

<div style="page-break-after: always"></div>

### Solution 1

**Correct Answer: B**

**Explanation:**
**Mean Reciprocal Rank (MRR)** measures how far down a user has to scroll to hit the *very first* relevant result. It completely ignores any subsequent relevant documents. The formula is the average of the reciprocals of the first relevant ranks:
$$ \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} $$
1. Calculate reciprocal for Query 1: $1 / 2 = 0.50$
2. Calculate reciprocal for Query 2: $1 / 1 = 1.00$
3. Calculate reciprocal for Query 3: $1 / 4 = 0.25$
4. Sum the reciprocals: $0.50 + 1.00 + 0.25 = 1.75$
5. Divide by total number of queries (3): $1.75 / 3 = 0.5833...$

---

## Question 2 (Strategic Evaluation Biases)

You have implemented an **LLM-as-a-Judge** pipeline using GPT-4 to automatically evaluate pairwise responses from your custom customer support model against a baseline model. You notice a systemic issue where GPT-4 heavily favors answers that are three paragraphs long, even if they contain less factual information than concise, one-paragraph baseline answers. 

Which known limitation of the LLM-as-a-Judge pattern is occurring here?

A) Position Bias
B) Self-Enhancement Bias
C) Groundedness Hallucination
D) Verbosity Bias

<div style="page-break-after: always"></div>

### Topics to know to answer Question 2
- [[../study/Module 9 - Evaluation - RAG to Agents.md#9.2 Strategic Evaluation (LLM-as-a-Judge)|Strategic Evaluation (LLM-as-a-Judge)]]

<div style="page-break-after: always"></div>

### Solution 2

**Correct Answer: D**

**Explanation:**
**Verbosity Bias** is a widely documented flaw in advanced Frontier models acting as judges (such as GPT-4 in the MT-Bench benchmark). LLMs inherently associate longer, more grammatically complex, and conversational text with "high quality" and "helpfulness", causing them to arbitrarily penalize dense, concise, and highly factual answers simply because they are shorter.
- *Position Bias* occurs when the Judge automatically gives the first answer in a pairwise test (Model A vs Model B) a higher score simply because it read it first.
- *Self-Enhancement Bias* occurs when an LLM favors text that happens to match its native underlying training distribution style.

---

## Question 3 (RAG Assessment Triad)

When evaluating the generative portion of a RAG system, frameworks like RAGAS utilize a "triad" of evaluation metrics. If a user asks "What is the capital of France?", and the Vector database retrieves a document about the Eiffel Tower (which does not mention Paris), but the LLM uses its parametric memory to correctly answer "Paris", which specific metric inside the triad will strictly fail?

A) Context Relevance
B) Answer Relevance
C) Groundedness / Faithfulness
D) Generation Precision

<div style="page-break-after: always"></div>

### Topics to know to answer Question 3
- [[../study/Module 9 - Evaluation - RAG to Agents.md#9.2 Strategic Evaluation (LLM-as-a-Judge)|Strategic Evaluation (LLM-as-a-Judge)]]

<div style="page-break-after: always"></div>

### Solution 3

**Correct Answer: C**

**Explanation:**
The generative aspect of RAG is evaluated strictly to prevent the LLM from relying on its unverified parametric memory. 
- **Groundedness / Faithfulness** explicitly measures whether the generated answer can be entirely traced back to and supported *only* by the retrieved context. Because the retrieved context (the Eiffel Tower document) did not contain the word "Paris", the LLM hallucinated the answer relative to the context provided. Therefore, Groundedness fails.
- *Answer Relevance* would pass, because "Paris" perfectly answers the user's initial question.
- *Context Relevance* would likely also fail (or score very low), because a document about the Eiffel Tower is not specifically relevant to extracting the capital city.
