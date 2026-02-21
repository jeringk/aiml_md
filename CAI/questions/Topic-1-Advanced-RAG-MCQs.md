# Topic 1: Advanced RAG Architectures & Retrieval - In-Depth MCQs

## Question 1 (Chunking Strategies)

An engineering team is struggling with retrieval accuracy in their RAG pipeline because standard token-based chunking is frequently splitting sentences in half, causing the embedding model to lose the core semantic meaning. They decide to implement an approach where they embed very small, precise sentences (to guarantee accurate retrieval matches) but pass the entire surrounding paragraph to the LLM during the generation phase to provide richer context. What is this specific technique called?

A) Semantic Chunking
B) Sliding Window Chunking
C) Small-to-Big Retrieval (Parent-Child Chunking)
D) Contextual Retrieval Summarization

<div style="page-break-after: always"></div>

### Topics to know to answer Question 1
- [[../study/Module 7 - RAG - Foundations to Advanced.md#7.1 Processing & Chunking Strategies|Processing & Chunking Strategies]]

<div style="page-break-after: always"></div>

### Solution 1

**Correct Answer: C**

**Explanation:**
**Small-to-Big Retrieval (also known as Parent-Child Chunking)** involves divorcing the chunk size used for *embedding/searching* from the chunk size used for *LLM generation*. By embedding small text chunks (the child), you ensure that the Vector DB search hits precise, highly concentrated semantic queries without dilution. However, if you only pass that single sentence to the LLM, it lacks context. Therefore, the system automatically fetches the larger overarching parent block (e.g., the paragraph) and injects that into the prompt. 
- *Semantic chunking* dynamically splits by natural breaks (paragraphs/headers) rather than arbitrary algorithms.
- *Sliding window* introduces an deliberate token overlap between sequential chunks.

---

## Question 2 (Re-ranking)

In standard Bi-Encoder vector architectures, queries and documents are embedded separately and compared via Cosine Similarity, which can lack precision. A company introduces a "Two-Stage Retrieval" architecture. In stage two, they pass both the user's query and the top 100 retrieved documents simultaneously through a specialized model. What is this specialized model, and why is it mathematically superior at scoring relevance?

A) Cross-Encoder; It evaluates the deep, token-level self-attention relationships between the query terms and the document terms concurrently.
B) Sparse-Encoder; It utilizes BM25 to count exact keyword occurrences, overcoming the Bi-Encoder's inability to find exact acronyms.
C) Cross-Encoder; It generates a hypothetical answer to the query (HyDE) and embeds the answer to calculate semantic distance.
D) Generative Autoencoder; It attempts to reconstruct the original query strictly from the document tokens.

<div style="page-break-after: always"></div>

### Topics to know to answer Question 2
- [[../study/Module 7 - RAG - Foundations to Advanced.md#7.2 Re-ranking & Contextual Retrieval|Re-ranking & Contextual Retrieval]]

<div style="page-break-after: always"></div>

### Solution 2

**Correct Answer: A**

**Explanation:**
A standard Vector DB relies on **Bi-Encoders**, which encode the query and document independently into two separate vectors and measure their geometric distance (very fast but lacks deep linguistic understanding). A **Cross-Encoder** is used as a Re-ranker. It concatenates the `[Query] + [Document]` into a single sequence and processes them *together* through the Transformer's attention layers. This allows the model to map the complex, cross-contextual relationships between the specific words in the query and the text, yielding a far more accurate relevance probability score. Because this operation is computationally expensive, it is only applied as a secondary "Stage 2" ranker to the top ~100 results retrieved by the faster Bi-Encoder.

---

## Question 3 (Agentic RAG)

A user prompts a customer service agent with: *"What is the policy for remote work and how much budget do I have remaining for home office equipment?"* 
The system does not run a single blind vector search. Instead, it utilizes an LLM to actively classify the query and triggers two separate searches: one to a specialized HR Vector Database (for the policy) and one to a structured SQL database (for the budget). 

What specific Advanced RAG technique is this system demonstrating?

A) Iterative / Self-Reflective Retrieval
B) Hypothetical Document Embeddings (HyDE)
C) Reciprocal Rank Fusion (RRF)
D) Query Routing

<div style="page-break-after: always"></div>

### Topics to know to answer Question 3
- [[../study/Module 7 - RAG - Foundations to Advanced.md#7.3 Advanced RAG Techniques (Agentic RAG, Routing)|Advanced RAG Techniques (Agentic RAG, Routing)]]

<div style="page-break-after: always"></div>

### Solution 3

**Correct Answer: D**

**Explanation:**
**Query Routing** fundamentally shifts a system from static RAG into Agentic RAG. Before any search is performed, an LLM classifier actively "routes" or directs the query to the correct specialized data silo. In this complex multi-part query, the system identifies that unstructured semantic search is required for the policy (Vector DB), but strict deterministic retrieval is required for the user's remaining equipment budget (SQL DB). 
- *Iterative Retrieval* involves executing a search, drafting an answer, finding gaps, and issuing a follow-up search autonomously.
- *HyDE* handles query expansion by hallucinating an answer to improve vector distance matching.
