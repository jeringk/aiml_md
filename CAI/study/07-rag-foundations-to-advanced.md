# Module 2 — Advanced RAG Architectures & Retrieval

## Topics

- [[#7.1 Processing & Chunking Strategies|Processing & Chunking Strategies]]
- [[#7.2 Re-ranking & Contextual Retrieval|Re-ranking & Contextual Retrieval]]
- [[#7.3 Advanced RAG Techniques (Agentic, Routing)|Advanced RAG Techniques (Agentic, Routing)]]

---

## 7.1 Processing & Chunking Strategies

In standard RAG, long documents are chunked into smaller pieces (e.g., 500-1000 tokens) because of context window limitations and to ensure embedding models capture specific semantic concepts accurately without diluting the vectors.

**Advanced Chunking Strategies:**
1. **Semantic Chunking:** Instead of splitting by arbitrary character or token counts, algorithms split by natural semantic boundaries (e.g., at ends of sentences, paragraphs, or specifically leveraging Markdown headers).
2. **Small-to-Big Retrieval (Parent-Child Chunking):** Embed very small chunks (e.g., a single sentence) to ensure precise, high-accuracy retrieval. However, when returning the context to the LLM, pass the entire "Parent" paragraph or document encompassing that sentence to give the LLM full, rich context for generation.
3. **Sliding Window Chunking:** Include an overlap (e.g., 10-20% duplication) between sequential chunks to ensure critical concepts spanning chunk boundaries aren't aggressively cut in half.

---

## 7.2 Re-ranking & Contextual Retrieval

**Re-ranking (Cross-Encoders):**
- **Problem:** Fast retrieved results from Bi-Encoders (Vector DBs calculating Cosine Similarity) are not perfectly accurate because the query and the document are embedded separately and independently.
- **Solution - Two-Stage Retrieval:**
  1. Retrieve top 100 documents quickly using a standard Vector DB + BM25 (Hybrid).
  2. Pass these 100 documents $+ \text{the query}$ simultaneously through a **Cross-Encoder model** (e.g., Cohere Rerank, BGE Reranker). A cross-encoder analyzes the rich token-level attention relationships between the query and document.
  3. It outputs an exact relevance probability, re-sorting the top 100 to yield a perfect top 5 context window for the LLM.

**Contextual Retrieval:**
- As highlighted by Anthropic, when a long document is chunked, the isolated chunk loses its broader context (e.g., a chunk might just say "The revenue grew by 20%" without mentioning the company name).
- **Solution:** Before embedding the text, pass the physical chunk + the full parent document to a fast, cheap LLM. Instruct the LLM to write a 1-sentence contextual summary (e.g., "This chunk discusses Acme Corp's Q3 2023 financial results"). Append this summary to the chunk *before* embedding and storing. This massively improves retrieval accuracy.

---

## 7.3 Advanced RAG Techniques (Agentic RAG, Routing)

Moving from static "Retrieve-then-Generate" architectures to **Agentic RAG**:
- **Query Routing:** Before executing a search, use a fast LLM or classifier to decide *where* to search.
  - "What is our company's leave policy?" $\rightarrow$ Route to vector search over HR PDFs.
  - "Summarize my meetings today" $\rightarrow$ Route to SQL database via tool calling.
- **Query Transformation / Expansion:**
  - Standard user queries are often vague.
  - Using algorithms like **HyDE** (Hypothetical Document Embeddings): The LLM answers the query blindly (hallucinating), and you embed that hypothetical answer to find similar real documents in the vector space.
- **Iterative / Adaptive Retrieval (Self-Reflective RAG):**
  - The context is retrieved and an answer is drafted.
  - An internal critic agent scores if the drafted answer actually verified the query based completely on the context. If the context is missing information, it triggers a *new*, more specific search, iterating until the answer is complete.

---

## References

- Anthropic "Contextual Retrieval" (2024): https://www.anthropic.com/news/contextual-retrieval
