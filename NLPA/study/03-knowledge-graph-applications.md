# 03. Knowledge Graph Applications

## Knowledge Graphs (KGs)

A **Knowledge Graph** represents a collection of interlinked descriptions of entities – objects, events or concepts. KGs put data in context via linking and semantic metadata.

-   **Nodes**: Represent entities (e.g., "Albert Einstein", "Germany").
-   **Edges**: Represent relationships (e.g., "born_in", "capital_of").
-   **Triples**: The fundamental unit of a KG: *(Subject, Predicate, Object)*.
    -   Example: *(Paris, is_capital_of, France)*.

### Why do we need Knowledge Graphs?

1.  **Structured Knowledge**: Explicitly represents relationships, unlike unstructured text.
2.  **Inference**: Allows deriving new facts from existing ones (e.g., if A is parent of B, and B is parent of C, then A is grandparent of C).
3.  **Disambiguation**: Helps in resolving entity ambiguity by using context from the graph.
4.  **Explainability**: Paths in the graph provide a trace for reasoning.

### Entity Linking Accuracy

Entity linking maps mentions in text to canonical KG entities.
If linking accuracy is $a$ and a query contains $n$ detected mentions, expected correctly linked mentions are:

$$
\mathbb{E}[\text{correct links}] = n\times a
$$

### Knowledge Graphs in Chatbots

Knowledge Graphs support knowledge-grounded responses by enabling:
- entity disambiguation in user queries
- multi-hop retrieval over connected facts
- factual answer generation with traceable relations

## Retrieval-Augmented Generation (RAG)

Combines the power of Large Language Models (LLMs) with external knowledge sources.

### Traditional RAG

1.  **Retrieval**: Given a user query, retrieve relevant documents/chunks from a vector database based on similarity.
2.  **Augmentation**: Feed the retrieved context along with the query to the LLM.
3.  **Generation**: The LLM generates the answer using the augmented context.

**Limitations**:
-   Vector similarity might miss semantically relevant but lexically different information.
-   Struggles with multi-hop reasoning (connecting facts across different documents).

### Graph RAG (GraphRAG)

Enhances RAG by using a Knowledge Graph as the retrieval source or to augment vector retrieval.

-   **Structured Retrieval**: Retrieve sub-graphs relevant to the query entities.
-   **Multi-hop Reasoning**: Traverse the graph to find connected information that vector search might miss.
-   **Context Enrichment**: Use entity relationships to provide richer context to the LLM.

## Agentic RAG

integrates RAG with **AI Agents** that can perform actions and reasoning.

-   **Iterative Retrieval**: The agent can refine its search queries based on initial findings.
-   **Tool Use**: Agents can use tools (including RAG, calculators, APIs) to answer complex queries.
-   **Planning**: The agent breaks down a complex question into sub-tasks and executes them using RAG and other tools.
