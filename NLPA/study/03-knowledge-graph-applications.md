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

### Core KG Construction Pipeline

1. Information extraction from text (entities, relations, events).
2. Entity resolution/linking to canonical identifiers.
3. Ontology/schema alignment (type and relation constraints).
4. Triple storage and indexing in graph database.
5. Reasoning/query layer for downstream applications.

### Entity Linking Accuracy

Entity linking maps mentions in text to canonical KG entities.
If linking accuracy is $a$ and a query contains $n$ detected mentions, expected correctly linked mentions are:

$$
\mathbb{E}[\text{correct links}] = n\times a
$$

where:
- $n$: detected mentions in input
- $a$: entity-linking accuracy

Numerical example: for $n=5$ and $a=0.78$,

$$
\mathbb{E}[\text{correct links}] = 5\times0.78=3.9
$$

So approximately 4 entities are linked correctly.

### Knowledge Graphs in Chatbots

Knowledge Graphs support knowledge-grounded responses by enabling:
- entity disambiguation in user queries
- multi-hop retrieval over connected facts
- factual answer generation with traceable relations

### KG Querying with Triple Patterns

SPARQL-style idea:
- Match graph patterns `(subject, predicate, object)` with variables.
- Return bindings satisfying all constraints.

Example pattern:
- `( ?city, capital_of, India )`
- `( ?city, located_in, ?state )`

This supports compositional question answering instead of flat keyword search.

### Path-Based Reasoning Score

A simple multi-hop reasoning score can be modeled as:

$$
\operatorname{score}(u\leadsto v)=\sum_{\pi\in\Pi(u,v)} \prod_{e\in\pi} w(e)
$$

where:
- $u$: query entity node
- $v$: candidate answer node
- $\Pi(u,v)$: set of paths from $u$ to $v$
- $w(e)$: edge confidence/importance for edge $e$

### Numerical Example: Path Reasoning Score

Suppose two paths from $u$ to $v$:

- $\pi_1$ with edge weights $0.9,0.8$
- $\pi_2$ with edge weights $0.7,0.6$

Then:

$$
\operatorname{score}(u\leadsto v)=(0.9\times0.8)+(0.7\times0.6)=0.72+0.42=1.14
$$

## Retrieval-Augmented Generation (RAG)

Combines the power of Large Language Models (LLMs) with external knowledge sources.

### Traditional RAG

1.  **Retrieval**: Given a user query, retrieve relevant documents/chunks from a vector database based on similarity.
2.  **Augmentation**: Feed the retrieved context along with the query to the LLM.
3.  **Generation**: The LLM generates the answer using the augmented context.

**Limitations**:
-   Vector similarity might miss semantically relevant but lexically different information.
-   Struggles with multi-hop reasoning (connecting facts across different documents).

### Dense Retrieval Similarity

$$
\operatorname{sim}(q,d)=\frac{\phi(q)\cdot\phi(d)}{\|\phi(q)\|\,\|\phi(d)\|}
$$

where:
- $q$: user query text
- $d$: candidate document/chunk
- $\phi(\cdot)$: embedding function
- $\operatorname{sim}(q,d)$: cosine similarity used for top-$k$ retrieval

### Numerical Example: Cosine Similarity

Let $\phi(q)=[1,2]$ and $\phi(d)=[2,1]$.

$$
\phi(q)\cdot\phi(d)=1\times2+2\times1=4
$$

$$
\|\phi(q)\|=\|\phi(d)\|=\sqrt{5}
$$

$$
\operatorname{sim}(q,d)=\frac{4}{\sqrt{5}\sqrt{5}}=\frac{4}{5}=0.8
$$

### Graph RAG (GraphRAG)

Enhances RAG by using a Knowledge Graph as the retrieval source or to augment vector retrieval.

-   **Structured Retrieval**: Retrieve sub-graphs relevant to the query entities.
-   **Multi-hop Reasoning**: Traverse the graph to find connected information that vector search might miss.
-   **Context Enrichment**: Use entity relationships to provide richer context to the LLM.

### GraphRAG Retrieval Steps

1. Detect entities in query and map to KG nodes.
2. Expand neighborhood (1-hop/2-hop) or relation-constrained subgraph.
3. Rank paths/subgraph snippets by relevance.
4. Serialize selected facts as grounded context for generation.

Subgraph ranking with hybrid score:

$$
\operatorname{rank}(g\mid q)=\lambda\,\operatorname{sim}_{\text{text}}(q,g)+(1-\lambda)\,\operatorname{sim}_{\text{graph}}(q,g)
$$

where:
- $g$: candidate subgraph/context item
- $q$: query
- $\operatorname{sim}_{\text{text}}$: text embedding similarity
- $\operatorname{sim}_{\text{graph}}$: graph-structure/entity overlap score
- $\lambda\in[0,1]$: interpolation weight

### Numerical Example: Hybrid Subgraph Ranking

Assume:
- $\operatorname{sim}_{\text{text}}(q,g)=0.70$
- $\operatorname{sim}_{\text{graph}}(q,g)=0.50$
- $\lambda=0.6$

$$
\operatorname{rank}(g\mid q)=0.6(0.70)+0.4(0.50)=0.62
$$

Candidate with higher rank score is selected for context.

## Agentic RAG

integrates RAG with **AI Agents** that can perform actions and reasoning.

-   **Iterative Retrieval**: The agent can refine its search queries based on initial findings.
-   **Tool Use**: Agents can use tools (including RAG, calculators, APIs) to answer complex queries.
-   **Planning**: The agent breaks down a complex question into sub-tasks and executes them using RAG and other tools.

### Agentic KG Workflow Example

Question: "Which Indian city has the highest population among capitals of southern states?"

1. Agent queries KG for southern states.
2. Retrieves each state capital node.
3. Joins with population attributes (KG or table API).
4. Compares values and returns top city with evidence path.

### Common KG Applications in NLPA

- **Question answering**: multi-hop factual QA.
- **Entity disambiguation**: resolve ambiguous mentions.
- **Recommendation/Personalization**: graph-based neighborhood inference.
- **Semantic search**: relation-aware retrieval beyond keyword overlap.
