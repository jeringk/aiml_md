# Open-Book Exam Strategy for Conversational AI (CAI)

Since the BITS Comprehensive Exam is **open-book** (meaning you can bring your lecture slides and printed study materials), your preparation strategy needs to shift drastically from *rote memorization* to *fast information retrieval and conceptual application*.

Here is a structured strategy for creating your "Exam Cheat Sheet" and organizing your printed slides specifically to tackle the 5 core question types.

---

## 1. Indexing Your Printed Materials (Crucial Step)

Do not just bring a stack of unstapled slides. Instead, create a **Master Index Page** to place at the very front of your binder. Because you know the 5 exact topics being tested, format your Index strictly around them so you can flip to the exact slide instantly.

**Example Master Index:**
*   **Topic 1: Advanced RAG Architectures** $\rightarrow$ Tab Red
    *   Self-RAG (Slide \#)
    *   Adaptive RAG (Slide \#)
    *   CRAG (Slide \#)
*   **Topic 2: Evaluation Metrics (MATH)** $\rightarrow$ Tab Blue
    *   Formula: DCG, IDCG, NDCG (Slide \#)
    *   Formula: MRR & MAP (Slide \#)
*   **Topic 3: Ethics & Bias** $\rightarrow$ Tab Yellow
    *   Measurement vs Historical vs Representational (Slide \#)
    *   Guardrails / Red-teaming (Slide \#)
*   **Topic 4: Agentic Systems / ReAct** $\rightarrow$ Tab Green
    *   ReAct Core Diagram (Slide \#)
    *   Supervisor vs Swarm Diagrams (Slide \#) 
*   **Topic 5: Interoperability & Optimization** $\rightarrow$ Tab Orange
    *   MCP Architecture Diagram (Slide \#)
    *   A2A Protocol (Slide \#)
    *   Quantization / Distillation (Slide \#)

---

## 2. Topic-Specific "Cheat Sheets" to Print

Even with open slides, you waste precious minutes fishing for formulas or comparing concepts. You should bring a few 1-page condensed "Cheat Sheets" specifically formatted to answer the complex scenario questions.

### Cheat Sheet A: The Evaluation Math Sheet (Topic 2)
The problem question is guaranteed to be on Evaluation Metrics. In a high-pressure exam, calculating logs by hand is tedious. Print a pre-calculated table for your denominator:

**DCG Denominator Reference Table:**
(Formula: $\log_2(\text{rank} + 1)$)
- Rank 1: $1 / \log_2(2) = 1 / 1.00 = 1.00$
- Rank 2: $1 / \log_2(3) = 1 / 1.58 = 0.63$
- Rank 3: $1 / \log_2(4) = 1 / 2.00 = 0.50$
- Rank 4: $1 / \log_2(5) = 1 / 2.32 = 0.43$
- Rank 5: $1 / \log_2(6) = 1 / 2.58 = 0.38$
- Rank 6: $1 / \log_2(7) = 1 / 2.80 = 0.35$

*If a question asks you to calculate DCG for a document at Rank 5 with relevance 3, you instantly know the math is simply $3 \times 0.38 = 1.14$, saving you massive amounts of time.*

### Cheat Sheet B: Advanced RAG Comparison Matrix (Topic 1)
Since you are guaranteed a scenario question asking "when to use Self vs Adaptive vs Corrective RAG", print a highly structured comparison table.

| Architecture | Core Trigger | Primary Benefit | Flaw / Cost |
| :--- | :--- | :--- | :--- |
| **Adaptive RAG** | **Pre-Retrieval:** Query is routed *before* searching. | Bypasses slow vector searches for simple factual/math queries. | Relying on a smaller fast LLM router can misclassify complex prompts. |
| **Self-RAG** | **During Generation:** LLM actively critiques its own drafted chunks. | Prevents hallucination; ensures the generated text strictly aligns with retrieved data. | Extremely token-expensive; requires a specialized fine-tuned "critic" model. |
| **Corrective RAG (CRAG)** | **Post-Retrieval:** An evaluator scores the retrieved documents *before* generation. | If retrieved docs are irrelevant, it autonomously executes a web-search fallback. | Can significantly increase user latency if the fallback web-search is slow. |

### Cheat Sheet C: Protocol & Architecture "When to Use" Guide (Topic 4 & 5)
A scenario question will try to trick you into guessing the wrong architecture.

*   **MCP vs A2A:**
    *   Choose **MCP (Model Context Protocol)** when: A centrally controlled, singular LLM agent needs secure, local access to strictly deterministic tools (e.g., pulling internal GitHub repos, basic SQL database reads).
    *   Choose **A2A (Agent Protocol)** when: The system inherently requires highly decentralized, autonomous decision-loops stretching across organizational trust boundaries (e.g., an internal HR Agent explicitly negotiating with an external Vendor Agent).
*   **Swarm vs Supervisor (Multi-Agent):**
    *   Choose **Supervisor** when: The workflow is linear/hierarchical and rigorous error recovery (catching infinite ReAct loops) is mandatory.
    *   Choose **Swarm** when: The task is wildly unpredictable, requiring peers to creatively brainstorm or publish/subscribe to asynchronous event buses.
*   **LLM vs SLM:**
    *   Choose **Cloud LLM** when: Broad, unstructured reasoning or massive parametric world knowledge is required (e.g., "Draft a creative marketing email").
    *   Choose **Edge SLM** when: The task is highly repetitive/narrow (routing, JSON extraction), data privacy is absolute (offline), or hardware latency/VRAM is heavily constrained.

---

## 3. How to Answer Scenario Questions (The "2 Points / 1 Mark" Rule)

Because it is an open-book exam, simply copying the definition from a slide will earn you **zero marks**. The grader knows you have the slides. 
To satisfy the strict "2 points per mark" rule, map your answers like this:

**Scenario Example Structure:**
1. **Identify the Concept:** (e.g., "This scenario strictly mandates Corrective RAG (CRAG).")
2. **Define the Concept (from slide):** ("CRAG utilizes a retrieval evaluator to score the vector output...")
3. **Bridge/Adapt to the Scenario (Crucial):** ("...Therefore, when the medical assistant fetched the outdated 1990s papers, CRAG's evaluator would flag it as functionally useless for modern treatments...")
4. **State the Outcome/Benefit:** ("...and autonomously trigger a fallback web-search query restricted to post-2020 medical journals, preventing clinical hallucination.")

By explicitly dragging the names/nouns from the prompt (e.g., "medical assistant", "1990s papers") into the textbook definition, you guarantee maximum marks.
