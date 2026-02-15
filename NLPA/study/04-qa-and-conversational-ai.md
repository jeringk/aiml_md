# 04. Question Answering and Conversational AI

## Question Answering (QA)

Systems that automatically answer questions posed by humans in a natural language.

### Types of QA Systems

1.  **IR-based QA**:
    -   Retrieves relevant documents and extracts spans of text as answers.
    -   *Factoid QA*: Answers are simple facts (e.g., "Who consists of...").
2.  **Knowledge-based QA**:
    -   Maps questions to logical queries (e.g., SPARQL) to query a Knowledge Graph.
    -   Good for structured data questions.

## Conversational AI

Systems capable of conversing with humans (Chatbots, Dialogue Systems).

### Components of a Dialogue System

1.  **NLU (Natural Language Understanding)**: Intent classification, Slot filling.
2.  **DST (Dialogue State Tracking)**: Maintains the current state of the conversation.
3.  **Policy Learning**: Decides the next action (what to say/do).
4.  **NLG (Natural Language Generation)**: Generates the response text.

## Hybrid Systems

Combine rule-based logic with neural models.
-   **Rule-based**: Good for precise, business-critical flows.
-   **Neural**: Good for handling open-ended queries and chitchat.

## Gen AI Techniques in QA & Conversational AI

### Large Language Models (LLMs)

Models like GPT, Llama, Gemini have revolutionized QA and Conversational AI.

-   **Zero-shot / Few-shot Learning**: Can answer questions without specific training examples.
-   **Chain-of-Thought (CoT)**: Prompting the model to "think step-by-step" improves reasoning for complex QA.

### Implementation with Gen AI

1.  **LLM-based Chatbots**: Fine-tuned or prompted LLMs handle the entire dialogue pipeline (NLU -> DST -> NLG).
2.  **RAG for QA**: Using LLMs with retrieved context to answer domain-specific questions (as discussed in Topic 3).
3.  **Agentic AI**:
    -   **ReAct (Reasoning + Acting)**: The model reasons about the query and decides which tools (search, database, calculator) to use.
    -   Example: "What is the weather in London?" -> Agent calls Weather API -> Agent generates natural language response.
