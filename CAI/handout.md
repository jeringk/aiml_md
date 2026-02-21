# Conversational AI (AIMLCZG521) - Course Handout

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI**
**WORK INTEGRATED LEARNING PROGRAMMES**
**Digital**

---

## Part A: Content Design

| Field | Details |
| :--- | :--- |
| **Course Title** | Conversational AI |
| **Course No(s)** | AIMLCZG521 |
| **Credit Units** | 4 |
| **Credit Model** | |
| **Content Authors** | Bhagath S |

### Course Description
This course teaches you how to build real-world conversational AI systems that can think, plan, and act autonomously. You'll learn to create intelligent agents that go far beyond simple chatbots - systems that can use tools, remember past conversations, retrieve information from large databases, and work together with other agents to solve complex problems.

The course emphasizes production-ready skills that companies need today: building cost-effective systems, making them secure against attacks, ensuring they follow ethical guidelines, and connecting them using modern protocols. You'll work with Large Language Models (LLMs), implement sophisticated memory systems, and learn how to optimize costs while maintaining high performance.

By the end of this course, you'll be able to design, build, evaluate, and deploy conversational AI agents for enterprise applications. You'll understand both the technical architecture and the business considerations - from choosing the right model to implementing safety guardrails and monitoring system performance.

### Pre-requisites
*   **Python Programming:** Comfortable writing Python code, using functions, classes, and basic libraries.
*   **Machine Learning Basics:** Understanding of how ML models work (training, inference, evaluation).
*   **Deep Learning Fundamentals:** Familiarity with neural networks and transformers (from your Deep Learning course).
*   **API Basics:** Knowledge of how to make API calls and work with JSON data.
*   **Basic Statistics:** Understanding of metrics like precision, recall, and probability.

### Course Objectives

1.  **CO1:** Understand and explain how modern conversational AI systems work, including the architecture of LLM-based agents, memory systems, and tool integration.
2.  **CO2:** Design and build intelligent agents that can use multiple tools, remember context, retrieve relevant information, and reason through multi-step problems.
3.  **CO3:** Evaluate agent performance using appropriate metrics, identify failure modes, implement cost optimization strategies, and ensure system reliability.
4.  **CO4:** Apply security best practices, implement ethical safeguards, and integrate agents using modern protocols (MCP, A2A) while considering production deployment requirements.

### Learning Outcomes

1.  **LO1:** Build end-to-end conversational AI systems with proper architecture - including vector databases for retrieval, function calling for tool use, and graph-based workflows for complex reasoning.
2.  **LO2:** Implement cost-effective solutions using techniques like prompt caching, model routing, and efficient retrieval strategies that reduce token usage.
3.  **LO3:** Create secure agents that defend against prompt injection, handle sensitive data properly, and include monitoring dashboards for tracking performance, costs, and errors.
4.  **LO4:** Design agents that follow ethical guidelines, mitigate bias, include human oversight for critical decisions, and comply with emerging AI regulations and industry standards.

### Modules

1.  **Foundations** - Embeddings, retrieval, model landscape, and cost engineering.
2.  **Core Building Blocks** - Function calling, memory systems, RAG pipelines.
3.  **Autonomous Agents** - Planning, multi-agent systems, evaluation, optimization.
4.  **Production Ecosystem** - Security, protocols (MCP, A2A), ethics, and governance.

### Textbooks & References

#### Textbooks
**T1: Official Documentation and Technical Guides**
1.  Anthropic's "Building Effective Agents" guide
2.  OpenAI Function Calling documentation
3.  MCP (Model Context Protocol) & A2A specification
*Note: These official documentation resources are freely available online and provide the most current, practical guidance for building modern agent systems.*

#### References
**R1:** "Speech and Language Processing" by Daniel Jurafsky and James H. Martin
*For foundational NLP concepts and linguistic background*

**R2: Research Papers (provided during course):**
1.  "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)
2.  "Direct Preference Optimization" (Rafailov et al., 2023)
3.  "Dense Passage Retrieval for Open-Domain Question Answering" (2020)

**R3: Industry Resources & arXiv.org and Conference Proceedings**
*For latest research from ACL, EMNLP, NeurIPS, and ICML conferences*

---

## Part B: Learning Plan

### Session Plan

| Contact Session | List of Topic Title (from content structure in Part A) | Module # | Ref Book / resource |
| :---: | :--- | :---: | :--- |
| **L1** | **Foundations of Conversational AI:**<br>- Chatbots to Agentic Systems<br>- System Lifecycle & Architecture | Module 1 | "The Landscape of AI Agents" (2024) - https://arxiv.org/abs/2404.11584 |
| **L2** | **Embeddings, Vector Search & Hybrid Retrieval:**<br>- Semantic vs Keyword Search<br>- Vector Database Architecture (HNSW, ANN)<br>- BM25 + Dense Retrieval + RRF | Module 1 | "Dense Passage Retrieval" (Karpukhin et al., 2020) |
| **L3** | **Model Landscape & Cost Engineering:**<br>- LLMs, MoE, SLMs, SSMs Comparison<br>- Quantization Techniques & KV-Cache<br>- Prompt Caching & Model Routing | Module 1 | "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023) |
| **L4** | **Structured Outputs & Function Calling:**<br>- Native Function Calling APIs (OpenAI, Anthropic)<br>- ReAct Framework (Thought-Action-Observation)<br>- Error Handling & Validation | Module 2 | "ReAct: Synergizing Reasoning and Acting" (Yao et al., 2023) |
| **L5** | **Fine-Tuning & Preference Optimization:**<br>- Fine-Tune vs Prompt Engineering<br>- QLoRA for Parameter-Efficient Training<br>- DPO, GRPO Techniques | Module 2 | "Direct Preference Optimization" (Rafailov et al., 2023) |
| **L6** | **Agent Memory Systems:**<br>- Short-Term vs Long-Term Memory<br>- Hybrid Architecture (SQL + Vector) | Module 2 | "MemGPT: Towards LLMs as Operating Systems" (2023) + LangGraph Memory Docs |
| **L7 & L8** | **RAG: Foundations to Advanced:**<br>- Processing & Chunking Strategies<br>- Re-ranking & Contextual Retrieval<br>- Agentic RAG: Routing & Iteration<br>*Mid-Term Revision* | Module 2 | Anthropic "Contextual Retrieval" (2024)<br>https://www.anthropic.com/news/contextual-retrieval |
| **L9** | **Agent Planning & Multi-Agent Systems:**<br>- State Management & Planning Strategies<br>- Hierarchical & Collaborative Architectures<br>- Error Recovery & Iteration Limits | Module 3 | "MetaGPT: Meta Programming for Multi-Agent Systems" (2024) |
| **L10** | **Evaluation: RAG to Agents:**<br>- RAG & Agent Metrics<br>- LLM-as-Judge Pattern & Limitations<br>- Benchmarks | Module 3 | "Judging LLM-as-a-Judge with MT-Bench" (2023) + GAIA Benchmark |
| **L11** | **Cost Optimization & Prompt Caching:**<br>- Token Economics & Hidden Costs<br>- Prompt Caching<br>- Cache Warming & Invalidation<br>- Model Routing | Module 3 | Anthropic Prompt Caching<br>https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching |
| **L12** | **Security & Adversarial Robustness:**<br>- Prompt Injection (Direct & Indirect) Defense<br>- PII Detection & Redaction<br>- Red-Teaming Strategies | Module 4 | OpenAI Prompt Caching:<br>https://platform.openai.com/docs/guides/prompt-caching |
| **L13** | **MCP (Model Context Protocol) Deep Dive:**<br>- Architecture: Client-Server Model<br>- Primitives: Resources, Tools, Prompts<br>- Building MCP Servers | Module 4 | MCP Specification |
| **L14** | **A2A (Agent-to-Agent) & Interoperability:**<br>- A2A Components: Agent Cards, Task Lifecycle<br>- Protocol Comparison (A2A, Agent Protocol)<br>- Agent Orchestration Patterns | Module 4 | A2A Protocol Spec |
| **L15 & L16** | **Ethics, Governance & Bias Mitigation:**<br>- Bias Types & Manifestations<br>- Mitigation Strategies & Debiasing<br>- Self-Improving Agents & Risks<br>*Final Revision* | Module 4 | Anthropic's Responsible Scaling Policy:<br>https://www.anthropic.com/news/anthropics-responsible-scaling-policy |
