# Topic 4: Agentic Systems & Multi-Agent Collaboration - In-Depth MCQs

## Question 1 (Agentic Workflows)

When building an autonomous agent framework, a developer utilizes a prompt engineering paradigm strictly forcing the LLM to output a `Thought`, execute an `Action` (tool call), and ingest the subsequent `Observation` before producing another `Thought`. The loop continues until the system believes the query is handled. 

What foundational framework does this exact execution loop describe?

A) The GAIA Benchmark Loop
B) Plan-and-Solve
C) Self-Reflective Retrieval
D) ReAct (Reasoning and Acting)

<div style="page-break-after: always"></div>

### Topics to know to answer Question 1
- [[../study/Module 8 - Agent Planning & Multi-Agent Systems.md#8.1 Agentic workflows and frameworks|Agentic workflows and frameworks]]

<div style="page-break-after: always"></div>

### Solution 1

**Correct Answer: D**

**Explanation:**
The **ReAct (Reasoning and Acting)** framework (Yao et al., 2023) is arguably the most famous and foundational prompt engineering structure for AI agents. By rigorously interleaving reasoning traces (Thought) and task-specific actions (Action/Observation), it forces the LLM to stay grounded in reality. The structure prevents hallucination because the agent fundamentally cannot "guess" an answer; it must generate a visible *Thought* determining what API/tool to use, trigger the *Action*, and ground its next decision entirely strictly on the text returned in the *Observation*.

---

## Question 2 (Multi-Agent Architectures)

Why do enterprise conversational architectures drastically prefer utilizing a Multi-Agent Swarm or Supervisor network instead of relying entirely on a single, massive, monolithic LLM zero-shot prompt?

A) A single monolithic prompt is fundamentally unable to execute external APIs/Tools, whereas Multi-Agent frameworks inherently have internet access.
B) Specializing multiple narrow agents drastically reduces token context sprawl, allowing individual sub-agents to only receive the exact tools and strict instructions necessary for their discrete role, greatly maximizing reasoning accuracy and reducing token costs.
C) Multi-Agent architectures bypass the need for an LLM entirely, shifting operations exclusively to programmatic Python scripts.
D) Only multi-agent systems are natively protected against Direct Prompt Injections and Jailbreaking.

<div style="page-break-after: always"></div>

### Topics to know to answer Question 2
- [[../study/Module 8 - Agent Planning & Multi-Agent Systems.md#8.2 Hierarchical & Collaborative Architectures (Multi-Agent)|Multi-Agent Collaboration]]

<div style="page-break-after: always"></div>

### Solution 2

**Correct Answer: B**

**Explanation:**
Loading an LLM with 50 pages of instructions covering every possible organizational SOP (how to code, how to review code, how to talk to users, how to execute SQL) leads to catastrophic context fatigue, severe hallucination, and exorbitant API token costs. **Multi-Agent Collaboration** surgically decouples this logic. By instantiating highly specialized agents (e.g., a "Coder Agent" with only 3 tools and strict coding system instructions interacting with a separate "Reviewer Agent"), you maintain pristine context windows, leading to far higher execution accuracy, fault isolation, and significantly cheaper API calls over long iterations.

---

## Question 3 (A2A Collaboration Protocols)

In a highly decentralized multi-agent ecosystem, Agent A (HR Support) must request secure budgetary data from an unknown, distinct peer Agent B (Finance Controller). To dynamically discover Agent B's capabilities, input schema, semantic endpoints, and verify security protocols without human intervention, what standard artifact does Agent A fundamentally rely upon reading?

A) The ReAct Observation Trace
B) Agent Cards 
C) Model Routing Weights
D) The RAG Context Chunk

<div style="page-break-after: always"></div>

### Topics to know to answer Question 3
- [[../study/Module 13 - A2A & Interoperability.md#13.2 Agent Cards, Task Lifecycle|Agent Cards, Task Lifecycle]]

<div style="page-break-after: always"></div>

### Solution 3

**Correct Answer: B**

**Explanation:**
In the A2A (Agent-to-Agent) Protocol specification, **Agent Cards** are the fundamental declarative building blocks for decentralized discovery. Conceptually similar to "Model Cards" in HuggingFace or OpenAPI/Swagger specs in REST development, an Agent Card is a rigid metadata file (YAML/JSON) that officially broadcasts exactly what an Agent is capable of. It defines its required payload structure, the authorized tools it has access to, latency SLAs, and the specific authorization schema required to initiate a handshake, allowing disparate agents to orchestrate complex tasks dynamically.
