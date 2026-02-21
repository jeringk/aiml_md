# Module 3 — Agentic Systems & Multi-Agent Collaboration

## Topics

- [[#8.1 Agentic workflows and frameworks|Agentic workflows and frameworks]]
- [[#8.2 Hierarchical & Collaborative Architectures|Multi-Agent Collaboration]]
- [[#8.3 Error Recovery & Iteration Limits|Error Recovery & Iteration Limits]]

---

## 8.1 Agentic workflows and frameworks

While an LLM simply generates static text, an **Agent** is an LLM architecturally granted access to *Tools* (functions/APIs) and a concrete *Workflow* enabling it to logically execute reasoning over iterations.

**Key Agentic Workflows:**
1. **ReAct (Reasoning and Acting):**
   - The agent operates strictly within a continuous programmatic loop of: `Thought $\rightarrow$ Action $\rightarrow$ Observation`.
   - *Thought:* "I need to find the current weather in Paris."
   - *Action:* `get_weather(location="Paris")`
   - *Observation:* The API returns "15°C, Sunny".
   - *Thought:* "Now that I have the weather, I can write the response to the user."
2. **Plan-and-Solve:**
   - The agent strictly separates macroscopic planning from microscopic execution to avoid getting lost in dense context loops.
   - First, the LLM generates a sequential, numbered step-by-step master plan. Next, it independently executes each point of the plan.
3. **Reflection & Self-Correction:**
   - The agent finishes generating the response to a task, but before returning it to the user flow, an internal deterministic loop asks: "Is this correct? Did I adhere strictly to all user constraints?". If flawed, it recursively corrects its own work.

**Core Agentic Frameworks:**
- **LangChain / LangGraph:** Industry-standard frameworks that represent agent workflows algorithmically as cyclical graphs (the nodes are LLM calls or programmatic tools, edges are conditional logic routing). Critical for maintaining state over long multi-step agent actions.
- **MetaGPT:** Functionally focuses strictly on instantiating standard operating procedures (SOPs) into individual agents. It simulates an entire cross-functional software engineering team (Product Manager Agent, Coder Agent, QA Agent).

---

## 8.2 Hierarchical & Collaborative Architectures (Multi-Agent)

Instead of relying on an unreliable single massive monolithic prompt/agent attempting to do all subtasks flawlessly, **Multi-Agent Collaboration** surgically distributes the architecture among highly specialized, strictly scoped independent agents.

**Fundamental Benefits:**
- **Reduced Context Sprawl:** Each individual sub-agent requires only the highly specific instructions and tools necessary for its narrow role, dramatically reducing token fatigue.
- **Specialized Accuracy:** A "Coder Agent" solely outputs code, a disparate "Code Reviewer Agent" only critiques it. 

**Architectural Collaboration Patterns:**
1. **Supervisor Pattern (Hierarchical):** A root node "Supervisor Agent" ingests the primary complex user request, conceptually decomposes it into discrete subtasks, and dynamically routes them to domain-expert worker agents. The supervisor then aggregates and controls the final combined output state.
2. **Network / Swarm Pattern (Collaborative):** Agents operate fundamentally as flat peers and publish/message each other dynamically to negotiate state resolutions and tackle deeply arbitrary workflows together.

---

## 8.3 Error Recovery & Iteration Limits

Unchecked agents heavily run the risk of infinite loops (e.g., repeatedly generating broken Python code, executing it, reading the error, generating the exact same broken Python code).
- **Iteration Constraints:** Hardcoding a strict integer limit (e.g., `max_iterations=5`) before the agent programmatically catches a timeout error and falls back gracefully to a human.
- **Dynamic State Management:** Using frameworks to serialize and track the comprehensive memory history of attempted tool calls. This allows the backend to force the agent to introspect and deduce: "I have repeatedly attempted this API structure and it failed; I am required to try a dramatically different logical approach."

---

## References

- "MetaGPT: Meta Programming for Multi-Agent Systems" (2024)
- "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)
