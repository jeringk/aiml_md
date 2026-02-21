# Module 4 — A2A (Agent-to-Agent) & Interoperability

## Topics

- [[#13.1 Collaboration Protocols (A2A)|Collaboration Protocols (A2A)]]
- [[#13.2 Agent Cards, Task Lifecycle|Agent Cards, Task Lifecycle]]
- [[#13.3 Agent Orchestration Patterns|Agent Orchestration Patterns]]

---

## 13.1 Collaboration Protocols (A2A)

As decentralized enterprise architectures scale to deploy hundreds of independent AI agents (e.g., an autonomous HR Agent, an IT Support Agent, a CRM Finance Agent), they must feature programmatic pathways to interact, verify, and authenticate with each other. This is fundamentally more complex than standard API REST endpoints.

**A2A (Agent-to-Agent) Protocol:**
- An industry standard specification defining how completely decoupled AI agents successfully discover each other on a lattice, safely negotiate capabilities, and delegate complex tasks.
- **Core Use-Case:** If an employee requests a new laptop via the HR Agent chat, the HR Agent must execute A2A handshake calls to directly ping the IT Agent (for hardware specs) and the Finance Agent (for budget checks), synchronously passing conversational state and retrieving decisions.
- **Agent Protocol:** An active open-source standard created by the AI Engineer Foundation establishing a unified, single REST API schema for developers to interact with agents. It allows previously isolated agent frameworks (LangChain, AutoGPT, CrewAI) to seamlessly interoperate via standard JSON payloads.

---

## 13.2 Agent Cards, Task Lifecycle

**Agent Cards:**
- Structurally mirroring "Model Cards" in open-source ML, an Agent Card is a strict, standardized JSON/YAML metadata document outlining an Agent state.
- **Core Parameters:** Describes the exact purpose scope of the agent, the specific Tools/APIs it securely accesses, semantic endpoints, latency SLA guarantees, and required JWT authorization schemas.
- Foundational during runtime for programmatic discovery. A Supervisor agent retrieves the Agent Card of a subordinate node exclusively to verify its internal capabilities map to the required task logic.

**A2A Collaborative Task Lifecycle:**
1. **Discovery:** Agent identifies target peer dynamically via an internal organizational registry (reading parsed Agent Cards).
2. **Handshake & Auth:** System verifies if Agent A holds proper security/RBAC clearance to query Agent B for sensitive dataset execution.
3. **Execution & Streaming:** Agent A asynchronously streams natural language or JSON instructions; conversely, Agent B streams intermediate observations or final output chunks backward.
4. **Resolution/Fault:** Task formally succeeds or faults, returning deterministic structured stack traces instead of messy, unparseable conversational text.

---

## 13.3 Agent Orchestration Patterns

- **Choreography:** Autonomous agents trigger responses in peers completely via decoupled event-driven architecture (publish/subscribe buses). Features high vertical scalability but is notoriously difficult to trace and debug log-loops.
- **Orchestration:** A central stateful controller (similar to a Kubernetes control plane serving Agents) rigidly manages the exact execution workflow. The Orchestrator actively handles timeout retries, pauses agents awaiting mandatory human approval (HITL safeguards), and maps shared persistent context memory layers across the entire underlying fleet.

---

## References

- A2A Protocol Specification
