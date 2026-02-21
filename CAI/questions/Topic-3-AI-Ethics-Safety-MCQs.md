# Topic 3: AI Ethics, Bias, & Safety - In-Depth MCQs

## Question 1 (Data Privacy and Guardrails)

A healthcare Agentic system is designed to summarize daily patient encounter notes and push them to an external LLM provider API. To ensure compliance with HIPAA data security standards, the engineering team deploys Microsoft Presidio ahead of the sequence. What explicit operational step is being executed to guarantee data safety?

A) Direct Prompt Injection defense via XML delimiters appended to the prompt string.
B) Pre-processing Redaction, dynamically masking sensitive elements like names and SSNs before they leave the localized infrastructure.
C) Adversarial testing of the LLM endpoint to guarantee it drops connection upon seeing PHI.
D) De-tokenization, forcing the external API to generate encrypted tokens that only decrypt locally.

<div style="page-break-after: always"></div>

### Topics to know to answer Question 1
- [[../study/Module 11 - Security & Adversarial Robustness.md#11.2 Security & Guardrails (PII Detection & Redaction)|Security & Guardrails: PII Classification & Redaction]]

<div style="page-break-after: always"></div>

### Solution 1

**Correct Answer: B**

**Explanation:**
In enterprise AI pipelines dealing with sensitive and regulated information (like Healthcare PHI or Financial PII), the fundamental safeguard is preventing the raw text from ever leaving your secure local infrastructure boundaries. Tools like Microsoft Presidio are deployed as a **Pre-processing Redaction layer**. They run highly deterministic Regex or lightweight Named Entity Recognition (NER) locally to detect and mask sensitive entities (e.g., replacing "John Doe" with "[REDACTED_NAME]"). This guarantees that the payload sent to external commercial APIs contains absolutely zero protected data. 

---

## Question 2 (Bias Identification)

A multinational bank trains a new generative AI tool to draft loan approval determinations based on an employee's summarized inputs. Upon auditing, they realize the tool consistently advises rejecting applicants from specific zip codes at an astonishingly high rate compared to others. The development team proves the model is functioning perfectly mathematically and tracing the company's past 25 years of loan decisions with $99\%$ accuracy. 

What specific type of algorithmic bias has critically infected this system?

A) Representational Bias
B) Measurement Bias
C) Historical Bias
D) Self-enhancement Bias

<div style="page-break-after: always"></div>

### Topics to know to answer Question 2
- [[../study/Module 14 - Ethics, Governance & Bias Mitigation.md#14.1 Bias Identification|Bias Identification & Manifestations]]

<div style="page-break-after: always"></div>

### Solution 2

**Correct Answer: C**

**Explanation:**
**Historical Bias** occurs when an AI model accurately and faithfully reproduces the exact patterns present in the training data, but that underlying historical data represents a fundamentally flawed or prejudiced societal reality (like historical redlining in bank loans or sexist hiring practices over previous decades). 
- *Representational Bias* would happen if the model failed because it had absolutely no training data on users from a certain country.
- *Measurement Bias* happens when the proxy variables chosen to train the model are skewed (e.g., measuring "good policing" strictly by "number of arrests").

---

## Question 3 (Alignment & Mitigation)

As autonomous agents evolve to handle looping execution with tools (e.g., executing code or managing financial ledgers without user prompting), what is the most robust, architectural governance mechanism an enterprise can implement to prevent devastating "Instrumental Convergence" failures?

A) Requiring all agents to use open-source Small Language Models (SLMs) rather than Frontier LLMs to limit latent reasoning.
B) Implementing rigorous "Human-in-the-Loop" (HITL) checkpoints that programmatically halt execution state and demand verified human supervisor approval before triggering authoritative system changes.
C) Aggressively applying DPO (Direct Preference Optimization) on the base model to fine-tune it to be perfectly harmless.
D) Applying context caching algorithms to ensure the agent only remembers the last 5 iterations, preventing long-form destructive planning.

<div style="page-break-after: always"></div>

### Topics to know to answer Question 3
- [[../study/Module 14 - Ethics, Governance & Bias Mitigation.md#14.3 Self-Improving Agents & Alignment Risks|Self-Improving Agents & Alignment Risks]]

<div style="page-break-after: always"></div>

### Solution 3

**Correct Answer: B**

**Explanation:**
While aligning the base foundation models via RLHF and DPO is critical, it is never statistically flawless against jailbreaks or complex logic loops. When agents are granted authoritative access to modify external systems (sending emails, modifying databases, executing trades), the ultimate governance architecture is a **Human-in-the-Loop (HITL)** safeguard. Frameworks like LangGraph explicitly support edge nodes that serialize the agent's current state into a frozen graph, pushing a notification to an admin dashboard, and strictly waiting for explicit human approval before the edge is permitted to continue executing the final, irreversible action.
