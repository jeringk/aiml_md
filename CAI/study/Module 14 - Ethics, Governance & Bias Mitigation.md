# Module 4 — Ethics, Governance & Bias Mitigation

## Topics

- [[#14.1 Bias Identification|Bias Identification & Manifestations]]
- [[#14.2 Mitigation Strategies & Debiasing|Mitigation Strategies & Debiasing]]
- [[#14.3 Self-Improving Agents & Alignment Risks|Self-Improving Agents & Alignment Risks]]

---

## 14.1 Bias Identification

Language Models fundamentally reflect the deep systematic biases nested within the human-generated data they were trained on.
- **Representational Bias:** Occurs when the training data drastically lacks representation of certain minority demographics (e.g., facial recognition systems failing heavily on darker skin tones; language models struggling with AAVE).
- **Historical Bias:** Present when models accurately reflect reality/history, but the historical data itself was highly prejudiced (e.g., an AI resume screening tool learning to automatically penalize women because historical company hires for that engineering role were 95% men).
- **Measurement Bias:** Arises when the specific features or labels chosen to train a model are inadvertently skewed (e.g., predicting "crime risk" based on raw arrest records, which heavily skews results owing to disproportionate over-policing in specific neighborhoods).

---

## 14.2 Mitigation Strategies & Debiasing

Mitigation and safeguard procedures must be applied across every stage of the machine learning lifecycle:
1. **Data Level:** Aggressive re-sampling, algorithmic re-weighting, and augmenting datasets to guarantee equal demographic representation. Manual data curation and scraping to intentionally filter out explicitly toxic text bases.
2. **Algorithm/Training Level:**
   - **RLHF (Reinforcement Learning from Human Feedback) / DPO:** Conditioning the model post-training to specifically output helpful/harmless responses through strict preference optimization.
   - **Adversarial Debiasing:** Training a secondary classifier simultaneously to functionally *predict* the protected attribute (like race/gender) based on the model's embeddings. If it correctly predicts it, the primary model is penalized, forcing the system to learn internal representations strictly independent of that attribute.
3. **Inference Level (Guardrails):** Systematically appending hidden steering prompts or executing programmatic filtering modules dynamically to prevent outputs if they violate explicit safety and ethics bounds.

---

## 14.3 Self-Improving Agents & Alignment Risks

As modern conversational AI shifts exclusively toward autonomous agents (systems that can autonomously write and execute code, access the web, and correct their own logic loops):
- **The Alignment Problem:** Guaranteeing the autonomous agent's actual operational goals flawlessly align with humanity's intended safety goals.
- **Instrumental Convergence:** An agent tasked with a completely benign goal (e.g., "Calculate pi to its furthest extent") might recursively take catastrophic, dangerous steps to achieve it (e.g., taking over all regional computing infrastructure to unilaterally increase its compute power).
- **Governance:** Instantiating rigorous **"Human-in-the-Loop" (HITL)** architecture for mission-critical agent actions. Architecturally requiring agents to explicitly break execution loops to ask human supervisors for documented approval before escalating state changes (like sending a destructive email or executing a financial ledger transaction).

---

## References

- Anthropic's Responsible Scaling Policy: https://www.anthropic.com/news/anthropics-responsible-scaling-policy
