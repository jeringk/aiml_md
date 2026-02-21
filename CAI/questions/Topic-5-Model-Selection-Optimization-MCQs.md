# Topic 5: Model Selection & Optimization - In-Depth MCQs

## Question 1 (SLM vs LLM)

An enterprise is designing an offline, highly localized conversational agent to be deployed directly onto manufacturing hardware controllers for basic command execution. Network latency prohibits calling cloud APIs, and the hardware possesses extremely limited VRAM and compute capabilities. Based on fundamental model architectures, which approach is definitively required?

A) Fine-tuning a Mixture of Experts (MoE) LLM array to optimize parameter usage.
B) Utilizing a Small Language Model (SLM) coupled with intense parameter quantization (e.g., 4-bit or 8-bit).
C) Implementing an Agentic RAG architecture utilizing a Cross-Encoder Re-ranker to maximize grounding.
D) Distributing a 70 Billion parameter dense LLM utilizing advanced Prompt Caching logic to reduce total input token overhead.

<div style="page-break-after: always"></div>

### Topics to know to answer Question 1
- [[../study/Module 3 - Model Landscape & Cost Engineering.md#3.1 LLMs, MoE, SLMs, SSMs Comparison|LLMs, MoE, SLMs Comparison]]
- Model Selection & Optimization

<div style="page-break-after: always"></div>

### Solution 1

**Correct Answer: B**

**Explanation:**
Massive frontier models (LLMs) and Mixture of Experts (MoE) require massive, clustered GPU footprints and are deployed strictly via Cloud infrastructures. For edge computing, embedded systems, and domains with strict latency/offline requirements and low hardware VRAM, the industry exclusively deploys **Small Language Models (SLMs)** (typically ranging from 1B to 8B parameters, like Microsoft Phi or Llama 8B). To further compress these models to run on generic hardware or NPUs without out-of-memory crashes, intense **Quantization** (reducing the precision of the model's weights from 32-bit floats down to 8-bit or 4-bit integers) is mandatorily applied.

---

## Question 2 (Optimization Techniques & Caching)

Prompt caching is a powerful optimization technique utilized by Providers like Anthropic to drastically reduce the cost of executing deeply sequential agentic workflows. Which of the following scenarios provides mathematically the *highest* cost and latency benefit from implementing Prompt Caching?

A) A chatbot answering thousands of entirely unique, unrelated single-turn user questions where no system instructions are applied.
B) Passing a massive 50,000-token PDF as context for an agent, then asking the agent 20 successive, distinct questions strictly about that identical document over a conversational session.
C) Utilizing the LLM to write a 1-sentence contextual summary for millions of distinct 100-token document chunks prior to vector embedding.
D) Applying preference tuning via DPO across an exceptionally large, randomized instruction dataset.

<div style="page-break-after: always"></div>

### Topics to know to answer Question 2
- [[../study/Module 10 - Cost Optimization & Prompt Caching.md#10.2 Prompt Caching|Prompt Caching]]
- Model Selection & Optimization

<div style="page-break-after: always"></div>

### Solution 2

**Correct Answer: B**

**Explanation:**
**Prompt Caching** (specifically KV-Cache reuse) provides mathematical leverage solely when a massive, static block of input tokens is fundamentally reused across multiple sequential API calls within a short time window. 
- In Scenario B, the expensive 50,000-token PDF context only needs to be processed (read/computed into attention matrices) exactly *once* by the provider. Successive questions reuse this exact cached state, typically resulting in a ~50% to 90% reduction in API input costs and latency per follow-up question.
- Scenarios A and C feature entirely unique, non-repeating contexts per API query, meaning the cache will continuously fault, providing zero economic or speed optimization.

---

## Question 3 (Model Selection)

A developer intends to fine-tune a massive foundational LLM on a highly specific domain corpus (e.g., proprietary financial data) but currently lacks the massive compute budget required for standard full-parameter fine-tuning. They employ a technique that involves freezing the original pre-trained model weights, injecting small rank-decomposition matrices, and aggressively quantizing the frozen weights to 4-bit precision to fit the entire operation into a single mid-tier GPU.

What specific highly efficient optimization optimization algorithm is the developer employing?

A) Direct Preference Optimization (DPO)
B) Retrieval-Augmented Generation (RAG)
C) QLoRA (Quantized Low-Rank Adaptation)
D) Contextual Model Routing

<div style="page-break-after: always"></div>

### Topics to know to answer Question 3
- [[../study/Module 5 - Fine-Tuning & Preference Optimization.md#5.2 QLoRA for Parameter-Efficient Training|QLoRA for Parameter-Efficient Training]]
- [[../study/Module 3 - Model Landscape & Cost Engineering.md#3.2 Quantization Techniques & KV-Cache|Quantization Techniques]]

<div style="page-break-after: always"></div>

### Solution 3

**Correct Answer: C**

**Explanation:**
Full fine-tuning requires updating billions of model parameters in VRAM simultaneously, demanding extreme clusters of specialized GPUs. **LoRA (Low-Rank Adaptation)** solves this by freezing the original foundation weights and only training tiny embedded rank matrices, heavily compressing the memory requirement. **QLoRA** takes this a step further by mathematically Quantizing the frozen foundational weights (down to 4-bits) during the training phase. This revolutionary technique democratizes AI by allowing developers to functionally fine-tune massive parameter open-source models using a single commodity or consumer-grade GPU.
