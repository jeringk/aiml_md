# Module 10 — Language Modeling

## Topics

- Motivation and Introduction
- Introduction to Language Models
- A digression into Transformer, word2Vec, BERT, …, GPT

---

## 10.1 Motivation

- Language modeling: **predict the next token** given context
- Foundation for modern NLP (translation, summarization, QA, dialogue)
- Language models are **autoregressive generative models** over tokens:

$$p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t | x_{<t})$$

---

## 10.2 Language Models Overview

### N-gram Models
- Approximate context with last $(n-1)$ words
- Limitations: fixed context, sparsity, no generalization

### Neural Language Models
- Feed-forward (fixed context), RNN/LSTM (variable context), Transformer (attention)

### Evaluation
- **Perplexity**: $PP = \exp\left(-\frac{1}{T}\sum_t \log p(x_t | x_{<t})\right)$ — lower is better

---

## 10.3 word2Vec

- Dense vector representations from co-occurrence
- **CBOW**: context → center word | **Skip-gram**: center word → context
- Semantic arithmetic: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$
- Static embeddings (same vector regardless of context)

---

## 10.4 Transformer

### Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

### Transformer Block
1. Multi-head self-attention + residual + LayerNorm
2. Feed-forward network + residual + LayerNorm
3. Positional encoding (sinusoidal or learned)

---

## 10.5 BERT

- **Encoder-only** transformer — bidirectional attention
- Pre-training: **Masked Language Modeling** (MLM) + **Next Sentence Prediction** (NSP)
- Contextual embeddings — fine-tune with task-specific heads
- Best for: classification, NER, QA (understanding tasks)

---

## 10.6 GPT

- **Decoder-only** transformer — causal (left-to-right) attention
- Pre-training: next token prediction
- GPT-1 (117M) → GPT-2 (1.5B) → GPT-3 (175B) → GPT-4 (~1.8T)
- In-context learning, few-shot prompting
- Best for: generation tasks

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder | Decoder |
| Attention | Bidirectional | Causal |
| Best for | Understanding | Generation |

---

## References

- T2: Goodfellow et al., Ch. 12 — Applications (NLP)
- Vaswani et al., "Attention Is All You Need" (2017)
