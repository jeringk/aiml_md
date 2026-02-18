# Module 10 — Language Modeling

## Topics

- [[#10.1 Motivation|Motivation and Introduction]]
- [[#10.2 Language Models Overview|Introduction to Language Models]]
- A digression into [[#10.4 Transformer Architecture|Transformer]], [[#10.3 Word Embeddings|word2Vec]], [[#10.5 BERT (Bidirectional Encoder Representations from Transformers)|BERT]], [[#10.6 GPT (Generative Pre-trained Transformer)|GPT]]

---

## 10.1 Motivation

- Language modeling: **predict the next token** given context
- Foundation for modern NLP (translation, summarization, QA, dialogue)
- Language models are **autoregressive generative models** over tokens:

$$p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \| x_{<t})$$

---

## 10.2 Language Models Overview

### N-gram Models
- Approximate context with last $(n-1)$ words
- Limitations: fixed context, sparsity, no generalization

### Neural Language Models
- Feed-forward (fixed context), RNN/LSTM (variable context), Transformer (attention)

### Evaluation
- **Perplexity**: $PP = \exp\left(-\frac{1}{T}\sum_t \log p(x_t \| x_{<t})\right)$ — lower is better

---

## 10.3 word2Vec

- Dense vector representations learned from co-occurrence patterns
- **Static embeddings** — same vector regardless of context
- Semantic arithmetic: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$

### CBOW (Continuous Bag of Words)

- **Input:** context words (surrounding words within a window) → **Output:** center word
- Averages the context word embeddings, then predicts the center word
- One training sample per center word position

### Skip-gram

- **Input:** center word → **Output:** each context word separately
- One training sample per (center word, context word) pair
- Typically generates more training samples than CBOW for the same corpus

### Architecture & Parameters

Both CBOW and Skip-gram use two weight matrices:

| Matrix | Shape | Role |
|--------|-------|------|
| $W_{in}$ (input embeddings) | $V \times d$ | Lookup table for input words |
| $W_{out}$ (output embeddings) | $d \times V$ | Projection to vocabulary for prediction |

- **Total trainable parameters** = $2 \times V \times d$
- $V$ = vocabulary size, $d$ = embedding dimension
- Both architectures have the **same number of parameters**

### Training Set Construction

- **Window size $k$**: consider $\lfloor k/2 \rfloor$ words on each side of the center word
- **CBOW training set size** = number of center word positions in the corpus (one sample per position)
- **Skip-gram training set size** = total number of (center, context) pairs (one sample per pair)
- **Boundary effects**: words near the edges of a sentence have fewer context neighbors, reducing the number of valid pairs

### Training Objectives

- **Skip-gram with Negative Sampling (SGNS)**:

$$J = -\log \sigma(u_{w_O}^T v_{w_I}) - \sum_{k=1}^{K} \log \sigma(-u_{w_k}^T v_{w_I})$$

- **Hierarchical Softmax**: replaces full softmax with binary tree traversal — $O(\log V)$ instead of $O(V)$

---

### 10.3.1 GloVe

- **GloVe (Global Vectors for Word Representation)** — Pennington et al., 2014
- Combines **count-based** (co-occurrence matrix) and **predictive** (neural) approaches
- Builds a global **co-occurrence matrix** $X$ where $X_{ij}$ = number of times word $j$ appears in context of word $i$

**Objective Function:**

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

where $f(x)$ is a weighting function that caps very frequent co-occurrences:

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

### GloVe vs word2Vec

| Property | word2Vec | GloVe |
|----------|----------|-------|
| **Training signal** | Local context windows | Global co-occurrence statistics |
| **Objective** | Predictive (softmax/negative sampling) | Weighted least squares on log co-occurrence |
| **Captures** | Local patterns | Both local and global patterns |
| **Embeddings** | Static | Static |

---

### 10.3.2 CoVe (Contextualized Word Vectors)

- **CoVe** — McCann et al., 2017
- Produces **contextualized embeddings** — the same word gets **different** representations depending on its surrounding sentence context
- Uses a **BiLSTM** encoder trained on machine translation (English→German) as the feature extractor

### BiLSTM Architecture for CoVe

Given input embedding $\mathbf{x}_t$:

**Forward LSTM:**
$$h_t^f = \text{LSTM}_f(\mathbf{x}_t, h_{t-1}^f)$$

**Backward LSTM:**
$$h_t^b = \text{LSTM}_b(\mathbf{x}_t, h_{t+1}^b)$$

**CoVe computation** (via summation):
$$\text{CoVe}(\mathbf{x}_t) = h_t^f + h_t^b$$

Or via concatenation: $\text{CoVe}(\mathbf{x}_t) = [h_t^f; h_t^b]$

- Simplified single-layer case (exam-style): $h_f = \tanh(W_f \mathbf{x} + b_f)$, $h_b = \tanh(W_b \mathbf{x} + b_b)$

### CoVe vs Static Embeddings

| Property | word2Vec / GloVe | CoVe |
|----------|-----------------|------|
| **Type** | Static (context-independent) | Contextualized (context-dependent) |
| **Polysemy** | Same vector for "bank" (river) and "bank" (financial) | Different vectors based on sentence |
| **Architecture** | Lookup table | BiLSTM encoder |
| **Pre-training** | Unsupervised (co-occurrence) | Supervised (machine translation) |

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
