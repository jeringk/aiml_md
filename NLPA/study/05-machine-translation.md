# 05. Machine Translation

## Statistical Machine Translation (SMT)

Using statistical models to translate text from a Source Language ($S$) to a Target Language ($T$).

### Noisy Channel Model for MT

We want to find the best target sentence $\hat{T}$ that maximizes $P(T|S)$.

$$ \hat{T} = \operatorname*{argmax}_T P(T|S) = \operatorname*{argmax}_T P(S|T) P(T) $$

-   $P(S|T)$: **Translation Model** - "Faithfulness" (How well does $T$ capture the meaning of $S$?). Learned from parallel corpora.
-   $P(T)$: **Language Model** - "Fluency" (Is $T$ valid grammatical target language?). Learned from monolingual corpora.

### Alignment

The process of identifying which words in the source sentence correspond to which words in the target sentence.
-   **IBM Models**: A series of probabilistic models for word alignment.

---

## Neural Machine Translation (NMT)

Using deep neural networks to perform translation end-to-end.

### Encoder-Decoder Architecture (Seq2Seq)

1.  **Encoder**: Reads the input sentence $S$ and compresses it into a fixed-length context vector (hidden state).
    -   Usually RNNs (LSTM/GRU) or Transformers.
2.  **Decoder**: Generates the output sentence $T$ word-by-word, conditioned on the context vector.

### Attention Mechanism

Solves the bottleneck problem of fixed-length context vectors.
-   Allows the decoder to "look back" at specific parts of the source sentence relevant to the current word being generated.
-   Computes a weighted sum of encoder hidden states (context vector) for each decoding step.

$$ c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j $$
Where $\alpha_{ij}$ are attention weights.

---

## Indic Language Translation

Translation involving Indian languages (e.g., Hindi, Tamil, Bengali).

### Challenges

1.  **Low Resource**: Lack of large-scale parallel corpora compared to English-French/German.
2.  **Morphological Richness**: Indian languages are highly agglutinative and morphologically rich.
3.  **Script Diversity**: Many different scripts (Devanagari, Dravidian, etc.).
4.  **Code-Mixing**: Frequent mixing of English and local languages in daily usage (Hinglish).

### Approaches

-   **Multilingual Models**: Training a single model on many languages to enable transfer learning (e.g., mBART, IndicTrans).
-   **Back-Translation**: Generating synthetic parallel data using monolingual data.
-   **Transliteration**: Mapping scripts to a common space for processing.
