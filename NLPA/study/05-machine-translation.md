# 05. Machine Translation

## Statistical Machine Translation (SMT)

Using statistical models to translate text from a Source Language ($S$) to a Target Language ($T$).

### Noisy Channel Model for MT

We want to find the best target sentence $\hat{T}$ that maximizes $P(T\|S)$.

$$ \hat{T} = \operatorname*{argmax}_T P(T\|S) = \operatorname*{argmax}_T P(S\|T) P(T) $$

-   $P(S\|T)$: **Translation Model** - "Faithfulness" (How well does $T$ capture the meaning of $S$?). Learned from parallel corpora.
-   $P(T)$: **Language Model** - "Fluency" (Is $T$ valid grammatical target language?). Learned from monolingual corpora.

### N-gram Language Model in SMT

SMT decoders commonly use an n-gram LM in log-linear scoring.

$$
P(T)=\prod_{i=1}^{m} P(t_i\mid t_{i-n+1},\dots,t_{i-1})
$$

where:
- $T=(t_1,\dots,t_m)$: target sentence tokens
- $m$: target sentence length
- $n$: n-gram order
- $P(t_i\mid\cdot)$: LM next-token probability

### Log-Linear SMT Decoder Objective

Modern phrase-based SMT uses weighted feature functions:

$$
\hat{T}=\operatorname*{argmax}_T \sum_{k=1}^{K}\lambda_k h_k(S,T)
$$

where:
- $h_k(S,T)$: feature function (translation probs, LM score, distortion, word penalty)
- $\lambda_k$: tuned weight for feature $k$
- $K$: number of features

This framework allows balancing adequacy and fluency with tunable weights.

### Alignment

The process of identifying which words in the source sentence correspond to which words in the target sentence.
-   **IBM Models**: A series of probabilistic models for word alignment.

#### IBM Model 1 and EM Training

IBM Model 1 estimates lexical translation probabilities with EM.

$$
t(f\|e)=\frac{c(f,e)}{c(e)}
$$

where:
- $f$: target-language word
- $e$: source-language word
- $c(f,e)$: expected count of pair $(f,e)$ from E-step
- $c(e)$: expected count of source word $e$

E-step posterior alignment for target word position $j$:

$$
P(a_j=i\mid f_j,\mathbf{e})=\frac{t(f_j\|e_i)}{\sum_{i'} t(f_j\|e_{i'})}
$$

where:
- $a_j$: aligned source position for target token $f_j$
- $\mathbf{e}$: source sentence token sequence
- $i$: source position index

### Phrase-Based SMT (PB-SMT)

PB-SMT translates contiguous phrases instead of isolated words.

Core score components:
- phrase translation probabilities
- lexical weighting
- language model score
- reordering/distortion penalty

Distortion penalty is often exponential in jump distance:

$$
d(\Delta)=\exp(-\alpha\|\Delta\|)
$$

where:
- $\Delta$: jump between consecutive translated phrase positions
- $\alpha>0$: distortion strength

#### Uniform Initialization Effect

When all initial $t(f\|e)$ values are equal, first-iteration posteriors become uniform over candidate source positions.
This is why the first E-step often gives equal alignment probability to each source token.

#### HMM Alignment and Locality Penalty

HMM alignment adds transition probability across alignment positions, unlike IBM Model 1.

A common locality penalty form is:

$$
s(d)=\alpha^{\|d\|}
$$

where:
- $d$: jump distance between positions
- $\alpha\in(0,1)$: locality decay factor

Larger jumps receive smaller probability mass.

---

## Neural Machine Translation (NMT)

Using deep neural networks to perform translation end-to-end.

### Encoder-Decoder Architecture (Seq2Seq)

1.  **Encoder**: Reads the input sentence $S$ and compresses it into a fixed-length context vector (hidden state).
    -   Usually RNNs (LSTM/GRU) or Transformers.
2.  **Decoder**: Generates the output sentence $T$ word-by-word, conditioned on the context vector.

Teacher-forced training objective:

$$
\mathcal{L}_{\text{NMT}}=-\sum_{i=1}^{m}\log P(t_i\mid t_{<i}, S;\theta)
$$

where:
- $\mathcal{L}_{\text{NMT}}$: negative log-likelihood loss
- $t_{<i}$: generated/prefix target tokens before step $i$
- $S$: source sentence
- $\theta$: model parameters

### Attention Mechanism

Solves the bottleneck problem of fixed-length context vectors.
-   Allows the decoder to "look back" at specific parts of the source sentence relevant to the current word being generated.
-   Computes a weighted sum of encoder hidden states (context vector) for each decoding step.

$$ c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j $$
where:
- $c_i$: context vector at decoder step $i$
- $h_j$: encoder hidden state at source step $j$
- $T_x$: source sentence length
- $\alpha_{ij}$: attention weight from decoder step $i$ to source step $j$

Scaled dot-product attention in Transformers:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

where:
- $Q$: query matrix
- $K$: key matrix
- $V$: value matrix
- $d_k$: key dimension

#### Attention Entropy and Alignment Confidence

Attention concentration can be summarized with entropy:

$$
H=-\sum_i p_i\log_2 p_i
$$

where $p_i$ are attention weights over source tokens.
Lower entropy indicates more concentrated attention.

## Evaluation of MT Quality BLEU

BLEU compares machine translation output against one or more reference translations using modified n-gram precision and brevity penalty.

Higher BLEU generally indicates better translation quality at corpus level.

$$
\operatorname{BLEU}=\operatorname{BP}\cdot \exp\left(\sum_{n=1}^{N} w_n\log p_n\right)
$$

where:
- $\operatorname{BP}$: brevity penalty
- $p_n$: modified precision for n-grams of order $n$
- $w_n$: weight for order $n$ (often uniform, $1/N$)
- $N$: maximum n-gram order (typically $4$)

$$
\operatorname{BP}=
\begin{cases}
1, & c>r \\
\exp(1-r/c), & c\le r
\end{cases}
$$

where:
- $c$: candidate translation length
- $r$: effective reference length

### Short BLEU Example

Candidate: `the cat is on mat`  
Reference: `the cat is on the mat`

- Unigram clipped matches = $5$, candidate unigrams = $5$ $\Rightarrow p_1=1.0$
- Candidate is shorter ($c=5$, $r=6$), so $\operatorname{BP}<1$
- Final BLEU reduces because of brevity penalty and higher-order n-gram mismatches

## Domain Shift in MT

Model quality often drops when test-domain data differs from training data (news vs social media vs technical text).
This is observed as BLEU degradation on out-of-domain content.

---

## Indic Language Translation

Translation involving Indian languages (e.g., Hindi, Tamil, Bengali).

### Challenges

1.  **Low Resource**: Lack of large-scale parallel corpora compared to English-French/German.
2.  **Morphological Richness**: Indian languages are highly agglutinative and morphologically rich.
3.  **Script Diversity**: Many different scripts (Devanagari, Dravidian, etc.).
4.  **Code-Mixing**: Frequent mixing of English and local languages in daily usage (Hinglish).

### OOV and Code-Mixing in Indic MT

Out-of-vocabulary (OOV) rates typically increase with code-mixed inputs because mixed-script and mixed-language forms are less frequent in parallel corpora.

A relative increase calculation is:

$$
\text{new OOV}=\text{base OOV}\times(1+r)
$$

where $r$ is the fractional increase due to code-mixing.

### Approaches

-   **Multilingual Models**: Training a single model on many languages to enable transfer learning (e.g., mBART, IndicTrans).
-   **Back-Translation**: Generating synthetic parallel data using monolingual data.
-   **Transliteration**: Mapping scripts to a common space for processing.

### Back-Translation Pipeline (Exam-Focused)

1. Train reverse model $T\rightarrow S$ using available parallel data.
2. Translate target-language monolingual corpus into synthetic source sentences.
3. Combine synthetic parallel pairs with real parallel data.
4. Retrain forward model $S\rightarrow T$ for improved low-resource quality.

### Typical Error Types in Indic MT

- **Morphology errors**: wrong inflections/case markers.
- **Agreement errors**: gender-number-person mismatch.
- **Script/romanization mismatch**: degraded quality on transliterated input.
- **Code-mixed ambiguity**: unstable translations for mixed language tokens.
