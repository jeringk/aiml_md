# 07. Sentiment Analysis

## Sentiment Analysis (Opinion Mining)

The computational study of people's opinions, sentiments, emotions, and attitudes expressed in written language.

### Levels of Analysis

1.  **Document Level**: Classify the sentiment of an entire document (e.g., a movie review) as positive, negative, or neutral.
2.  **Sentence Level**: Classify individual sentences.
3.  **Aspect Level**: Identify sentiments towards specific aspects of an entity (e.g., "The [camera]_pos is great but the [battery]_neg sucks").

### Numerical Illustration Across Levels

Review: `The camera is excellent, but battery life is poor.`

- Document-level output: overall score = $-0.1$ -> Negative/Neutral boundary case
- Sentence-level outputs:
  - Sentence 1 score = $+0.8$
  - Sentence 2 score = $-0.9$
- Aspect-level outputs:
  - `camera` = $+0.9$
  - `battery life` = $-0.95$

Aspect-level analysis is most informative when opposing sentiments occur in one review.

## Sentiment Analysis Methods

### Lexicon-based Approaches

-   Use a dictionary of sentiment words (lexicon) where each word has a sentiment score.
-   **Algorithm**: Sum the scores of words in the text. Handle negations (e.g., "not good" flips the score).
-   **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A rule-based model tuned for social media text (handles emojis, capitalization, slang).

Lexicon polarity score:

$$
\operatorname{Polarity}(D)=\sum_{i=1}^{n} s(w_i)\cdot m_i
$$

where:
- $D$: document/sentence
- $w_i$: token at position $i$
- $s(w_i)$: lexicon sentiment score for token $w_i$
- $m_i$: modifier factor (negation/intensifier/diminisher)
- $n$: number of tokens considered

### Numerical Example: Lexicon-Based Score

Sentence: `The phone is very good but not durable.`

Assume:
- $s(\text{good})=+2.0$
- $s(\text{durable})=+1.5$
- intensifier `very` gives factor $1.5$
- negation on `durable` gives factor $-1$

Then:
- contribution of `very good` = $2.0\times1.5=3.0$
- contribution of `not durable` = $1.5\times(-1)=-1.5$

Total:

$$
\operatorname{Polarity}(D)=3.0+(-1.5)=+1.5
$$

So final polarity is positive.

### Machine Learning Approaches

-   Treat sentiment analysis as a standard text classification problem.
-   **Features**: N-grams (unigrams, bigrams), TF-IDF vectors, Part-of-Speech tags.
-   **Classifiers**: Naive Bayes, Support Vector Machines (SVM), Logistic Regression.

### TF-IDF Feature Weight

$$
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\cdot \log\frac{N}{\operatorname{df}(t)}
$$

where:
- $t$: term/token
- $d$: document
- $\operatorname{tf}(t,d)$: term frequency in $d$
- $N$: number of documents in corpus
- $\operatorname{df}(t)$: document frequency of term $t$

### Numerical Example: TF-IDF

Assume:
- corpus size $N=1000$
- token `excellent` appears in $\operatorname{df}=50$ documents
- in review $d$, `excellent` appears $3$ times

Then:

$$
\operatorname{tfidf}(\text{excellent},d)=3\cdot\log\frac{1000}{50}
=3\cdot\log(20)
$$

Using natural log, $\log(20)\approx2.996$:

$$
\operatorname{tfidf}\approx 8.99
$$

### Naive Bayes for Sentiment Classification

For a document $D$ and class $c$, Naive Bayes uses:

$$
\text{Score}(c\mid D)\propto P(c)\prod_{k} P(w_k\mid c)
$$

where:
- $P(c)$: class prior
- $P(w_k\mid c)$: likelihood of token $w_k$ under class $c$

With Laplace smoothing:

$$
P(w\mid c)=\frac{\operatorname{count}(w,c)+\alpha}{\sum_{w'}\operatorname{count}(w',c)+\alpha\|V\|}
$$

where:
- $\alpha>0$: smoothing constant
- $V$: vocabulary set
- $\|V\|$: vocabulary size

### Numerical Example: Naive Bayes Sentiment

Document tokens: `NOT_good`, `movie`

Given:
- $P(+)=0.4,\;P(-)=0.6$
- $P(\text{NOT\_good}\mid +)=0.01,\;P(\text{movie}\mid +)=0.05$
- $P(\text{NOT\_good}\mid -)=0.07,\;P(\text{movie}\mid -)=0.04$

Scores:

$$
\operatorname{Score}(+\mid D)=0.4\times0.01\times0.05=0.0002
$$

$$
\operatorname{Score}(-\mid D)=0.6\times0.07\times0.04=0.00168
$$

Since $0.00168>0.0002$, predicted sentiment is Negative.

### Negation Handling with NOT Tokens

A common heuristic is to convert tokens following negation into negated features (for example, `not good` -> `NOT_good`) until punctuation.
This helps model polarity reversal.

### Decision Rule Using Posterior Scores

Compute class scores and predict the class with the larger score.
Normalization is optional for argmax-based classification.

In practice, log-scores are used for numerical stability:

$$
\log \operatorname{Score}(c\mid D)=\log P(c)+\sum_k \log P(w_k\mid c)
$$

where:
- $\log \operatorname{Score}(c\mid D)$: log posterior score for class $c$

Argmax is unchanged by log transformation.

## Neural Networks for Sentiment Analysis

1.  **RNNs / LSTMs**:
    -   Process text sequentially. The final hidden state captures the sentiment of the sentence/doc.
    -   Good for capturing long-distance dependencies and compositionality.
2.  **CNNs (Convolutional Neural Networks)**:
    -   Use 1D convolutions over word embeddings to capture local n-gram features.
    -   Effective for detecting key phrases indicative of sentiment.
3.  **Transformers (BERT, RoBERTa)**:
    -   Fine-tune pre-trained language models for sequence classification.
    -   Currently the state-of-the-art approach.
    -   Contextual embeddings allow distinguishing "good" in "good movie" vs "good grief".

### Neural Sentiment Classifier Objective

For representation $\mathbf{h}$ and class logits $z=W\mathbf{h}+b$:

$$
P(c\mid D)=\operatorname{softmax}(z)_c
$$

Cross-entropy loss:

$$
\mathcal{L}=-\sum_{c} y_c \log P(c\mid D)
$$

where:
- $D$: input text
- $\mathbf{h}$: encoder output representation
- $y_c$: one-hot ground-truth for class $c$
- $W,b$: classifier parameters

### Numerical Example: Softmax and Cross-Entropy

Suppose logits for classes `[Positive, Negative, Neutral]` are:

$$
z=[1.2,\;2.0,\;0.3]
$$

Softmax probabilities (approx):

$$
P=[0.269,\;0.598,\;0.133]
$$

If true class is Negative, then:

$$
\mathcal{L}=-\log(0.598)\approx0.514
$$

Lower loss indicates better confidence on the correct class.

## Rule-Based, ML-Based, and Hybrid Systems

### Rule-Based Systems

- Depend on sentiment lexicons, negation/intensity rules, pattern templates.
- High precision on known linguistic phenomena.
- Weak generalization to domain shift and informal phrasing.

Example:
- Rule: if token contains `not` before positive word, flip polarity.
- Text: `not useful` -> negative by rule even without training data.

### ML-Based Systems

- Learn sentiment decision boundaries from labeled datasets.
- Better domain adaptability with sufficient data.
- Can be less interpretable than pure rule systems.

Example:
- Logistic regression learns weight(`excellent`) = $+1.4$, weight(`worst`) = $-1.8$.
- Review score is weighted sum of active features.

### Hybrid Systems

Combine hand-crafted rules with learned models.

Hybrid score:

$$
\operatorname{score}_{\text{hybrid}}=\lambda\,\operatorname{score}_{\text{ML}}+(1-\lambda)\,\operatorname{score}_{\text{rule}}
$$

where:
- $\operatorname{score}_{\text{ML}}$: model-predicted polarity score
- $\operatorname{score}_{\text{rule}}$: lexicon/rule-based score
- $\lambda\in[0,1]$: interpolation weight

### Numerical Example: Hybrid Combination

Assume:
- $\operatorname{score}_{\text{ML}}=-0.70$
- $\operatorname{score}_{\text{rule}}=-0.20$
- $\lambda=0.6$

Then:

$$
\operatorname{score}_{\text{hybrid}}=0.6(-0.70)+0.4(-0.20)=-0.50
$$

Final prediction remains negative, with stronger influence from ML model.

## Sentiment Evaluation Metrics

For binary sentiment:

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}
$$

where:
- $TP$: true positives
- $TN$: true negatives
- $FP$: false positives
- $FN$: false negatives
- $P$: precision
- $R$: recall

### Numerical Example: Metrics

Given confusion counts:
- $TP=42,\;TN=38,\;FP=8,\;FN=12$

$$
\text{Accuracy}=\frac{42+38}{100}=0.80
$$

$$
\text{Precision}=\frac{42}{42+8}=0.84,\quad
\text{Recall}=\frac{42}{42+12}=0.7778
$$

$$
F1=\frac{2\times0.84\times0.7778}{0.84+0.7778}\approx0.8077
$$

## NLP Features for Sentiment Analysis

Common feature groups:
- lexical: unigrams, bigrams, char n-grams
- syntactic: POS tags, dependency relations
- semantic: contextual embeddings, sentiment lexicon categories
- discourse/pragmatic: negation scope, intensifiers, contrastive markers (`but`, `however`)

### 1) Lexical Features (Surface Word Signals)

Lexical features capture direct polarity words and short phrases.

Example sentence: `The movie is not good at all.`

- Unigrams: `movie`, `not`, `good`
- Bigrams: `not good`, `at all`
- Character n-grams (for robustness): `goo`, `ood` from `good`

If negation transformation is used:
- `not good` -> `NOT_good`
- Model learns `NOT_good` as strong negative signal.

### 2) Syntactic Features (Structure-Aware Signals)

Syntactic features use grammar and dependencies to reduce ambiguity.

Example sentence: `The camera is light but feels cheap.`

- POS patterns:
  - `light` as adjective (`JJ`) linked to `camera`
  - `cheap` as adjective (`JJ`) linked to omitted subject (`camera`)
- Dependency relations:
  - `amod(camera, light)` (positive aspect clue)
  - `xcomp(feels, cheap)` (negative quality clue)

These features help aspect-level sentiment when multiple opinions appear in one sentence.

### 3) Semantic Features (Meaning-Level Signals)

Semantic features generalize beyond exact words.

Example:
- Words `excellent`, `great`, `fantastic` often map to similar embedding regions (positive sentiment cluster).
- Words `awful`, `terrible`, `worst` map to negative regions.

A contextual model can disambiguate:
- `The battery is light.` -> likely positive (weight)
- `The plot is light.` -> may be neutral/negative depending on domain context

### 4) Discourse and Pragmatic Features

Discourse markers often flip overall interpretation.

Example sentence: `The screen is sharp, but the battery drains quickly.`

- Clause-1 sentiment: positive
- Clause-2 sentiment: negative
- Marker `but` usually gives higher decision weight to the clause after it.

So document-level prediction is often negative despite initial positive words.

### Feature Vector Form

$$
\mathbf{x}=[x_1,x_2,\dots,x_d]^\top
$$

where:
- $\mathbf{x}$: final feature vector for classifier
- $x_i$: feature value for dimension $i$
- $d$: total number of engineered/learned features

### Example: Building a Combined Feature Vector

Sentence: `I thought the phone looked premium, but performance is poor.`

Possible selected features:
- lexical: `premium` = 1, `poor` = 1, `but` = 1
- bigram: `performance poor` = 1
- syntactic: `amod(phone, premium)` = 1
- discourse: `has_contrast_marker` = 1
- semantic: embedding vector for full sentence

A hybrid feature representation:

$$
\mathbf{x}=[\mathbf{x}_{\text{tfidf}};\mathbf{x}_{\text{syntax}};\mathbf{x}_{\text{discourse}};\mathbf{x}_{\text{embed}}]
$$

where:
- $\mathbf{x}_{\text{tfidf}}$: sparse lexical vector
- $\mathbf{x}_{\text{syntax}}$: dependency/POS indicators
- $\mathbf{x}_{\text{discourse}}$: negation/contrast/intensity indicators
- $\mathbf{x}_{\text{embed}}$: dense semantic embedding

### Feature Design Tips for Exams and Practice

1. Start with lexical baseline (unigram + bigram TF-IDF).
2. Add negation and contrast-marker features first (high impact, low complexity).
3. Add aspect-aware syntactic features for multi-aspect sentences.
4. Use contextual embeddings when domain vocabulary is varied.

### Short Aspect-Level Example

Sentence: `The display is excellent but the battery life is poor.`

- Aspect `display` -> positive polarity
- Aspect `battery life` -> negative polarity
- Final label can be stored as two aspect-sentiment tuples instead of one document-level label.
