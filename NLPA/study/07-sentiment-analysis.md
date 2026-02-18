# 07. Sentiment Analysis

## Sentiment Analysis (Opinion Mining)

The computational study of people's opinions, sentiments, emotions, and attitudes expressed in written language.

### Levels of Analysis

1.  **Document Level**: Classify the sentiment of an entire document (e.g., a movie review) as positive, negative, or neutral.
2.  **Sentence Level**: Classify individual sentences.
3.  **Aspect Level**: Identify sentiments towards specific aspects of an entity (e.g., "The [camera]_pos is great but the [battery]_neg sucks").

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

### Negation Handling with NOT Tokens

A common heuristic is to convert tokens following negation into negated features (for example, `not good` -> `NOT_good`) until punctuation.
This helps model polarity reversal.

### Decision Rule Using Posterior Scores

Compute class scores and predict the class with the larger score.
Normalization is optional for argmax-based classification.

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

## Rule-Based, ML-Based, and Hybrid Systems

### Rule-Based Systems

- Depend on sentiment lexicons, negation/intensity rules, pattern templates.
- High precision on known linguistic phenomena.
- Weak generalization to domain shift and informal phrasing.

### ML-Based Systems

- Learn sentiment decision boundaries from labeled datasets.
- Better domain adaptability with sufficient data.
- Can be less interpretable than pure rule systems.

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

## NLP Features for Sentiment Analysis

Common feature groups:
- lexical: unigrams, bigrams, char n-grams
- syntactic: POS tags, dependency relations
- semantic: contextual embeddings, sentiment lexicon categories
- discourse/pragmatic: negation scope, intensifiers, contrastive markers (`but`, `however`)

### Feature Vector Form

$$
\mathbf{x}=[x_1,x_2,\dots,x_d]^\top
$$

where:
- $\mathbf{x}$: final feature vector for classifier
- $x_i$: feature value for dimension $i$
- $d$: total number of engineered/learned features

### Short Aspect-Level Example

Sentence: `The display is excellent but the battery life is poor.`

- Aspect `display` -> positive polarity
- Aspect `battery life` -> negative polarity
- Final label can be stored as two aspect-sentiment tuples instead of one document-level label.
