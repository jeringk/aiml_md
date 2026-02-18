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

### Machine Learning Approaches

-   Treat sentiment analysis as a standard text classification problem.
-   **Features**: N-grams (unigrams, bigrams), TF-IDF vectors, Part-of-Speech tags.
-   **Classifiers**: Naive Bayes, Support Vector Machines (SVM), Logistic Regression.

### Naive Bayes for Sentiment Classification

For a document $D$ and class $c$, Naive Bayes uses:

$$
\text{Score}(c\mid D)\propto P(c)\prod_{k} P(w_k\mid c)
$$

where:
- $P(c)$: class prior
- $P(w_k\mid c)$: likelihood of token $w_k$ under class $c$

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
