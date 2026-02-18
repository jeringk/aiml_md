# 06. Information Extraction

## Information Extraction (IE)

The process of turning the unstructured information embedded in texts into structured data (e.g., databases, knowledge graphs).

### Named Entity Recognition (NER)

The task of finding and classifying named entities in text into pre-defined categories (e.g., Person, Organization, Location, Date, Time).

**Example:**
"[Apple Inc.]_ORG is planning to open a new office in [London]_LOC on [next Monday]_DATE."

**Approaches:**
1.  **Rule-based / Regex**: Good for phone numbers, emails, dates.
2.  **Sequence Labeling**:
    -   **HMM (Hidden Markov Models)**: Probabilistic generative model.
    -   **CRF (Conditional Random Fields)**: Discriminative model, often outperforms HMMs by using arbitrary overlapping features.
    -   **Bi-LSTM + CRF**: Neural approach. Bi-LSTM captures context, CRF ensures valid label transitions (e.g., I-ORG cannot follow B-PER).
    -   **Transformers (BERT)**: Fine-tuning BERT for token classification.

### Relation Extraction

The task of identifying relationships between identified entities.
Example: *founded_by(Apple Inc., Steve Jobs)*.

**Approaches:**
1.  **Hand-built Patterns**: "X was founded by Y", "Y, the founder of X". High precision, low recall.
2.  **Supervised Learning**: Train a classifier on annotated data.
3.  **Semi-supervised (Bootstrapping)**: Start with a few seed tuples, find sentences containing them, learn patterns, find new tuples, repeat.
4.  **Distant Supervision**: Use a large database (Freebase, DBPedia) to automatically label a large corpus. If a tuple (X, Y) exists in DB, assume any sentence containing X and Y expresses the relation.

### Extracting Events and Time

**Event Extraction**:
-   Identifying events mentioned in text.
-   Identifying the time and location of the event.
-   Identifying the participants (who did what to whom).

**Temporal Expression Extraction**:
-   Extracting time expressions (absolute like "Jan 1, 2024" or relative like "next Tuesday").
-   Normalizing them to a standard format (ISO 8601).

### Evaluation Metrics for NER and Event Extraction

#### Strict Exact Match

For strict IE evaluation, a prediction is correct only if span and label/type match exactly.
For events/relations, event type and required arguments must all match exactly.

#### Precision, Recall, and F1 for IE

$$
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}
$$

where:
- $TP$: true positives
- $FP$: false positives
- $FN$: false negatives
- $P$: precision
- $R$: recall
