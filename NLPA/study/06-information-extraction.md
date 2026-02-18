# 06. Information Extraction

## Information Extraction (IE)

The process of turning the unstructured information embedded in texts into structured data (e.g., databases, knowledge graphs).

### Typical IE Pipeline

1. Text preprocessing (sentence split, tokenization, normalization).
2. NER to detect entity mentions and types.
3. Relation extraction across entity pairs.
4. Event extraction with triggers and arguments.
5. Temporal extraction and normalization.
6. Storage as structured tuples/triples.

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

### Sequence Labeling with BIO Tags

Common tag set:
- `B-X`: beginning of entity type $X$
- `I-X`: inside entity type $X$
- `O`: outside any entity

Example:
- Tokens: `Apple Inc. opened office in London`
- Tags: `B-ORG I-ORG O O O B-LOC`

### CRF Sequence Score

For token sequence $\mathbf{x}=(x_1,\dots,x_n)$ and tag sequence $\mathbf{y}=(y_1,\dots,y_n)$:

$$
\operatorname{score}(\mathbf{x},\mathbf{y})=\sum_{i=1}^{n}\left(A_{y_{i-1},y_i}+P_{i,y_i}\right)
$$

where:
- $A_{u,v}$: transition score from tag $u$ to $v$
- $P_{i,y_i}$: emission score for assigning tag $y_i$ at position $i$
- $n$: sequence length

Conditional probability:

$$
P(\mathbf{y}\mid\mathbf{x})=\frac{\exp(\operatorname{score}(\mathbf{x},\mathbf{y}))}{\sum_{\mathbf{y'}\in\mathcal{Y}(\mathbf{x})}\exp(\operatorname{score}(\mathbf{x},\mathbf{y'}))}
$$

where:
- $\mathcal{Y}(\mathbf{x})$: all valid tag sequences for $\mathbf{x}$

Best decoding uses Viterbi:

$$
\hat{\mathbf{y}}=\operatorname*{argmax}_{\mathbf{y}} \operatorname{score}(\mathbf{x},\mathbf{y})
$$

### Relation Extraction

The task of identifying relationships between identified entities.
Example: *founded_by(Apple Inc., Steve Jobs)*.

**Approaches:**
1.  **Hand-built Patterns**: "X was founded by Y", "Y, the founder of X". High precision, low recall.
2.  **Supervised Learning**: Train a classifier on annotated data.
3.  **Semi-supervised (Bootstrapping)**: Start with a few seed tuples, find sentences containing them, learn patterns, find new tuples, repeat.
4.  **Distant Supervision**: Use a large database (Freebase, DBPedia) to automatically label a large corpus. If a tuple (X, Y) exists in DB, assume any sentence containing X and Y expresses the relation.

### Relation Classification Formulation

Given sentence representation $\mathbf{h}$ for entity pair $(e_1,e_2)$:

$$
P(r\mid \mathbf{h})=\operatorname{softmax}(W\mathbf{h}+b)
$$

where:
- $r$: relation label (including `No-Relation`)
- $W,b$: trainable classifier parameters
- $\mathbf{h}$: contextual feature vector from encoder

Prediction:

$$
\hat{r}=\operatorname*{argmax}_{r} P(r\mid \mathbf{h})
$$

### Relation Extraction Mini Example

Sentence: `Steve Jobs founded Apple in 1976.`  
Entities: `Steve Jobs` (PER), `Apple` (ORG), `1976` (DATE)

- Candidate pair: (`Steve Jobs`, `Apple`)
- Predicted relation: `founded_by(Apple, Steve Jobs)`
- Additional temporal argument can be stored for event timeline.

### Extracting Events and Time

**Event Extraction**:
-   Identifying events mentioned in text.
-   Identifying the time and location of the event.
-   Identifying the participants (who did what to whom).

**Temporal Expression Extraction**:
-   Extracting time expressions (absolute like "Jan 1, 2024" or relative like "next Tuesday").
-   Normalizing them to a standard format (ISO 8601).

### Event Extraction Schema

Event tuple can be represented as:

$$
\text{Event}=(\text{trigger},\text{type},\text{arguments},\text{time},\text{location})
$$

where:
- `trigger`: lexical cue (e.g., `acquired`, `launched`)
- `type`: event category (e.g., Acquisition, Attack, Meeting)
- `arguments`: role-labeled entities (Agent, Target, Buyer, Seller)
- `time`: normalized temporal value
- `location`: resolved place mention

### Time Normalization for Relative Expressions

Given document creation time (DCT) $t_0$, relative phrase offset $\Delta$:

$$
t_{\text{norm}} = t_0 + \Delta
$$

where:
- $t_{\text{norm}}$: normalized timestamp/date
- $t_0$: reference date-time (document time)
- $\Delta$: relative shift (for example, `+2 days`, `-1 week`)

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

### Micro vs Macro F1

Micro-F1 aggregates counts across all classes before computing F1, favoring frequent classes.
Macro-F1 computes per-class F1 and averages them, highlighting minority-class performance.
