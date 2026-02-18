# 02. Grammar and Spellcheckers

## Spell Checking

Spell checking is the process of detecting and correcting misspelled words in text.

### Types of Errors

1.  **Non-word Errors**: The string is not a valid word in the dictionary (e.g., "graffe" for "giraffe").
    -   *Detection*: Check if the word exists in a dictionary.
    -   *Correction*: Generate candidates using edit distance and rank them.
2.  **Real-word Errors**: The string is a valid word but incorrect in context (e.g., "three minutes" vs "there minutes").
    -   *Detection / Correction*: Require context-aware methods (e.g., N-grams, Noisy Channel Model).

### Edit Distance (Levenshtein Distance)

The minimum number of editing operations (insertion, deletion, substitution) needed to transform one string into another.

**Definition:**
Let $X$ be a source string of length $n$ and $Y$ be a target string of length $m$. $D(i, j)$ is the edit distance between $X[1..i]$ and $Y[1..j]$.

$$
D(i,j) = \min \begin{cases} 
D(i-1, j) + 1 & \text{(deletion)} \\
D(i, j-1) + 1 & \text{(insertion)} \\
D(i-1, j-1) + \mathbb{I}(X[i] \neq Y[j]) & \text{(substitution)}
\end{cases}
$$

**Base Cases:**
- $D(i, 0) = i$ (delete all characters from source)
- $D(0, j) = j$ (insert all characters into target)

### Noisy Channel Model

A probabilistic approach to spell correction. We want to find the intended word $w$ given the observed (misspelled) word $x$.

$$ \hat{w} = \operatorname*{argmax}_{w \in V} P(w\|x) $$

Using Bayes' Theorem:

$$ \hat{w} = \operatorname*{argmax}_{w \in V} \frac{P(x\|w) P(w)}{P(x)} $$
$$ \hat{w} = \operatorname*{argmax}_{w \in V} P(x\|w) P(w) $$

-   $P(x|w)$: **Error Model** (Likelihood) - Probability of typing $x$ when intending $w$ (based on edit distance or confusion matrices).
-   $P(w)$: **Language Model** (Prior) - Probability of word $w$ appearing (frequency or n-gram probability).

---

## Grammar Checking

Grammar checking involves detecting syntactic errors and ensuring grammatical correctness.

### Common Grammatical Errors

-   **Agreement Errors**: Subject-verb agreement (e.g., "He *go* to school").
-   **Tense Errors**: Incorrect verb tense usage.
-   **Preposition Errors**: Wrong preposition choice.

### Techniques

1.  **Rule-Based Approaches**:
    -   Manually defined rules (features) for common errors.
    -   Example: If subject is singular third-person, verb must end in 's'.
2.  **Statistical / Probabilistic Parsing**:
    -   Use a parser to generate a parse tree.
    -   Errors often result in low-probability trees or failure to parse.
3.  **Neural Approaches**:
    -   Treat grammar correction as a **Sequence-to-Sequence (Seq2Seq)** translation task.
    -   Input: Incorrect sentence.
    -   Output: Corrected sentence.
    -   Training data: Pairs of (incorrect, correct) sentences.

### Parsing

**Constituency Parsing**: Breaks a sentence into sub-phrases (NP, VP).
-   *Context-Free Grammars (CFG)*: $A \to \alpha$.
-   *CKY Algorithm*: Bottom-up dynamic programming for parsing.

**Dependency Parsing**: Identifies grammatical relationships between words (Head-Dependent).
-   Subject, Object, Modifier relationships.
