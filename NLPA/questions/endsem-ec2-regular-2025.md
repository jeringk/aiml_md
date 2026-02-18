# NLPA — End-Semester Test (EC-2 Regular) 2024-2025

**Course No.:** AIML* ZG519 | **Course Title:** Natural Language Processing Applications  
**Nature:** Open Book | **Weightage:** 30% | **Date:** 7 September 2025, 9:00 AM

---

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. IBM Model 1 and HMM Word Alignment (English-Hindi)

**Marks:** [3+3=6] | **Source:** End-Sem EC-2 Regular 2024-25

A Statistical Machine Translation system is being developed for English-Hindi translation using IBM Model 1. You are given the following parallel corpus and need to perform word alignment analysis.

Parallel Corpus:

1. English: "The cat sits" -> Hindi: "बिल्ली बैठती है"

2. English: "The dog runs" -> Hindi: "कुत्ता दौड़ता है"

3. English: "Cat runs fast" -> Hindi: "बिल्ली तेज दौड़ती है"

Initial uniform translation probabilities:

- P(बिल्ली|cat)=P(बिल्ली|the)=P(बिल्ली|sits)=P(बिल्ली|dog)=P(बिल्ली|runs)=P(बिल्ली|fast)=1/6

- Similarly for all other Hindi words: बैठती, है, कुत्ता, दौड़ता, तेज, दौड़ती

- Total vocabulary: English (6 words), Hindi (7 words)

Tasks:

**a)** Calculate the alignment probabilities for sentence 1 after the first E-step of EM algorithm and compute the updated translation probabilities P(बिल्ली|cat) and P(बिल्ली|the) after the first M-step. **[3 marks]**

**b)** A competing system uses HMM (Hidden Markov Model)-based alignment. Explain how the translation probabilities would differ from IBM Model 1 and calculate the locality penalty for jumping from position 1 to position 3 in a 4-word sentence, given that adjacent transitions have probability 0.6 and the jump distance penalty follows $s(d)=0.6^{|d|}$ where $d$ is the distance. Compare which model would be better for English-Hindi translation considering the SOV (Subject-Object-Verb) structure of Hindi. **[3 marks]**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Topics to Know

To answer this question, study the following:

- **IBM Model 1 + EM updates** — 📖 [IBM Model 1 and EM Training](../study/05-machine-translation.md#ibm-model-1-and-em-training) · [Alignment](../study/05-machine-translation.md#alignment)
  - E-step posterior and M-step re-estimation with expected counts

- **Uniform first-step behavior** — [Uniform Initialization Effect](../study/05-machine-translation.md#uniform-initialization-effect)
  - Why first E-step can produce equal alignment probabilities

- **Position-aware alignment** — [HMM Alignment and Locality Penalty](../study/05-machine-translation.md#hmm-alignment-and-locality-penalty)
  - Transition modeling and distance penalty $s(d)=\alpha^{|d|}$

- **Word-order impact (English-Hindi)** — [Indic Language Translation](../study/05-machine-translation.md#indic-language-translation) · [Challenges](../study/05-machine-translation.md#challenges)
  - SVO to SOV reordering implications for alignment

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Solution

### Part (a): IBM Model 1 E-step and first M-step

Sentence 1:
- English words: [the, cat, sits]
- Hindi words: [बिल्ली, बैठती, है]

IBM Model 1 posterior for each Hindi word $f_j$ aligning to English position $i$:

$$
P(a_j=i\mid f_j,\mathbf{e})
=\frac{t(f_j|e_i)}{\sum_{i'=1}^{3}t(f_j|e_{i'})}
$$

With uniform initialization, each denominator has three equal terms, so:

$$
P(a_j=i\mid f_j,\mathbf{e})=\frac{1}{3}
$$

So in sentence 1, each Hindi word aligns to each English word with probability $1/3$.

Expected counts needed for first M-step (over all 3 sentence pairs):

- c(बिल्ली,cat) = 1/3 + 1/3 = 2/3 (from sentence 1 and sentence 3)
- c(बिल्ली,the) = 1/3 (from sentence 1 only)

Denominators:

- $c(cat)=1+\frac{4}{3}=\frac{7}{3}$
  - sentence 1 contributes $3\times\frac{1}{3}=1$
  - sentence 3 contributes $4\times\frac{1}{3}=\frac{4}{3}$

- $c(the)=1+1=2$
  - sentence 1 contributes $3\times\frac{1}{3}=1$
  - sentence 2 contributes $3\times\frac{1}{3}=1$

Updated probabilities:

`t(बिल्ली|cat) = c(बिल्ली,cat)/c(cat) = (2/3)/(7/3) = 2/7 ≈ 0.2857`

`t(बिल्ली|the) = c(बिल्ली,the)/c(the) = (1/3)/2 = 1/6 ≈ 0.1667`

### Part (b): HMM comparison and locality penalty

Given $s(d)=0.6^{|d|}$ and jump from position 1 to 3:

$$
|d|=|3-1|=2,\quad s(2)=0.6^2=0.36
$$

So the locality weight/penalty is **0.36**.

Comparison:
- IBM Model 1 does not model position transitions; it relies only on lexical translation probabilities.
- HMM alignment includes transition/locality behavior, so it captures positional continuity.
- For English-Hindi (SVO to SOV), reordering is important; HMM is generally better than IBM Model 1, though strong locality bias can penalize long jumps.

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 2 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q2. Knowledge Graph Linking and IndicTrans2 Performance

**Marks:** [2+3+3=8] | **Source:** End-Sem EC-2 Regular 2024-25

**(a)** A tourism Knowledge Graph contains 15,000 entities with 78% entity linking accuracy. If a query identifies 5 entities, calculate how many will be correctly linked. Name one application of Knowledge Graphs in chatbots. **[2 marks]**

**(b)** You are analyzing an NMT system for English-Hindi translation using IndicTrans2.

Given:
- Hindi sentence with code-mixing: "मैं school जाता हूँ" (I go to school)
- English translation: "I go to school"
- Normal Hindi OOV rate: 5-8% (use 6%)
- Code-mixed OOV increase: 40-60% (use 50%)
- IndicTrans2 uses SentencePiece tokenization
- BLEU scores:
  - News: 25.0
  - Social media (code-mixed): 15.0
  - Technical documentation: 10.0
- Attention weights for Hindi word "school":
  - attends to "school": 0.80
  - attends to "go": 0.15
  - attends to "I": 0.05

Tasks:

1. Calculate expected OOV rate for the code-mixed sentence. If normal OOV is 6%, what is the new OOV rate with 50% increase due to code-mixing?
2. Calculate performance degradation percentage for social media and technical domains compared to news.
3. Compute attention entropy for "school" using:

$$
H=-\sum_i p_i\log_2(p_i)
$$

Given that good attention should have entropy $<2.0$ bits, determine whether this indicates good alignment quality.
4. Explain one reason why technical documentation has the lowest BLEU score.

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 2 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q2. Topics to Know

To answer this question, study the following:

- **Entity linking expectation** — 📖 [Entity Linking Accuracy](../study/03-knowledge-graph-applications.md#entity-linking-accuracy)
  - Expected correct links from query size and linking accuracy

- **KG use in assistants** — [Knowledge Graphs in Chatbots](../study/03-knowledge-graph-applications.md#knowledge-graphs-in-chatbots)
  - Knowledge-grounded response generation and disambiguation

- **Code-mixing and OOV in Indic MT** — [OOV and Code-Mixing in Indic MT](../study/05-machine-translation.md#oov-and-code-mixing-in-indic-mt)
  - Relative OOV increase under mixed-language text

- **BLEU and domain degradation** — [Evaluation of MT Quality BLEU](../study/05-machine-translation.md#evaluation-of-mt-quality-bleu) · [Domain Shift in MT](../study/05-machine-translation.md#domain-shift-in-mt)
  - Baseline-vs-domain quality comparison

- **Attention concentration quality** — [Attention Entropy and Alignment Confidence](../study/05-machine-translation.md#attention-entropy-and-alignment-confidence)
  - Entropy $H=-\sum_i p_i\log_2 p_i$ as alignment confidence proxy

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 2 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q2. Solution

### Part (a): Knowledge Graph linking

Expected correctly linked entities:

$$
5\times0.78=3.9
$$

So expected value is **3.9**, i.e., approximately **4 entities**.

One chatbot application:
- Knowledge-grounded answers (e.g., tourist place facts, opening hours, nearby attractions).

### Part (b): OOV, BLEU degradation, and attention entropy

Given normal OOV = 6% and code-mixed increase = 50%:

$$
\text{new OOV}=6\%\times(1+0.50)=9\%
$$

So expected code-mixed OOV is **9%**.

Degradation relative to news BLEU (25.0):

$$
\text{degradation(\%)}=\frac{25-\text{domain BLEU}}{25}\times100
$$

Social media:

$$
\frac{25-15}{25}\times100=40\%
$$

Technical documentation:

$$
\frac{25-10}{25}\times100=60\%
$$

Attention entropy for weights [0.80, 0.15, 0.05]:

$$
H=-(0.80\log_2 0.80+0.15\log_2 0.15+0.05\log_2 0.05)
\approx 0.884\ \text{bits}
$$

Since $0.884 < 2.0$, attention is concentrated, which indicates **good alignment quality** for this token.

Reason technical BLEU is lowest (one valid reason):
- Technical text has more domain-specific terminology and rare words, causing greater mismatch/OOV and lower translation quality.

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 3 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q3. NER and EE/RE Evaluation Metrics

**Marks:** [4+4=8] | **Source:** End-Sem EC-2 Regular 2024-25

Quantitatively evaluate an NLP system designed to extract business insights from a tech-news excerpt.

News excerpt:
"San Francisco, CA - On May 15, 2024, Innovate Inc. officially launched its new AI platform, 'CogniSphere'. CEO Jane Doe stated the launch coincided with a new strategic partnership with Global Tech Solutions, a deal finalized last month. CogniSphere is expected to generate over $50 million in revenue by Q4."

Data:

1. Prototype System Output

- NER:
  - May 15, 2024 (DATE)
  - Innovate Inc. (ORGANIZATION)
  - CogniSphere (PRODUCT)
  - Jane Doe (ORGANIZATION)
  - Global Tech Solutions (ORGANIZATION)
  - $50 million (MONEY)

- EE/RE:
  - (Product_Launch, Innovate Inc., CogniSphere, May 15, 2024)
  - (Partnership, Innovate Inc., Global Tech Solutions, May 16, 2024)

2. Gold Standard

- NER:
  - San Francisco, CA (LOCATION)
  - May 15, 2024 (DATE)
  - Innovate Inc. (ORGANIZATION)
  - CogniSphere (PRODUCT)
  - Jane Doe (PERSON)
  - Global Tech Solutions (ORGANIZATION)
  - last month (DATE)
  - $50 million (MONEY)
  - Q4 (FISCAL_PERIOD)

- EE/RE:
  - (Product_Launch, Innovate Inc., CogniSphere, May 15, 2024)
  - (Partnership, Innovate Inc., Global Tech Solutions, last month)
  - (Revenue_Projection, CogniSphere, $50 million, by Q4)

Tasks:

**Part A:** Calculate Precision, Recall, and F1-score for NER. Entity is correct only if span and type exactly match. **[4 marks]**

**Part B:** Calculate Precision, Recall, and F1-score for EE/RE. Event is correct only if event type and all primary arguments exactly match. **[4 marks]**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 3 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q3. Topics to Know

To answer this question, study the following:

- **Strict IE correctness criteria** — 📖 [Evaluation Metrics for NER and Event Extraction](../study/06-information-extraction.md#evaluation-metrics-for-ner-and-event-extraction) · [Strict Exact Match](../study/06-information-extraction.md#strict-exact-match)
  - Exact span/type for NER and exact event+arguments for EE/RE

- **Precision, recall, F1 computation** — [Precision, Recall, and F1 for IE](../study/06-information-extraction.md#precision-recall-and-f1-for-ie)
  - Metric computation from TP, FP, FN

- **Event and temporal argument extraction context** — [Extracting Events and Time](../study/06-information-extraction.md#extracting-events-and-time)
  - Why date/argument mismatch invalidates strict event match

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 3 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q3. Solution

### Part A: NER metrics (strict exact match)

Prototype NER count = 6  
Gold NER count = 9

Exact matches:
- May 15, 2024 (DATE)
- Innovate Inc. (ORGANIZATION)
- CogniSphere (PRODUCT)
- Global Tech Solutions (ORGANIZATION)
- $50 million (MONEY)

So:
- True Positives (TP) = 5
- False Positives (FP) = 6 - 5 = 1
- False Negatives (FN) = 9 - 5 = 4

Precision:

$$
P=\frac{TP}{TP+FP}=\frac{5}{6}=0.8333
$$

Recall:

$$
R=\frac{TP}{TP+FN}=\frac{5}{9}=0.5556
$$

F1-score:

$$
F1=\frac{2PR}{P+R}
=\frac{2\times 0.8333\times 0.5556}{0.8333+0.5556}
=0.6667
$$

NER results:
- **Precision = 83.33%**
- **Recall = 55.56%**
- **F1 = 66.67%**

### Part B: EE/RE metrics (strict exact event + arguments)

Prototype EE/RE count = 2  
Gold EE/RE count = 3

Exact event matches:
- (Product_Launch, Innovate Inc., CogniSphere, May 15, 2024) -> correct
- Partnership event is incorrect (prototype has May 16, 2024; gold has last month)
- Revenue_Projection event is missing in prototype

So:
- TP = 1
- FP = 2 - 1 = 1
- FN = 3 - 1 = 2

Precision:

$$
P=\frac{1}{2}=0.5
$$

Recall:

$$
R=\frac{1}{3}=0.3333
$$

F1-score:

$$
F1=\frac{2PR}{P+R}
=\frac{2\times 0.5\times 0.3333}{0.5+0.3333}
=0.4
$$

EE/RE results:
- **Precision = 50.00%**
- **Recall = 33.33%**
- **F1 = 40.00%**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 4 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q4. Naive Bayes Sentiment with Negation Handling

**Marks:** [2+4+2=8] | **Source:** End-Sem EC-2 Regular 2024-25

Scenario: You are building a simplified Naive Bayes-like sentiment classifier for movie reviews.

Review text ($D$): "not good movie"

Sentiment classes: Positive (+), Negative (-)

Given:
- Priors:
  - P(+) = 0.4
  - P(-) = 0.6
- Likelihoods:
  - P(good|+) = 0.08
  - P(good|-) = 0.01
  - P(movie|+) = 0.05
  - P(movie|-) = 0.04
  - P(NOT_good|+) = 0.01
  - P(NOT_good|-) = 0.07

Negation rule:
- Prefix "NOT_" to words after a negation token until punctuation.
- For this question, "not good" is treated as one effective token: "NOT_good".

Tasks:

1. Identify effective tokens after applying negation. **[2 marks]**
2. Calculate unnormalized posterior score for each class:

`Score(Class|D) = P(Class) * P(token1|Class) * P(token2|Class)`

**[4 marks]**
3. Determine final sentiment classification from scores. **[2 marks]**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 4 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q4. Topics to Know

To answer this question, study the following:

- **Naive Bayes sentiment scoring** — 📖 [Naive Bayes for Sentiment Classification](../study/07-sentiment-analysis.md#naive-bayes-for-sentiment-classification)
  - Posterior score from class priors and token likelihoods

- **Negation feature transformation** — [Negation Handling with NOT Tokens](../study/07-sentiment-analysis.md#negation-handling-with-not-tokens)
  - Converting `not good` to `NOT_good`

- **Final class decision rule** — [Decision Rule Using Posterior Scores](../study/07-sentiment-analysis.md#decision-rule-using-posterior-scores)
  - Predict class with larger score

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 4 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q4. Solution

### Part 1: Effective tokens (2 marks)

Given review: "not good movie"

Using the rule, "not good" becomes `NOT_good`.

Effective tokens are:
- `NOT_good`
- `movie`

### Part 2: Unnormalized posterior scores (4 marks)

Positive class:

`Score(+|D) = P(+) * P(NOT_good|+) * P(movie|+)`

$$
=0.4\times 0.01\times 0.05
=0.0002
$$

Negative class:

`Score(-|D) = P(-) * P(NOT_good|-) * P(movie|-)`

$$
=0.6\times 0.07\times 0.04
=0.00168
$$

### Part 3: Final sentiment (2 marks)

Compare scores:
- $\text{Score}(+|D)=0.0002$
- $\text{Score}(-|D)=0.00168$

Since $0.00168 > 0.0002$, predicted class is **Negative**.

<div style="page-break-after: always;"></div>

---

## Navigation

- [Questions Index](./)
- [Study](../study/)
- [Course Home](../)
- [Back to Homepage](../../)
