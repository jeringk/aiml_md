## 3.1 Information Extraction

- **Goal**: automatically extract **structured information** from unstructured text
- Critical for social media — converting noisy text into structured, actionable data

### Key IE Tasks

#### Named Entity Recognition (NER)

- Identify and classify mentions of named entities: **Person, Organization, Location, Date, Product**, etc.
- Social media challenges:
  - Informal text, abbreviations, lack of capitalization
  - Novel entities (memes, emerging brands, hashtags)
  - Short, contextless text
- Approaches:
  - Rule-based (gazetteers, regex)
  - CRF / BiLSTM-CRF (sequence labeling)
  - Transformer-based: BERT, RoBERTa fine-tuned for NER
  - Few-shot NER with LLMs

#### Relation Extraction

- Identify **relationships** between entities
- Example: "Elon Musk is the CEO of Tesla" → (Elon Musk, CEO_of, Tesla)
- Methods: dependency parsing, distant supervision, neural RE models

#### Event Extraction

- Detect **events** and their attributes (who, what, when, where)
- Social media applications: breaking news detection, disaster tracking
- Example: "Earthquake of magnitude 6.5 hit Turkey today" → Event(type=earthquake, magnitude=6.5, location=Turkey, date=today)

#### Hashtag and Keyword Extraction

- Identify trending topics, relevant hashtags
- TF-IDF, TextRank, YAKE for keyword extraction

---

## 3.2 Text Summarization

- **Goal**: produce concise summaries of longer text or collections of texts
- In social media: summarize trending discussions, event-related tweets, product reviews

### Types

| Type | Description | Example Methods |
|------|-------------|-----------------|
| **Extractive** | Select key sentences from source text | TextRank, LexRank, LSA |
| **Abstractive** | Generate new text that paraphrases the source | Seq2Seq, T5, BART, GPT |

### Extractive Summarization

- **TextRank**: graph-based method
  - Build graph of sentences (nodes), edges weighted by similarity
  - Apply PageRank to rank sentences
  - Select top-ranked sentences
- **LexRank**: similar to TextRank but uses cosine similarity on TF-IDF vectors
- **Centroid-based**: select sentences closest to the centroid of the document

### Abstractive Summarization

- **Encoder-decoder models**: encode source, decode summary
- **Attention mechanism**: focus on relevant parts of source
- **Transformer models**: BART, T5, Pegasus — state-of-the-art
- **Challenges**: hallucination (generating facts not in the source), maintaining factual consistency

### Multi-Document Summarization

- Summarize information from **multiple sources** (e.g., many tweets about the same event)
- Additional challenges: redundancy detection, information fusion, timeline ordering

---

## 3.3 Leveraging GenAI

- **Generative AI** (LLMs like GPT, Gemini, Claude) transforms social media analytics

### Applications in Social Media NLP

| Task | GenAI Application |
|------|-------------------|
| **Sentiment analysis** | Zero-shot/few-shot classification with prompting |
| **Summarization** | Abstractive summaries with LLMs |
| **Information extraction** | Prompt-based NER, relation extraction |
| **Content generation** | Automated social media posts, replies |
| **Translation** | Multilingual social media analysis |
| **Question answering** | Answering questions about social media trends |

### Prompt Engineering for Social Media

- **Zero-shot**: "Classify the sentiment of: '{tweet}'" → positive/negative/neutral
- **Few-shot**: provide examples before the query
- **Chain-of-thought**: reason step-by-step for complex analysis

### LLMs for Social Media Analysis

- **Advantages**:
  - No task-specific training data needed (zero-shot)
  - Handle noisy, informal text well
  - Multilingual capabilities
  - Can perform multiple tasks with one model
- **Limitations**:
  - Hallucination risk
  - Cost and latency at scale
  - Privacy concerns (sending data to external APIs)
  - May not capture domain-specific nuances without fine-tuning

### Retrieval-Augmented Generation (RAG)

- Combine LLMs with a retrieval system — retrieve relevant social media posts, then generate analysis
- Reduces hallucination, grounds outputs in actual data

---

## Key Takeaways

- Information extraction converts unstructured social media text into structured data (entities, relations, events)
- Text summarization (extractive and abstractive) condenses large volumes of social content
- GenAI/LLMs enable zero-shot and few-shot analysis, dramatically reducing the need for labeled data
- Prompt engineering is a key skill for leveraging LLMs in social media analytics

