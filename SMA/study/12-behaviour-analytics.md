# Module 12 — Applications of Social Media Analytics: Behaviour Analytics

## Topics

- [[#12.1 Individual Behaviour|Individual Behaviour]]
- [[#12.2 Collective Behaviour|Collective Behaviour]]

---

## 12.1 Individual Behaviour

- Analyzing how **individual users** behave on social media
- Behavior reveals preferences, personality, intentions, and anomalies

### User Profiling

| Attribute | Description | Methods |
|-----------|-------------|---------|
| **Demographics** | Age, gender, location | Classification from text, images, network |
| **Personality** | Big Five traits (OCEAN) | NLP on posts, psycholinguistic features (LIWC) |
| **Political leaning** | Political orientation | Follows, shares, hashtag usage |
| **Interests** | Topics of interest | Topic modeling, followed pages/accounts |
| **Credibility** | How credible is the user? | Account age, verification, consistency |

### Behavioral Patterns

- **Posting patterns**: frequency, timing (diurnal cycles), content type
- **Engagement patterns**: like/share/comment ratios, reciprocity
- **Mobility patterns**: check-ins, geotagged posts
- **Language use**: formality, emoticon usage, vocabulary richness

### Anomalous Individual Behavior

- **Bot detection**: automated accounts with unnatural patterns
  - Features: posting rate, response time, content similarity, network structure
  - Models: Random Forest, neural networks on behavioral features
- **Sockpuppets**: multiple accounts controlled by one person
- **Trolling**: deliberately disruptive or inflammatory behavior
- **Radicalization detection**: monitoring for extremist behavior patterns

### Prediction Tasks

- **Churn prediction**: will a user leave the platform?
- **Activity prediction**: when and how often will a user post?
- **Interest prediction**: what topics will a user engage with next?

---

## 12.2 Collective Behaviour

- Analyzing behaviors that emerge from **groups** of users interacting
- Patterns that cannot be understood by studying individuals alone

### Emergent Phenomena

| Phenomenon | Description | Example |
|------------|-------------|---------|
| **Herding** | Following the crowd without independent evaluation | Mass buying after a viral recommendation |
| **Information cascades** | Sequential decisions based on predecessors | Retweet chains, viral content |
| **Wisdom of crowds** | Aggregated group judgment outperforms individuals | Prediction markets, Wikipedia |
| **Collective intelligence** | Group solves problems better than individuals | Open-source projects, crowdsourcing |
| **Groupthink** | Desire for conformity leads to poor decisions | Echo chambers, filter bubbles |

### Trending and Viral Content

- **Trend detection**: identify topics/hashtags gaining unusual momentum
  - Statistical methods: burst detection, anomaly in time series
  - Comparison against baseline activity levels
- **Virality prediction**: predict which content will go viral
  - Features: content features, network position of poster, early engagement metrics
  - Early cascade structure is predictive

### Crowd Behavior Analysis

- **Flash mobs and coordinated action**: detecting organized group behavior
- **Protest detection**: identifying emerging social movements from social data
- **Panic and rumor spreading**: how misinformation propagates during crises

### Echo Chambers and Filter Bubbles

- **Echo chamber**: social environment where users encounter only beliefs similar to their own
- **Filter bubble**: algorithmic curation shows users only aligned content
- Measurement:
  - Political ideology partitioning in following/retweet networks
  - Content diversity in users' feeds
  - Cross-ideological exposure metrics

### Collective Sentiment

- Aggregate sentiment of populations over time
- **Applications**: predicting stock markets from Twitter sentiment, tracking public mood during events
- **Methods**: sentiment aggregation, time series analysis, causal modeling

---

## Key Takeaways

- Individual behavior analysis enables user profiling, bot detection, and churn prediction
- Collective behavior reveals emergent phenomena like cascades, herding, and echo chambers
- Bot detection combines behavioral, content, and network features
- Echo chambers and filter bubbles are critical concerns for platform health and democracy
- Both individual and collective analysis are essential for comprehensive social media analytics

---

## References

- T1: Zafarani et al., Ch. 10 — Behaviour Analytics
