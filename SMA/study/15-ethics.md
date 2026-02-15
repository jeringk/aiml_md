# Module 15 — Ethics in Social Media

## Topics

- [[#15.1 Privacy and Data Protection|Privacy and Data Protection]]
- [[#15.2 Bias and Fairness|Bias and Fairness]]
- [[#15.3 Misinformation and Manipulation|Misinformation and Manipulation]]
- [[#15.4 Platform Responsibility and Regulation|Platform Responsibility and Regulation]]
- [[#15.5 Ethical Frameworks for Social Media Analytics|Ethical Frameworks for Social Media Analytics]]

---

## 15.1 Privacy and Data Protection

- Social media analytics inherently involves personal data — raising significant privacy concerns

### Key Privacy Issues

| Issue | Description |
|-------|-------------|
| **Informed consent** | Do users know their data is being analyzed? |
| **Data minimization** | Collecting only what's necessary |
| **Re-identification** | Anonymized data can often be de-anonymized |
| **Scope creep** | Data collected for one purpose used for another |
| **Third-party sharing** | Data shared with advertisers, researchers, governments |

### Regulatory Frameworks

| Regulation | Jurisdiction | Key Requirements |
|------------|-------------|-----------------|
| **GDPR** | EU | Consent, right to be forgotten, data portability, DPO |
| **CCPA/CPRA** | California | Opt-out of data sales, right to know, right to delete |
| **DPDPA** | India | Consent, purpose limitation, data fiduciary obligations |

### Privacy-Preserving Analytics

- **Differential privacy**: add noise to query results to protect individual records
- **Federated learning**: train models without centralizing data
- **Anonymization**: remove PII, but beware of re-identification
- **Aggregation**: report aggregate statistics rather than individual-level data

### Cambridge Analytica Case

- Facebook user data harvested via a quiz app — shared with third parties without consent
- Used for political targeting in elections
- Led to increased privacy regulation and platform policy changes

---

## 15.2 Bias and Fairness

### Sources of Bias in Social Media Analytics

| Bias Type | Description | Example |
|-----------|-------------|---------|
| **Sampling bias** | Social media users ≠ general population | Twitter users skew younger, urban |
| **Algorithmic bias** | ML models perpetuate/amplify biases in training data | Sentiment models biased against AAVE |
| **Selection bias** | Which data gets collected/analyzed | API access limitations |
| **Confirmation bias** | Seeking data that confirms hypotheses | Cherry-picking supporting evidence |
| **Presentation bias** | How results are displayed influences interpretation | Misleading visualizations |

### Fairness in Algorithms

- **Group fairness**: equal outcomes across demographic groups
- **Individual fairness**: similar individuals receive similar treatment
- **Counterfactual fairness**: outcome would be the same in a counterfactual world

### Addressing Bias

- Audit training data for representativeness
- Measure and report disparate impact across groups
- Use fairness-aware ML techniques
- Diverse teams building and evaluating systems
- Transparency in methodology and limitations

---

## 15.3 Misinformation and Manipulation

### Types

| Type | Description |
|------|-------------|
| **Misinformation** | False information shared without intent to deceive |
| **Disinformation** | Deliberately false information created to deceive |
| **Malinformation** | True information shared with intent to harm (doxxing, leaked private info) |

### Detection Methods

- **Content analysis**: NLP to detect false claims, fact-checking
- **Network analysis**: identify bot networks, coordinated inauthentic behavior
- **Propagation patterns**: misinformation spreads differently from truth (broader, faster, more novel)
- **Source credibility**: assess trustworthiness of information sources

### Platform Interventions

- Content labeling and warnings
- Reduced algorithmic amplification
- Fact-checking partnerships
- Account suspension and content removal
- Transparency reports

### Deepfakes and Synthetic Media

- AI-generated images, videos, audio that look/sound real
- Detection methods: visual artifacts, inconsistencies, forensic analysis
- Ethical implications: erosion of trust in media

---

## 15.4 Platform Responsibility and Regulation

### Key Debates

| Issue | Arguments |
|-------|-----------|
| **Content moderation** | Free speech vs. safety; who decides what's harmful? |
| **Algorithmic transparency** | Should platforms reveal how algorithms work? |
| **Section 230 (US)** | Platforms are not liable for user content — should this change? |
| **Data monetization** | Users generate value — should they be compensated? |
| **Child safety** | COPPA, age verification, protecting minors |

### Emerging Regulation

- **EU Digital Services Act (DSA)**: transparency, content moderation rules for large platforms
- **EU AI Act**: risk-based regulation of AI systems including social media algorithms
- **India IT Rules**: intermediary guidelines, content takedown obligations

---

## 15.5 Ethical Frameworks for Social Media Analytics

### Research Ethics

- **IRB approval**: research involving social media data may require ethics board review
- **Belmont Report principles**:
  - **Respect for persons**: informed consent, protecting vulnerable populations
  - **Beneficence**: maximize benefits, minimize harms
  - **Justice**: fair distribution of research benefits and burdens

### Practical Guidelines

1. **Transparency**: disclose data sources, methods, limitations
2. **Consent**: respect users' expectations of privacy
3. **Proportionality**: data collection proportional to research benefit
4. **Accountability**: take responsibility for impacts of analysis
5. **Reproducibility**: enable others to verify results

### Ethical Dilemmas in Practice

- Analyzing public posts: consent is implicit but users may not expect analysis
- Studying vulnerable communities: potential for harm even with public data
- Predictive policing from social media: risk of discrimination
- Mental health detection: beneficence vs. surveillance

---

## Key Takeaways

- Privacy, bias, and misinformation are the three pillars of social media ethics
- GDPR, CCPA, and other regulations impose legal requirements on data handling
- Algorithmic bias can perpetuate and amplify societal inequalities
- Researchers and practitioners must follow ethical frameworks and consider societal impact
- Social media ethics is an evolving field — laws and norms continue to develop

---

## References

- CS224W of Stanford — Ethics in Network Analysis
- GDPR, CCPA, DPDPA regulatory documents
