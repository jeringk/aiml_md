## 14.1 SNA for Social Media Marketing Strategies

- **Social Network Analysis (SNA)** applied to marketing — use network structure to optimize campaigns

### Influencer Marketing

- Identify influencers using network centrality measures:
  - **Degree centrality**: large audience
  - **Betweenness centrality**: bridge between communities
  - **PageRank**: endorsed by other influential people
- Types of influencers:

| Type | Followers | Characteristics |
|------|-----------|-----------------|
| **Mega-influencers** | 1M+ | Celebrities, broad reach, lower engagement rate |
| **Macro-influencers** | 100K–1M | Industry experts, good reach |
| **Micro-influencers** | 10K–100K | Niche audiences, high engagement |
| **Nano-influencers** | 1K–10K | Hyper-local, highest trust and engagement |

### Viral Marketing

- Leverage network structure to maximize information spread
- **Seed selection**: use influence maximization algorithms (greedy, CELF)
- **Network effects**: each new user/customer increases value for existing users

### Community-Based Marketing

- Detect communities → tailor messaging for each community's interests
- **Cross-community bridges**: target users who span multiple communities for broader reach
- **Community sentiment**: track per-community brand perception

### Word-of-Mouth (WoM) Analysis

- Map referral networks — who recommends to whom
- Identify super-spreaders and loyal advocates
- Measure WoM reach and conversion rates

---

## 14.2 SNA for Non-Marketing Use Cases

### Public Health

- **Disease surveillance**: track symptoms/outbreaks on social media
- **Contact tracing**: model disease spread through social networks
- **Mental health**: detect depression, anxiety from language patterns and network changes
- **Vaccination campaigns**: identify resistant communities, target outreach

### Law Enforcement & Security

- **Criminal network analysis**: detect organized groups, communication patterns
- **Terrorism/extremism**: identify radicalization pathways in networks
- **Fraud detection**: analyze networks of suspicious transactions/accounts
- **Cyberbullying**: detect harassment networks and victims

### Political Science

- **Political polarization**: measure ideological clustering in networks
- **Information operations**: detect foreign influence campaigns
- **Policy feedback**: understand public opinion on policies through network discussions
- **Political mobilization**: track activist networks and movement formation

### Disaster Management

- **Situational awareness**: real-time monitoring during disaster events
- **Resource allocation**: identify affected communities and communication hubs
- **Rumor control**: detect and counter misinformation during crises
- **Volunteer coordination**: network-based coordination of relief efforts

---

## 14.3 Multi-modal Analytics Leveraging GenAI

- **Multi-modal analytics**: analyzing data from multiple modalities — text, images, audio, video — together
- Social media is inherently multi-modal

### Modalities in Social Media

| Modality | Examples | Analysis Tasks |
|----------|----------|----------------|
| **Text** | Posts, comments, captions | Sentiment, topic, NER, summarization |
| **Image** | Photos, infographics, memes | Object detection, OCR, meme analysis |
| **Audio** | Podcasts, voice messages, live audio | Speech-to-text, emotion detection |
| **Video** | Reels, stories, live streams | Activity recognition, content moderation |
| **Network** | Follower graphs, interaction networks | Community detection, influence |

### Multi-modal Fusion Strategies

| Strategy | Description |
|----------|-------------|
| **Early fusion** | Combine raw features from all modalities before analysis |
| **Late fusion** | Analyze each modality separately, combine predictions |
| **Hybrid fusion** | Combine at intermediate representation levels |
| **Cross-modal attention** | Use attention mechanisms to relate modalities |

### GenAI for Multi-modal Analysis

- **Vision-Language Models** (GPT-4V, Gemini, Claude): understand images + text together
  - Meme understanding and classification
  - Image-text consistency checking (fake news detection)
  - Visual sentiment analysis
- **Audio/Speech Models** (Whisper, Gemini): transcribe and analyze audio content
- **Video Models**: analyze video content frame-by-frame or holistically

### Applications

| Application | Multi-modal Approach |
|-------------|---------------------|
| **Fake news detection** | Cross-reference text claims with image/video evidence |
| **Meme analysis** | Understand humor/sarcasm from image-text combination |
| **Content moderation** | Detect harmful content across text, images, video |
| **Brand monitoring** | Track brand appearances in images, mentions in text |
| **Accessibility** | Generate alt-text, captions, audio descriptions |

### Challenges

- Modality alignment: synchronizing information across modalities
- Missing modalities: not all posts have all modalities
- Computational cost: multi-modal models are resource-intensive
- Cultural context: memes and visual content are culturally specific

---

## Key Takeaways

- SNA enables data-driven marketing strategies through influencer identification, viral seeding, and community targeting
- Non-marketing SNA applications span public health, security, politics, and disaster management
- Multi-modal analytics captures richer information than single-modality analysis
- GenAI (vision-language models) makes multi-modal analysis accessible and powerful
- Social media is inherently multi-modal — effective analytics must embrace all modalities

