# Module 10 — Applications of Social Media Analytics: Influence & Homophily

## Topics

- [[#10.1 Measuring Assortativity|Measuring Assortativity]]
- [[#10.2 Influence|Influence]]
- [[#10.3 Homophily|Homophily]]
- [[#10.4 Distinguishing Influence from Homophily|Distinguishing Influence from Homophily]]

---

## 10.1 Measuring Assortativity

- **Assortativity**: tendency of nodes to connect with similar nodes
- **Assortative mixing**: similar nodes connect (e.g., high-degree to high-degree)
- **Disassortative mixing**: dissimilar nodes connect (e.g., high-degree to low-degree)

### Degree Assortativity

- **Assortativity coefficient** $r$: Pearson correlation of degrees at either end of an edge

$$r = \frac{\sum_{ij} (A_{ij} - k_i k_j / 2m) k_i k_j}{\sum_{ij} (k_i \delta_{ij} - k_i k_j / 2m) k_i k_j}$$

- $r > 0$: **assortative** (hubs connect to hubs) — common in social networks
- $r < 0$: **disassortative** (hubs connect to low-degree) — common in biological, technological networks
- $r = 0$: no correlation

### Attribute Assortativity

- Can measure assortativity for any attribute (age, gender, location, interests)
- **Modularity** can be interpreted as assortativity of community labels

---

## 10.2 Influence

- **Influence**: the ability of a node to affect the behavior, opinion, or actions of other nodes
- Key question: do people adopt behaviors **because** their friends have?

### Types of Influence

| Type | Description | Example |
|------|-------------|---------|
| **Informational** | Learning from others' actions | "My friend recommended this product" |
| **Normative** | Conforming to social norms | "Everyone is talking about this movie" |
| **Identification** | Adopting behavior of admired others | Following an influencer's choices |
| **Peer pressure** | Direct social pressure | Friends encouraging specific behavior |

### Measuring Influence

- **Correlation ≠ causation** — observing that friends share behaviors doesn't prove influence
- Influence measurement approaches:
  - **Randomized experiments**: A/B testing on social platforms
  - **Natural experiments**: exploiting platform changes or random events
  - **Instrumental variables**: find variables that affect network exposure but not outcome directly
  - **Temporal analysis**: does a friend's adoption **precede** yours? (Granger causality)

### Influence in Social Media

- **Retweet cascades**: who influences whom through retweets
- **Adoption networks**: who adopts products/features after their friends
- **Opinion dynamics**: how opinions change through social exposure

---

## 10.3 Homophily

- **Homophily** ("love of the same"): the tendency for similar individuals to form connections
- _"Birds of a feather flock together"_

### Types of Homophily

| Type | Description | Example |
|------|-------------|---------|
| **Status homophily** | Similar socioeconomic status, demographics | Same age, education, income |
| **Value homophily** | Similar beliefs, values, attitudes | Political ideology, religion |
| **Behavioral homophily** | Similar behaviors | Same hobbies, consumption patterns |

### Choice vs. Induced Homophily

- **Choice homophily**: individuals deliberately choose similar friends
- **Induced homophily**: structural factors force similar people together (same school, workplace, neighborhood)

### Measuring Homophily

- **Homophily ratio**: fraction of edges connecting nodes with the same attribute vs. expected by chance

$$H = \frac{\text{observed same-attribute edges}}{\text{expected same-attribute edges under random mixing}}$$

- $H > 1$: homophily (more same-attribute connections than expected)
- $H < 1$: heterophily (fewer same-attribute connections)
- $H = 1$: random mixing

---

## 10.4 Distinguishing Influence from Homophily

- The **identification problem**: when friends share a behavior, is it due to:
  1. **Influence**: A adopted the behavior and then B adopted it because of A
  2. **Homophily**: A and B both like similar things independently, and that's why they're friends
  3. **Confounding**: both were exposed to the same external factor

### The Challenge

- Observational data alone cannot distinguish these mechanisms
- Example: if two friends both buy the same phone:
  - Was it influence (one recommended it)?
  - Homophily (they share similar tastes)?
  - Confounding (both saw the same ad)?

### Approaches to Disentangle

| Method | Description |
|--------|-------------|
| **Shuffle test** | Randomly permute node attributes; compare correlation in real vs. shuffled |
| **Temporal ordering** | If A adopts before B (and B is connected to A), suggestive of influence |
| **Matched sampling** | Compare adopters' friends to non-adopters' friends (Anagnostopoulos et al.) |
| **Causal inference** | Propensity score matching, instrumental variables |
| **Randomized experiments** | Gold standard: randomly vary exposure to friends' behaviors |

### Key Insight

- In most real-world settings, **both** influence and homophily operate simultaneously
- Disentangling them is one of the hardest problems in computational social science

---

## Key Takeaways

- Assortativity measures the tendency of similar nodes to connect — positive in social networks
- Influence is the process by which network neighbors affect each other's behavior
- Homophily is the tendency for similar people to form ties
- Distinguishing influence from homophily is a fundamental challenge — requires careful causal reasoning
- Both mechanisms are important for understanding social media dynamics

---

## References

- T1: Zafarani et al., Ch. 8 — Influence and Homophily
