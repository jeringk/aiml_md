# ADL — Makeup Examination (EC-3 Regular) 2024-2025

**Course No.:** AIMLCLZG513 | **Course Title:** Advanced Deep Learning
**Nature:** Open Book | **Weightage:** 40% | **Pages:** 2 | **Questions:** 4

---

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. RealNVP Flow & MLE for Bernoulli

**Marks:** [2+3+5=10] | **Source:** Makeup EC-3 Regular 2024-25

**A.** In a RealNVP flow model, given an input vector $\mathbf{x} = [1\ 1]^\top$,

**(i)** What will be the value of transformed output vector $\mathbf{y}$? **(2 marks)**

**(ii)** Calculate the log determinant of the Jacobian. **(3 marks)**

Given an input vector $\mathbf{x} = (x_1, x_2)$, the output vector $\mathbf{y} = (y_1, y_2)$ is computed by:

$$y_1 = x_1$$

$$y_2 = x_2 \cdot \exp(s(x_1)) + t(x_1)$$

Where $s(x_1)$ and $t(x_1)$ are scale and translation functions, typically parameterized by neural networks.

Suppose in a specific RealNVP layer:

- $s(x_1) = 0.5 x_1$
- $t(x_1) = 2x_1 + 1$

---

**B.** You are given a dataset of 4 small **binary images**, each of size **2×2 pixels**. Each pixel is either 0 (black) or 1 (white). Assume that each pixel in the images is an independent Bernoulli random variable with unknown success probability $p$, where:

$$P(\text{pixel} = 1) = p, \quad P(\text{pixel} = 0) = 1 - p$$

Estimate $p$ by maximizing the log likelihood principle. **(5 marks)**

The 4 images are:

| Image 1       | Image 2       | Image 3       | Image 4       |
|:-------------:|:-------------:|:-------------:|:-------------:|
| $\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$ | $\begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$ | $\begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$ | $\begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}$ |

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Topics to Know

To answer this question, study the following:

- **RealNVP (Real-valued Non-Volume Preserving) Flow** — 📖 [5.4.2 NICE / RealNVP](../study/05-normalizing-flow-models.md#542-nice--realnvp)
  - Affine coupling layer: $y_1 = x_1$, $y_2 = x_2 \cdot \exp(s(x_1)) + t(x_1)$
  - Jacobian is lower-triangular → determinant = product of diagonal entries
  - $\frac{\partial y_1}{\partial x_1} = 1$, $\frac{\partial y_2}{\partial x_2} = \exp(s(x_1))$
  - Log determinant: $\log|\det J| = s(x_1)$

- **Maximum Likelihood Estimation (MLE) for Bernoulli** — 📖 [9.5 Training and Sampling from EBMs](../study/09-energy-score-based-models.md#95-training-and-sampling-from-ebms)
  - Likelihood: $L(p) = p^k (1 - p)^{n - k}$ where $k$ = number of 1s, $n$ = total pixels
  - Log-likelihood: $\ell(p) = k \log p + (n-k) \log(1-p)$
  - MLE: set $\frac{d\ell}{dp} = 0$ → $\hat{p} = \frac{k}{n}$

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Solution

### Part A(i): Transformed Output Vector $\mathbf{y}$ (2 marks)

Given $\mathbf{x} = [1, 1]^\top$, $s(x_1) = 0.5x_1$, $t(x_1) = 2x_1 + 1$:

**Compute $s(x_1)$ and $t(x_1)$:**

$$s(1) = 0.5 \times 1 = 0.5$$

$$t(1) = 2 \times 1 + 1 = 3$$

**Compute $\mathbf{y}$:**

$$y_1 = x_1 = 1$$

$$y_2 = x_2 \cdot \exp(s(x_1)) + t(x_1) = 1 \cdot \exp(0.5) + 3 = 1.6487 + 3 = 4.6487$$

$$\boxed{\mathbf{y} = [1,\ 4.6487]^\top}$$

---

### Part A(ii): Log Determinant of the Jacobian (3 marks)

The Jacobian of the RealNVP coupling layer:

$$J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\ \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ * & \exp(s(x_1)) \end{bmatrix}$$

The matrix is **lower-triangular**, so the determinant is the product of diagonal entries:

$$\det(J) = 1 \times \exp(s(x_1)) = \exp(0.5)$$

**Log determinant:**

$$\boxed{\log|\det(J)| = s(x_1) = 0.5}$$

---

### Part B: MLE for Bernoulli Parameter $p$ (5 marks)

**Step 1:** Count pixels across all 4 images:

| Image | Pixels         | 1s | 0s |
|-------|----------------|:--:|:--:|
| 1     | 1, 1, 1, 1     | 4  | 0  |
| 2     | 0, 1, 0, 0     | 1  | 3  |
| 3     | 1, 0, 1, 1     | 3  | 1  |
| 4     | 0, 0, 1, 0     | 1  | 3  |
| **Total** |            | **9** | **7** |

Total pixels: $n = 4 \times 4 = 16$, Total 1s: $k = 9$

**Step 2:** Write the log-likelihood:

$$\ell(p) = \sum_{i=1}^{n} \left[ x_i \log p + (1 - x_i) \log(1 - p) \right] = k \log p + (n - k) \log(1 - p)$$

$$\ell(p) = 9 \log p + 7 \log(1 - p)$$

**Step 3:** Differentiate and set to zero:

$$\frac{d\ell}{dp} = \frac{9}{p} - \frac{7}{1 - p} = 0$$

$$\frac{9}{p} = \frac{7}{1 - p}$$

$$9(1 - p) = 7p$$

$$9 - 9p = 7p$$

$$9 = 16p$$

$$\boxed{\hat{p} = \frac{9}{16} = 0.5625}$$

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 2 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q2. Word2Vec, GloVe & CoVe

**Marks:** [2+2+2+2+1+1=10] | **Source:** Makeup EC-3 Regular 2024-25

**A.** You are given a sentence of length 5. All 5 words $w_i$, $i = 1, \ldots, 5$ are unique. If window size of 3 (one word to the right and one to the left of the central word) is used and embedding size of 4 for obtaining word2vec:

**(i)** What will be the number of trainable parameters in CBOW based model? Show all steps. **(2 marks)**

**(ii)** What will be the number of training parameters in skip-gram based model? Show all steps. **(2 marks)**

**(iii)** What will be the size of training set for CBOW model? Show the training set (in terms of $w_i$). **(2 marks)**

**(iv)** What will be the size of training set for skip-gram model? Show the training set. **(2 marks)**

**C.** What is the benefit offered by GloVe over word2vec? **(1 mark)**

**D.** What is the benefit offered by CoVe over GloVe or word2vec? **(1 mark)**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 2 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q2. Topics to Know

To answer this question, study the following:

- **Word2Vec — CBOW (Continuous Bag of Words)** — 📖 [10.3 word2Vec: CBOW](../study/10-language-modeling.md#cbow-continuous-bag-of-words) · [Architecture & Parameters](../study/10-language-modeling.md#architecture--parameters)
  - Input: context words → Output: center word
  - Parameters: input embedding matrix $W_{in} \in \mathbb{R}^{V \times d}$ + output matrix $W_{out} \in \mathbb{R}^{d \times V}$
  - Total params = $2 \times V \times d$

- **Word2Vec — Skip-gram** — 📖 [10.3 word2Vec: Skip-gram](../study/10-language-modeling.md#skip-gram) · [Architecture & Parameters](../study/10-language-modeling.md#architecture--parameters)
  - Input: center word → Output: context words
  - Same parameter matrices as CBOW: $W_{in} \in \mathbb{R}^{V \times d}$ + $W_{out} \in \mathbb{R}^{d \times V}$
  - Total params = $2 \times V \times d$

- **Training Set Construction** — 📖 [10.3 Training Set Construction](../study/10-language-modeling.md#training-set-construction)
  - Window size $k$: consider $\lfloor k/2 \rfloor$ words on each side
  - CBOW: (context words) → center word (one sample per center word)
  - Skip-gram: center word → each context word (one sample per context-center pair)
  - Boundary effects: edge words have fewer context neighbors

- **GloVe vs Word2Vec** — 📖 [10.3.1 GloVe](../study/10-language-modeling.md#1031-glove)
  - GloVe uses global co-occurrence statistics; word2vec uses local context windows
  - GloVe captures both local and global patterns

- **CoVe vs GloVe/Word2Vec** — 📖 [10.3.2 CoVe](../study/10-language-modeling.md#1032-cove-contextualized-word-vectors) · [CoVe vs Static Embeddings](../study/10-language-modeling.md#cove-vs-static-embeddings)
  - CoVe produces contextualized embeddings (word meaning varies by sentence)
  - GloVe/Word2Vec produce static embeddings (same vector regardless of context)

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 2 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q2. Solution

### Part A(i): CBOW Trainable Parameters (2 marks)

**Given:** Vocabulary size $V = 5$, Embedding dimension $d = 4$

CBOW has two weight matrices:
- **Input (embedding) matrix:** $W_{in} \in \mathbb{R}^{V \times d} = 5 \times 4 = 20$ parameters
- **Output (projection) matrix:** $W_{out} \in \mathbb{R}^{d \times V} = 4 \times 5 = 20$ parameters

$$\boxed{\text{Total CBOW parameters} = 20 + 20 = 40}$$

---

### Part A(ii): Skip-gram Trainable Parameters (2 marks)

Skip-gram has the same two weight matrices as CBOW:
- **Input (embedding) matrix:** $W_{in} \in \mathbb{R}^{V \times d} = 5 \times 4 = 20$ parameters
- **Output (projection) matrix:** $W_{out} \in \mathbb{R}^{d \times V} = 4 \times 5 = 20$ parameters

$$\boxed{\text{Total Skip-gram parameters} = 20 + 20 = 40}$$

---

### Part A(iii): CBOW Training Set (2 marks)

Window size = 3 → 1 word on each side. CBOW: context → center word.

| Center Word | Context Words      | Training Sample                    |
|-------------|--------------------|------------------------------------|
| $w_1$       | $w_2$              | $(\{w_2\}, w_1)$                   |
| $w_2$       | $w_1, w_3$         | $(\{w_1, w_3\}, w_2)$             |
| $w_3$       | $w_2, w_4$         | $(\{w_2, w_4\}, w_3)$             |
| $w_4$       | $w_3, w_5$         | $(\{w_3, w_5\}, w_4)$             |
| $w_5$       | $w_4$              | $(\{w_4\}, w_5)$                   |

$$\boxed{\text{CBOW training set size} = 5}$$

---

### Part A(iv): Skip-gram Training Set (2 marks)

Skip-gram: center word → each individual context word (one pair per context word).

| Center Word | Context Word | Training Pair    |
|-------------|:------------:|:----------------:|
| $w_1$       | $w_2$        | $(w_1, w_2)$     |
| $w_2$       | $w_1$        | $(w_2, w_1)$     |
| $w_2$       | $w_3$        | $(w_2, w_3)$     |
| $w_3$       | $w_2$        | $(w_3, w_2)$     |
| $w_3$       | $w_4$        | $(w_3, w_4)$     |
| $w_4$       | $w_3$        | $(w_4, w_3)$     |
| $w_4$       | $w_5$        | $(w_4, w_5)$     |
| $w_5$       | $w_4$        | $(w_5, w_4)$     |

$$\boxed{\text{Skip-gram training set size} = 8}$$

---

### Part C: Benefit of GloVe over Word2Vec (1 mark)

GloVe leverages **global co-occurrence statistics** from the entire corpus (via a co-occurrence matrix), whereas word2vec only uses **local context windows**. This allows GloVe to capture both local and global word relationships more effectively, often leading to better performance on word analogy and similarity tasks.

---

### Part D: Benefit of CoVe over GloVe/Word2Vec (1 mark)

CoVe produces **contextualized word embeddings** — the same word gets different representations depending on its surrounding sentence context (via a BiLSTM encoder). In contrast, GloVe and word2vec produce **static embeddings** where each word always maps to the same vector regardless of context. This enables CoVe to handle polysemy (e.g., "bank" as financial institution vs. river bank).

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 3 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q3. Lipschitz Constants, Wasserstein Distance & Divergences

**Marks:** [3+3+4=10] | **Source:** Makeup EC-3 Regular 2024-25

**A.** Calculate the Lipschitz constant for the following functions? Which of these can be used in WGAN? Show all steps clearly.

**(i)** $f(x) = e^{x \cdot x}$

**(ii)** $f(x) = x^2$ on $[-1, 1]$.

**(3 marks)**

---

**B.** Let $\mu_1 = \mathcal{N}(3, 2)$ and $\mu_2 = \mathcal{N}(4, 1)$ be two normal distributions. Calculate the 2-Wasserstein distance. Note p-Wasserstein distance for two empirical distributions P and Q is denoted by

$$W_p(P, Q) = \left( \frac{1}{n} \sum_{i=1}^{n} \|X_{(i)} - Y_{(i)}\|^p \right)^{1/p}$$

Note 1-Wasserstein distance is also known as earth mover's distance! **(3 marks)**

---

**C.** Let $\mu_1 = \mathcal{N}(3, 2)$ and $\mu_2 = \mathcal{N}(4, 1)$ be two normal distributions. Calculate

**(i)** $\text{KLD}(\mu_1 \| \mu_2)$ **(1 mark)**

**(ii)** Will $\text{KLD}(\mu_1 \| \mu_2)$ and $\text{KLD}(\mu_2 \| \mu_1)$ be the same or different? Show mathematically. **(1 mark)**

**(iii)** Will the JS Divergence $\text{JSD}(\mu_1 \| \mu_2)$ and $\text{JSD}(\mu_2 \| \mu_1)$ be same or different? Show mathematically. **(2 marks)**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 3 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q3. Topics to Know

To answer this question, study the following:

- **Lipschitz Continuity** — 📖 [7.4.1 Wasserstein GAN](../study/07-generative-adversarial-network.md#741-wasserstein-gan-wgan)
  - A function $f$ is $K$-Lipschitz if $|f(x) - f(y)| \leq K|x - y|$ for all $x, y$
  - Lipschitz constant = $\sup |f'(x)|$ (supremum of the absolute derivative)
  - WGAN requires 1-Lipschitz functions (K ≤ 1)
  - Unbounded derivatives → not Lipschitz

- **Wasserstein Distance (for Gaussians)** — 📖 [7.4.1 Wasserstein GAN](../study/07-generative-adversarial-network.md#741-wasserstein-gan-wgan)
  - For 1D Gaussians: $W_2(\mathcal{N}(\mu_1, \sigma_1^2), \mathcal{N}(\mu_2, \sigma_2^2)) = \sqrt{(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2}$

- **KL Divergence (for Gaussians)** — 📖 [6.3 ELBO](../study/06-variational-inferencing.md#63-variational--evidence-lower-bound-elbo) · [6.4 KL Divergence](../study/06-variational-inferencing.md#kl-divergence-gaussian-case-closed-form)
  - $\text{KLD}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$
  - KLD is **asymmetric**: $\text{KLD}(P\|Q) \neq \text{KLD}(Q\|P)$ in general

- **Jensen-Shannon Divergence** — 📖 [7.2 Minimax Optimization](../study/07-generative-adversarial-network.md#72-minimax-optimization)
  - $\text{JSD}(P\|Q) = \frac{1}{2}\text{KLD}(P\|M) + \frac{1}{2}\text{KLD}(Q\|M)$ where $M = \frac{1}{2}(P + Q)$
  - JSD is **symmetric**: $\text{JSD}(P\|Q) = \text{JSD}(Q\|P)$ by definition

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 3 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q3. Solution

### Part A: Lipschitz Constants (3 marks)

**(i)** $f(x) = e^{x \cdot x} = e^{x^2}$

$$f'(x) = 2x \cdot e^{x^2}$$

As $x \to \infty$, $|f'(x)| = 2|x| \cdot e^{x^2} \to \infty$

The derivative is **unbounded**, so $f(x) = e^{x^2}$ is **not Lipschitz continuous**. ❌ **Cannot be used in WGAN.**

---

**(ii)** $f(x) = x^2$ on $[-1, 1]$

$$f'(x) = 2x$$

$$\sup_{x \in [-1,1]} |f'(x)| = \sup_{x \in [-1,1]} |2x| = 2 \quad \text{(at } x = \pm 1\text{)}$$

$$\boxed{K = 2}$$

The function is **2-Lipschitz** on $[-1, 1]$. Since WGAN requires **1-Lipschitz** ($K \leq 1$), this function as-is ❌ **cannot be directly used in WGAN** without rescaling (dividing by 2 would make it 1-Lipschitz).

---

### Part B: 2-Wasserstein Distance (3 marks)

For two 1D Gaussians, the closed-form 2-Wasserstein distance is:

$$W_2(\mathcal{N}(\mu_1, \sigma_1^2), \mathcal{N}(\mu_2, \sigma_2^2)) = \sqrt{(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2}$$

Given $\mu_1 = \mathcal{N}(3, 2)$ and $\mu_2 = \mathcal{N}(4, 1)$:

Here $\mu_1 = 3$, $\sigma_1^2 = 2 \Rightarrow \sigma_1 = \sqrt{2}$, and $\mu_2 = 4$, $\sigma_2^2 = 1 \Rightarrow \sigma_2 = 1$.

$$W_2 = \sqrt{(3 - 4)^2 + (\sqrt{2} - 1)^2} = \sqrt{1 + (1.4142 - 1)^2} = \sqrt{1 + (0.4142)^2}$$

$$W_2 = \sqrt{1 + 0.1716} = \sqrt{1.1716}$$

$$\boxed{W_2 \approx 1.0824}$$

---

### Part C(i): $\text{KLD}(\mu_1 \| \mu_2)$ (1 mark)

For univariate Gaussians:

$$\text{KLD}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

With $\mu_1 = 3$, $\sigma_1^2 = 2$, $\mu_2 = 4$, $\sigma_2^2 = 1$:

$$\text{KLD}(\mu_1 \| \mu_2) = \log\frac{1}{\sqrt{2}} + \frac{2 + (3 - 4)^2}{2 \times 1} - \frac{1}{2}$$

$$= \log\frac{1}{\sqrt{2}} + \frac{2 + 1}{2} - \frac{1}{2} = -\frac{1}{2}\log 2 + \frac{3}{2} - \frac{1}{2}$$

$$= -0.3466 + 1.5 - 0.5$$

$$\boxed{\text{KLD}(\mu_1 \| \mu_2) = 0.6534}$$

---

### Part C(ii): Is $\text{KLD}(\mu_1 \| \mu_2) = \text{KLD}(\mu_2 \| \mu_1)$? (1 mark)

Compute $\text{KLD}(\mu_2 \| \mu_1)$ with $\mu_2 = 4$, $\sigma_2^2 = 1$, $\mu_1 = 3$, $\sigma_1^2 = 2$:

$$\text{KLD}(\mu_2 \| \mu_1) = \log\frac{\sqrt{2}}{1} + \frac{1 + (4 - 3)^2}{2 \times 2} - \frac{1}{2}$$

$$= \frac{1}{2}\log 2 + \frac{1 + 1}{4} - \frac{1}{2} = 0.3466 + 0.5 - 0.5$$

$$\boxed{\text{KLD}(\mu_2 \| \mu_1) = 0.3466}$$

Since $0.6534 \neq 0.3466$, **KLD is asymmetric**: $\text{KLD}(\mu_1 \| \mu_2) \neq \text{KLD}(\mu_2 \| \mu_1)$.

---

### Part C(iii): Is $\text{JSD}(\mu_1 \| \mu_2) = \text{JSD}(\mu_2 \| \mu_1)$? (2 marks)

By definition:

$$\text{JSD}(P \| Q) = \frac{1}{2}\text{KLD}(P \| M) + \frac{1}{2}\text{KLD}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$.

Now consider $\text{JSD}(Q \| P)$:

$$\text{JSD}(Q \| P) = \frac{1}{2}\text{KLD}(Q \| M') + \frac{1}{2}\text{KLD}(P \| M')$$

where $M' = \frac{1}{2}(Q + P) = \frac{1}{2}(P + Q) = M$.

Therefore:

$$\text{JSD}(Q \| P) = \frac{1}{2}\text{KLD}(Q \| M) + \frac{1}{2}\text{KLD}(P \| M) = \text{JSD}(P \| Q)$$

$$\boxed{\text{JSD}(\mu_1 \| \mu_2) = \text{JSD}(\mu_2 \| \mu_1) \text{ — JSD is symmetric by definition.}}$$

This is because addition is commutative ($P + Q = Q + P$), so the mixture $M$ is identical regardless of argument order, and the two KLD terms simply swap order in the sum.

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 4 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q4. EBM Normalization, GAN Discriminator & VAE Loss

**Marks:** [3+2+2+3=10] | **Source:** Makeup EC-3 Regular 2024-25

**A.** What normalization is needed for a function $f(x) = \exp\left[-\frac{(x - 2.5)^2}{9}\right]$ to be used as the partition function in an energy-based generative model? **(3 marks)**

**B.** Let $\mu_{\text{data}} = \mathcal{N}(3, 2)$ and $\mu_g = \mathcal{N}(4, 1)$ be the training data distribution and generated data distribution, respectively. What is the optimal discriminator function for a GAN? **(2 marks)**

**C.** How is clipping related disadvantages of WGAN eliminated? Which architecture incorporates this strategy? **(2 marks)**

**D.** You are training a simple VAE on 1-dimensional data using a latent space of size 1.

For a single training example, the encoder outputs the following parameters for the latent variable $z$:

- Mean $\mu = 1.0$
- Log-variance $\log(\sigma^2) = -0.5$

The decoder reconstructs the input with:

- Reconstructed output $\hat{x} = 0.8$
- Original input $x = 1.0$

Calculate the reconstruction loss and total loss (sum of reconstruction loss and KL divergence loss). **(3 marks)**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 4 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q4. Topics to Know

To answer this question, study the following:

- **Energy-Based Models — Partition Function** — 📖 [9.1 Parametrizing Probability Distributions](../study/09-energy-score-based-models.md#91-parametrizing-probability-distributions)
  - Unnormalized density: $\tilde{p}(x) = e^{-E(x)}$
  - Partition function: $Z = \int e^{-E(x)}\, dx$
  - Normalized probability: $p(x) = \tilde{p}(x) / Z$
  - Recognizing Gaussian form: $e^{-(x-\mu)^2/(2\sigma^2)}$ has $Z = \sigma\sqrt{2\pi}$

- **Optimal GAN Discriminator** — 📖 [7.2 Minimax: Optimal Discriminator](../study/07-generative-adversarial-network.md#optimal-discriminator)
  - $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$

- **WGAN-GP (Gradient Penalty)** — 📖 [7.4.1 Wasserstein GAN](../study/07-generative-adversarial-network.md#741-wasserstein-gan-wgan)
  - Weight clipping issues: capacity underuse, exploding/vanishing gradients
  - Gradient penalty: $\lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$
  - Enforces Lipschitz constraint without clipping

- **VAE Loss Function** — 📖 [6.3 ELBO](../study/06-variational-inferencing.md#63-variational--evidence-lower-bound-elbo) · [6.4 KL Divergence](../study/06-variational-inferencing.md#kl-divergence-gaussian-case-closed-form)
  - Reconstruction loss (MSE): $L_{\text{recon}} = \frac{1}{2}(x - \hat{x})^2$
  - KL divergence (1D): $L_{\text{KL}} = -\frac{1}{2}(1 + \log(\sigma^2) - \mu^2 - \sigma^2)$
  - Total: $L = L_{\text{recon}} + L_{\text{KL}}$

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 4 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q4. Solution

### Part A: EBM Partition Function Normalization (3 marks)

The given function:

$$f(x) = \exp\left[-\frac{(x - 2.5)^2}{9}\right] = \exp\left[-\frac{(x - 2.5)^2}{2 \times 4.5}\right]$$

This is an **unnormalized Gaussian** with $\mu = 2.5$ and $\sigma^2 = 4.5$ (i.e., $\sigma = \sqrt{4.5} = \frac{3}{\sqrt{2}}$).

The partition function (normalization constant):

$$Z = \int_{-\infty}^{\infty} \exp\left[-\frac{(x - 2.5)^2}{9}\right] dx$$

Using the Gaussian integral $\int e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx = \sigma\sqrt{2\pi}$:

$$Z = \sqrt{4.5} \cdot \sqrt{2\pi} = \frac{3}{\sqrt{2}} \cdot \sqrt{2\pi} = 3\sqrt{\pi}$$

$$\boxed{Z = 3\sqrt{\pi} \approx 5.317}$$

The normalized density: $p(x) = \frac{1}{3\sqrt{\pi}} \exp\left[-\frac{(x - 2.5)^2}{9}\right]$

---

### Part B: Optimal Discriminator (2 marks)

The optimal discriminator for a GAN is:

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

With $p_{\text{data}}(x) = \mathcal{N}(3, 2)$ and $p_g(x) = \mathcal{N}(4, 1)$:

$$\boxed{D^*(x) = \frac{\frac{1}{\sqrt{4\pi}} e^{-\frac{(x-3)^2}{4}}}{\frac{1}{\sqrt{4\pi}} e^{-\frac{(x-3)^2}{4}} + \frac{1}{\sqrt{2\pi}} e^{-\frac{(x-4)^2}{2}}}}$$

This is a **sigmoid-like function** of $x$ that outputs values closer to 1 where the real data density dominates, and closer to 0 where the generated data density dominates.

---

### Part C: Eliminating Weight Clipping Disadvantages (2 marks)

Weight clipping in WGAN causes:
- **Capacity underuse** — pushes weights toward the clipping boundaries
- **Exploding/vanishing gradients** — sensitive to clipping threshold

These are eliminated by replacing weight clipping with a **gradient penalty** term:

$$\lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$

where $\hat{x}$ is sampled along straight lines between real and generated data points.

This architecture is called **WGAN-GP (Wasserstein GAN with Gradient Penalty)**.

---

### Part D: VAE Reconstruction Loss and Total Loss (3 marks)

**Reconstruction loss (MSE):**

$$L_{\text{recon}} = \frac{1}{2}(x - \hat{x})^2 = \frac{1}{2}(1.0 - 0.8)^2 = \frac{1}{2}(0.04) = 0.02$$

**KL divergence loss (1D):**

$$L_{\text{KL}} = -\frac{1}{2}\left(1 + \log(\sigma^2) - \mu^2 - \sigma^2\right)$$

Given $\mu = 1.0$, $\log(\sigma^2) = -0.5$, so $\sigma^2 = e^{-0.5} = 0.6065$:

$$L_{\text{KL}} = -\frac{1}{2}(1 + (-0.5) - 1.0^2 - 0.6065)$$

$$= -\frac{1}{2}(1 - 0.5 - 1.0 - 0.6065)$$

$$= -\frac{1}{2}(-1.1065) = 0.5533$$

**Total loss:**

$$L = L_{\text{recon}} + L_{\text{KL}} = 0.02 + 0.5533$$

$$\boxed{L = 0.5733}$$

<div style="page-break-after: always;"></div>

---

## Navigation

- [Questions Index](./)
- [Study](../study/)
- [Course Home](../)
- [Back to Homepage](../../)
