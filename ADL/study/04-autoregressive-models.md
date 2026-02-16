# Module 4 — Autoregressive Models

## Topics

- [[#4.1 Motivation|Motivation]]
- [[#4.2 Simple Generative Models: Histograms|Simple generative models: histograms]]
- [[#4.3 Parameterized Distributions and Maximum Likelihood|Parameterized distributions and maximum likelihood]]
- [[#4.4 Recurrent Neural Nets for AR Models|Recurrent Neural Nets]]
- [[#4.5 Masking-Based Models|Masking-based Models]]
  - [[#4.5.1 MADE (Masked Autoencoder for Distribution Estimation)|Masked AEs for Distribution Estimation (MADE)]]
  - [[#4.5.2 Masked Convolutions|Masked Convolutions]]
    - [[#WaveNet|WaveNet]]
    - [[#PixelCNN and Variants|PixelCNN and Variants]]
- [[#4.6 Applications|Applications in super-resolution, colorization]]
- [[#4.7 Speed Up Strategies|Speed Up Strategies]]

---

## 4.1 Motivation

- Goal: model the joint distribution $p(x_1, x_2, \ldots, x_D)$ over all dimensions of $x$
- Use the **chain rule of probability** to decompose into conditional distributions:

$$p(x) = \prod_{d=1}^{D} p(x_d | x_1, x_2, \ldots, x_{d-1})$$

- Each conditional can be modeled by a neural network
- **Autoregressive**: each output depends on previous outputs (sequential generation)
- Advantages: **exact likelihood computation**, tractable training
- Disadvantage: **slow sequential generation**

---

## 4.2 Simple Generative Models: Histograms

- Simplest density estimator: **histogram** over discretized data
- Divide each dimension into bins and count frequencies
- Problems:
  - **Curse of dimensionality**: $B^D$ bins for $B$ bins per dimension and $D$ dimensions
  - No parameter sharing across dimensions
  - Poor generalization

---

## 4.3 Parameterized Distributions and Maximum Likelihood

- Replace histograms with **parameterized models** $p_\theta(x)$
- **Maximum Likelihood Estimation (MLE)**: find parameters that maximize the probability of observed data:

$$\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(x_i)$$

- Equivalent to minimizing **KL divergence** between data distribution and model:

$$\min_\theta KL(p_{\text{data}} \| p_\theta)$$

- AR models parameterize each conditional $p_\theta(x_d | x_{<d})$ with a neural network

### MLE for Bernoulli Distribution

For binary data $x_i \in \{0, 1\}$ with success probability $p$:

- **Likelihood:** $L(p) = p^k (1 - p)^{n - k}$ where $k$ = number of 1s, $n$ = total observations
- **Log-likelihood:** $\ell(p) = k \log p + (n - k) \log(1 - p)$
- **MLE:** Set $\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$, solve:

$$\hat{p} = \frac{k}{n}$$

The MLE estimate is simply the **sample proportion** of successes.

---

## 4.4 Recurrent Neural Nets for AR Models

- Use RNN/LSTM/GRU to model the sequential dependencies
- Hidden state $h_t$ summarizes all previous observations:

$$h_t = f(h_{t-1}, x_{t-1})$$
$$p(x_t | x_{<t}) = g(h_t)$$

- Natural fit for sequential data (text, audio, time series)
- Limitations:
  - Training is sequential (cannot parallelize across time steps)
  - Long-range dependencies are difficult (despite LSTM/GRU)

---

## 4.5 Masking-Based Models

### 4.5.1 MADE (Masked Autoencoder for Distribution Estimation)

- Transform a standard autoencoder into an **autoregressive model** using **masks**
- Assign each hidden unit a number $m(k) \in \{1, \ldots, D-1\}$
- Mask weight connections so that output $\hat{x}_d$ only depends on inputs $x_1, \ldots, x_{d-1}$
- Mask condition: $M^W_{kd} = \mathbb{1}[m(k) \geq d]$ for input-to-hidden weights
- **Key advantage**: single forward pass computes all $D$ conditionals simultaneously (unlike RNN)
- Can use **order-agnostic training**: random orderings during training

### 4.5.2 Masked Convolutions

- Apply autoregressive property to **convolutional architectures**
- Mask the convolutional filters to ensure each output pixel only depends on previously generated pixels

#### WaveNet
- 1D autoregressive model for **raw audio generation**
- **Dilated causal convolutions**: exponentially increasing dilation for large receptive fields
  - Dilation rates: 1, 2, 4, 8, 16, ... → receptive field grows exponentially with depth
- **Causal**: output at time $t$ only depends on inputs at times $\leq t$
- Uses **gated activation units**: $\tanh(W_f * x) \odot \sigma(W_g * x)$
- Residual and skip connections for training deep networks
- Applications: speech synthesis (text-to-speech), music generation

#### PixelCNN and Variants
- 2D autoregressive model for **images**
- Generates images pixel by pixel (raster scan order: left-to-right, top-to-bottom)
- Uses **masked convolutions** (Type A and Type B masks):
  - **Type A mask**: excludes current pixel — used in first layer
  - **Type B mask**: includes current pixel — used in subsequent layers
- **Gated PixelCNN**: uses gated activation units + separate vertical and horizontal stacks
  - Solves the blind spot problem of vanilla PixelCNN
- **PixelCNN++**: logistic mixture likelihood, downsampling, short-cut connections

---

## 4.6 Applications

| Application | Description |
|-------------|-------------|
| **Super-resolution** | Generate high-res images conditioned on low-res input |
| **Colorization** | Generate color channels conditioned on grayscale input |
| **Image inpainting** | Fill in missing regions conditioned on known pixels |
| **Speech synthesis** | WaveNet for text-to-speech |
| **Music generation** | WaveNet-style models for raw audio |
| **Text generation** | Language models (GPT family) |

---

## 4.7 Speed Up Strategies

| Strategy | Description |
|----------|-------------|
| **Caching** | Cache hidden states from previous steps to avoid recomputation |
| **Fast WaveNet** | $O(L)$ generation instead of $O(L \cdot \text{receptive field})$ using caching |
| **Parallel generation** | Teacher-student distillation (Parallel WaveNet) |
| **Subscale pixel ordering** | Generate at multiple scales simultaneously |
| **Multi-scale AR models** | Hierarchical generation at different resolutions |
| **Knowledge distillation** | Train a faster non-AR student from AR teacher |

---

## Key Takeaways

- AR models decompose joint distribution using the chain rule → **exact likelihood**
- MADE makes autoencoders autoregressive via masking → parallel training
- WaveNet uses **dilated causal convolutions** for audio with large receptive fields
- PixelCNN uses **masked 2D convolutions** for image generation
- Generation is inherently **sequential and slow** — speed-up strategies are critical
- Foundation for understanding normalizing flows (Module 5) and transformers (Module 10)

---

## References

- T1: Prince, Ch. 12 — Autoregressive Models
- T2: Goodfellow et al., Ch. 20.10 — Directed Generative Nets
