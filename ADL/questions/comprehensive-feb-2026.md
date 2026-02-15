# ADL — Comprehensive February 2026

---

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. CNN-based VAE for MNIST

**Marks:** [4+4+3+1=12] | **Source:** Feb 2026 Comprehensive

**A.** You are implementing a **CNN-based VAE** for the **MNIST dataset** (28×28 grayscale images, 1 channel). The encoder architecture is defined as follows:

**Input shape:** (1, 28, 28)

1. **Conv2D Layer 1:** Filters: 32, Kernel: 3×3, Stride: 1, Padding: same, Activation: ReLU

2. **Conv2D Layer 2:** Filters: 64, Kernel: 3×3, Stride: 2, Padding: same, Activation: ReLU

3. **Flatten Layer**

4. **Dense Layer:** Outputs 128 units, ReLU activation

5. **Latent space:**
   - Mean layer ($\mu$): Dense layer → 10 units
   - Log-variance layer ($\log(\sigma^2)$): Dense layer → 10 units

The decoder architecture is defined as follows:

1. **Dense Layer:** Input = 10 → Output = 7×7×64 units, ReLU activation
2. **Reshape Layer:** Reshape to (64, 7, 7)

3. **ConvTranspose2D Layer 1:** Filters: 64, Kernel: 3×3, Stride: 2, Padding: same, Output padding: 1, Activation: ReLU

4. **ConvTranspose2D Layer 2:** Filters: 32, Kernel: 3×3, Stride: 1, Padding: same, Activation: ReLU

5. **ConvTranspose2D Layer 3 (Output):** Filters: 1, Kernel: 3×3, Stride: 1, Padding: same, Activation: Sigmoid

**(i)** What is the total number of trainable parameters in Encoder? Show all steps. **(4 marks)**

**(ii)** What is the total number of trainable parameters in Decoder? Show all steps. **(4 marks)**

---

**B.** You're working on training a Variational Autoencoder (VAE) that uses a Gaussian latent space with diagonal covariance. During training, for a single input image, the encoder network outputs the following:

Mean vector ($\mu$): $[1.5, -0.5, 0.0, 2.0]$

Log-variance vector ($\log(\sigma^2)$): $[0.0, \log(0.25), \log(1.0), \log(4.0)]$

**(i)** Suppose a random vector $\varepsilon$ is sampled from a standard normal distribution: $\varepsilon = [0.2, -1.0, 0.0, 1.5]$. Use the reparameterization trick to compute the latent vector $z$. **(3 marks)**

**(ii)** Why is the reparameterization trick necessary in VAE? **(1 mark)**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Topics to Know

To answer this question, study the following:

- **CNN Architecture & Parameter Counting**
  - Conv2D parameters: $(K_h \times K_w \times C_{in} + 1) \times C_{out}$ (weights + biases)
  - Dense layer parameters: $(D_{in} + 1) \times D_{out}$
  - How `padding: same` and `stride` affect spatial dimensions
  - Output size formula: $\lfloor \frac{n + 2p - k}{s} \rfloor + 1$

- **Transposed Convolution (ConvTranspose2D)**
  - Output size: $(H_{in} - 1) \times s - 2p + k + p_{out}$
  - Parameter count same formula as Conv2D: $(K_h \times K_w \times C_{in} + 1) \times C_{out}$

- **Variational Autoencoder (VAE) Architecture**
  - Encoder: maps input → latent distribution parameters ($\mu$, $\log \sigma^2$)
  - Decoder: maps latent vector $z$ → reconstructed output
  - Latent space: defined by mean and log-variance vectors

- **Reparameterization Trick**
  - Formula: $z = \mu + \sigma \odot \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, I)$
  - Converting $\log(\sigma^2)$ to $\sigma$: $\sigma = e^{\frac{1}{2}\log(\sigma^2)}$
  - **Why it's needed:** Enables backpropagation through the stochastic sampling step by making the randomness external to the computation graph

- **Flatten Layer**
  - Total units = $C \times H \times W$ after final conv layer

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Solution

### Part A(i): Encoder Trainable Parameters (4 marks)

**Input:** (1, 28, 28)

**Layer 1 — Conv2D (32 filters, 3×3, stride 1, same padding):**

- Parameters = $(3 \times 3 \times 1 + 1) \times 32 = 10 \times 32 = \mathbf{320}$
- Output shape: (32, 28, 28) — same padding with stride 1 preserves dimensions

**Layer 2 — Conv2D (64 filters, 3×3, stride 2, same padding):**

- Parameters = $(3 \times 3 \times 32 + 1) \times 64 = 289 \times 64 = \mathbf{18{,}496}$
- Output shape: (64, 14, 14) — stride 2 halves spatial dimensions: $\lfloor 28/2 \rfloor = 14$

**Layer 3 — Flatten:**

- No parameters
- Output: $64 \times 14 \times 14 = 12{,}544$ units

**Layer 4 — Dense (128 units):**

- Parameters = $(12{,}544 + 1) \times 128 = 12{,}545 \times 128 = \mathbf{1{,}605{,}760}$

**Layer 5a — Mean layer (10 units):**

- Parameters = $(128 + 1) \times 10 = 129 \times 10 = \mathbf{1{,}290}$

**Layer 5b — Log-variance layer (10 units):**

- Parameters = $(128 + 1) \times 10 = 129 \times 10 = \mathbf{1{,}290}$

**Total Encoder Parameters:**

$$320 + 18{,}496 + 1{,}605{,}760 + 1{,}290 + 1{,}290 = \boxed{1{,}627{,}156}$$

---

### Part A(ii): Decoder Trainable Parameters (4 marks)

**Input:** latent vector of size 10

**Layer 1 — Dense (7×7×64 = 3,136 units):**

- Parameters = $(10 + 1) \times 3{,}136 = 11 \times 3{,}136 = \mathbf{34{,}496}$
- Output: 3,136 → Reshaped to (64, 7, 7)

**Layer 2 — Reshape:**

- No parameters

**Layer 3 — ConvTranspose2D (64 filters, 3×3, stride 2, same, output padding 1):**

- Parameters = $(3 \times 3 \times 64 + 1) \times 64 = 577 \times 64 = \mathbf{36{,}928}$
- Output shape: (64, 14, 14)

**Layer 4 — ConvTranspose2D (32 filters, 3×3, stride 1, same):**

- Parameters = $(3 \times 3 \times 64 + 1) \times 32 = 577 \times 32 = \mathbf{18{,}464}$
- Output shape: (32, 14, 14)

**Layer 5 — ConvTranspose2D (1 filter, 3×3, stride 1, same):**

- Parameters = $(3 \times 3 \times 32 + 1) \times 1 = 289 \times 1 = \mathbf{289}$
- Output shape: (1, 14, 14)

> **Note:** With stride-1 and same padding in layers 4 and 5, the spatial dimensions remain 14×14, not 28×28. The only upsampling happens in Layer 3 (stride 2: 7→14). The final output would be (1, 14, 14), which doesn't match the original 28×28 input. This suggests the problem may expect ConvTranspose2D Layer 2 to also use stride 2 to achieve 28×28, or there's an implicit upsampling step. Following the question as stated:

**Total Decoder Parameters:**

$$34{,}496 + 36{,}928 + 18{,}464 + 289 = \boxed{90{,}177}$$

---

### Part B(i): Reparameterization Trick — Compute $z$ (3 marks)

Given:
- $\mu = [1.5, -0.5, 0.0, 2.0]$
- $\log(\sigma^2) = [0.0, \log(0.25), \log(1.0), \log(4.0)]$
- $\varepsilon = [0.2, -1.0, 0.0, 1.5]$

**Step 1:** Compute $\sigma$ from $\log(\sigma^2)$:

$$\sigma = e^{\frac{1}{2}\log(\sigma^2)}$$

| Dimension | $\log(\sigma^2)$ | $\frac{1}{2}\log(\sigma^2)$ | $\sigma = e^{(\cdot)}$ |
|-----------|-------------------|------------------------------|-------------------------|
| 1         | $0.0$             | $0.0$                        | $1.0$                   |
| 2         | $\log(0.25)$      | $\frac{1}{2}\log(0.25)$     | $\sqrt{0.25} = 0.5$    |
| 3         | $\log(1.0)$       | $0.0$                        | $1.0$                   |
| 4         | $\log(4.0)$       | $\frac{1}{2}\log(4.0)$      | $\sqrt{4.0} = 2.0$     |

So $\sigma = [1.0, 0.5, 1.0, 2.0]$

**Step 2:** Apply reparameterization trick $z = \mu + \sigma \odot \varepsilon$:

| Dim | $\mu_i$ | $\sigma_i$ | $\varepsilon_i$ | $z_i = \mu_i + \sigma_i \cdot \varepsilon_i$ |
|-----|---------|------------|------------------|----------------------------------------------|
| 1   | $1.5$   | $1.0$      | $0.2$            | $1.5 + 1.0 \times 0.2 = \mathbf{1.7}$       |
| 2   | $-0.5$  | $0.5$      | $-1.0$           | $-0.5 + 0.5 \times (-1.0) = \mathbf{-1.0}$  |
| 3   | $0.0$   | $1.0$      | $0.0$            | $0.0 + 1.0 \times 0.0 = \mathbf{0.0}$       |
| 4   | $2.0$   | $2.0$      | $1.5$            | $2.0 + 2.0 \times 1.5 = \mathbf{5.0}$       |

$$\boxed{z = [1.7, -1.0, 0.0, 5.0]}$$

---

### Part B(ii): Why is the Reparameterization Trick Necessary? (1 mark)

The reparameterization trick is necessary because **direct sampling from a probability distribution is a non-differentiable operation**, which prevents backpropagation through the stochastic latent layer.

By rewriting $z = \mu + \sigma \odot \varepsilon$ (where $\varepsilon \sim \mathcal{N}(0, I)$ is sampled externally), the randomness is moved outside the computational graph. This makes $z$ a **deterministic, differentiable function** of $\mu$ and $\sigma$, allowing gradients to flow from the decoder loss back through $\mu$ and $\sigma$ to the encoder weights.

<div style="page-break-after: always;"></div>
