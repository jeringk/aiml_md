# Question Template

> This file serves as a **reference template** for formatting questions across all courses.  
> Copy this structure when adding new questions. Each question spans **3 printed pages**.

---

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 1: QUESTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Example Question Title

**Marks:** 10 | **Source:** June 2025 Mid-Sem / Generated

Write the full question text here. Include any sub-parts as needed:

**(a)** Describe the concept of backpropagation in neural networks. **(3 marks)**

**(b)** Given a neural network with the following architecture, compute the forward pass output:

$$z = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$$

where $\sigma$ is the sigmoid activation function, $W_1 \in \mathbb{R}^{4 \times 3}$, $W_2 \in \mathbb{R}^{2 \times 4}$, and $x \in \mathbb{R}^3$. **(4 marks)**

**(c)** Explain the vanishing gradient problem and how ReLU activation helps mitigate it. **(3 marks)**

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 2: TOPICS TO KNOW                    -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Topics to Know

To answer this question, study the following:

- **Backpropagation**
  - Chain rule for computing gradients
  - Computational graph representation
  - Gradient flow through layers

- **Forward Pass Computation**
  - Matrix multiplication: $z = Wx + b$
  - Sigmoid activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$
  - Layer-by-layer computation

- **Vanishing Gradient Problem**
  - Why sigmoid/tanh cause vanishing gradients
  - Gradient magnitude through deep networks
  - ReLU: $f(x) = \max(0, x)$ and its derivative
  - Variants: Leaky ReLU, ELU, GELU

<div style="page-break-after: always;"></div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- QUESTION 1 — PAGE 3: SOLUTION                          -->
<!-- ═══════════════════════════════════════════════════════ -->

## Q1. Solution

### (a) Backpropagation (3 marks)

Backpropagation is an algorithm for computing gradients of the loss function with respect to all weights in a neural network using the **chain rule**.

For a loss $L$ and weight $w_{ij}$:

$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_j} \cdot \frac{\partial a_j}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}$$

where $a_j$ is the activation output and $z_j$ is the pre-activation value.

### (b) Forward Pass (4 marks)

**Step 1:** Compute hidden layer pre-activation:

$$h = W_1 \cdot x + b_1$$

**Step 2:** Apply sigmoid activation:

$$a = \sigma(h) = \frac{1}{1 + e^{-h}}$$

**Step 3:** Compute output:

$$z = W_2 \cdot a + b_2$$

### (c) Vanishing Gradient Problem (3 marks)

The sigmoid function saturates for large $|x|$, producing derivatives close to zero:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x)) \leq 0.25$$

In deep networks, these small gradients multiply through layers:

$$\frac{\partial L}{\partial w^{(1)}} = \prod_{l=1}^{L} \sigma'(z^{(l)}) \cdot \frac{\partial L}{\partial a^{(L)}} \approx 0$$

**ReLU** mitigates this because its gradient is either 0 or 1:

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

This prevents gradient decay through active neurons.

<div style="page-break-after: always;"></div>

---

## Navigation

- [Questions Index](./)
- [Study](../study/)
- [Course Home](../)
- [Back to Homepage](../../)

<!-- ═══════════════════════════════════════════════════════ -->
<!-- ADD MORE QUESTIONS BELOW FOLLOWING THE SAME PATTERN    -->
<!-- Q2, Q3, Q4... each with 3 pages                        -->
<!-- ═══════════════════════════════════════════════════════ -->
