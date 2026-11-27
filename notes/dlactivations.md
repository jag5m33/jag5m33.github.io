# Activation functions

Sigmoid, tanh, ReLU and leaky ReLU: what each one does, its pros and cons, and which to use where.

**On this page**

- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Which activation function to use

![Top row: each function. Bottom row: its derivative. Sigmoid's derivative peaks at 0.25, tanh's at 1, ReLU's is 0 or 1, leaky ReLU's is 0.01 or 1.](figures/dl-activations.svg)

## 1) Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Output between 0 and 1, crossing 0.5 at $x = 0$.
- **+** Smooth gradient = better output value.
- Derivative $\dfrac{d\sigma(x)}{dx}$: **0 → 0.25** → the vanishing gradient problem.

**Disadvantages:**

- Exponential function, so calculation time ↑: **time complexity ↑**.
- **Not 0 centred**: doesn't pass through the origin.
- Prone to the vanishing gradient problem.
- Poor for **deep** neural networks: they're complex, lots of back propagation = **lose gradient**:

$$w_{new} \approx w_{old} - \underbrace{\alpha}_{\text{small}} \underbrace{\frac{dL}{dw}}_{\text{small}} \to \text{tiny}$$

> **Check:** in my notes it says "prone to gradient descent". It means prone to the **vanishing gradient** problem: gradient descent is the training method itself.

## 2) Tanh

$$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

- Output (y axis): **−1 → +1**, and it passes through 0.
- Derivative $\dfrac{d\tanh(x)}{dx}$: **0 → 1**.

**Advantages and disadvantages:**

- Better at avoiding the vanishing gradient than sigmoid:

| | Derivative range |
|---|---|
| Sigmoid | 0 → 0.25 |
| Tanh | 0 → 1: lasts longer for back propagation |

- Still poor though (for very deep networks).
- In **binary classification**: use tanh for the hidden layers and sigmoid for the output.

> **Check:** my notes say "better at avoiding gradient descent" here too. It means better at avoiding the **vanishing gradient**.

## 3) ReLU (rectified linear unit)

$$\text{ReLU}(x) = \max(0, x)$$

- Whenever $x \le 0$, $y = 0$; else $y = x$.
- Derivative $\dfrac{d\,\text{ReLU}(x)}{dx}$: either **0 or 1**.

**The dead neuron:** if the input is negative, the derivative = 0:

$$w_{new} = w_{old} - \alpha \cdot 0 = w_{old}$$

Plateau, no change: the **neuron dies**. When the input is negative, ReLU will die.

**But:**

- Faster in time complexity
- Faster computation, not limited
- No vanishing gradient
- Not 0 centred

> **Check:** "not 0 centred" is listed with the advantages in my notes, but it's a **disadvantage** (the same one sigmoid has).

## 4) Leaky ReLU

$$\text{Leaky ReLU}(x) = \max(0.01x, \; x)$$

- For positive $x$: max = $x$, so $y = x$.
- For negative $x$: the minimum is **0.01x**, not 0 like ReLU. The output is a bit higher due to the min being 0.01x instead of 0, so the small number is still valid.
- → **Solves the dead neuron problem.**

## Which activation function to use

| Problem | Hidden layers | Output layer |
|---|---|---|
| **Binary classification** | ReLU | Sigmoid |
| **Multiclass classification** | ReLU (the default) | Softmax |
| **Regression** (continuous output) | ReLU | Linear |

- If convergence (gradient descent with back propagation) is not working, change ReLU to **ELU**, **leaky ReLU** or **PReLU**.
- Regression uses the regression loss functions: see [Loss functions](#/t/dlloss).

## Quick reference

| Function | Output | Derivative | Main problem |
|---|---|---|---|
| Sigmoid | 0 → 1 | 0 → 0.25 | vanishing gradient, slow, not 0 centred |
| Tanh | −1 → 1 | 0 → 1 | still poor in very deep networks |
| ReLU | max(0, x) | 0 or 1 | dead neurons, not 0 centred |
| Leaky ReLU | max(0.01x, x) | 0.01 or 1 | solves dead neurons |
| Softmax | probabilities summing to 1 | | multiclass output |
