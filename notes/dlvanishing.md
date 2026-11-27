# The vanishing gradient problem

Why weights near the start of a deep network can stop learning.

**On this page**

- The setup
- The sigmoid's derivative is small
- Multiplying small numbers
- The fix

## The setup

```
x₁ → O ─ w₁ ─ O ─ w₂ ─ O ─ w₃ ─ O ─ w₄ ─ O → O/P
```

- Loss function: $\tfrac{1}{2}(y - \hat y)^2$ (squared error)
- **Sigmoid** activation function in every neuron
- Update the first weight:

$$w_{1\,new} = w_{1\,old} - \alpha \frac{dL}{dw_{1\,old}}$$

To update $w_1$, the chain rule goes all the way back from the output:

$$\frac{dL}{dw_1} = \frac{dL}{dO_5} \times \frac{dO_5}{dO_4} \times \frac{dO_4}{dO_3} \times \frac{dO_3}{dO_2} \times \frac{dO_2}{dO_1} \times \frac{dO_1}{dw_1}$$

Back propagate from output 5 → 1.

> **Check:** my notes end the chain at $\frac{dO_2}{dO_1}$. It needs one more term, $\frac{dO_1}{dw_1}$, to reach the weight itself.

## The sigmoid's derivative is small

![Sigmoid goes from 0 to 1 and crosses 0.5 at 0. Its derivative is a hump with a peak of only 0.25.](figures/dl-activations.svg)

$$z = \frac{1}{1 + e^{-z}} \to 0 \text{ or } 1 \qquad \ge 0.5 \Rightarrow 1, \quad < 0.5 \Rightarrow 0$$

- The input to the sigmoid is $y = w^T x + b$.
- The derivative of the sigmoid activation function is **always between 0 → 0.25**.

$O_5$, for example, is the output at the 5th neuron layer:

$$O_5 = \sigma\big((O_4 \times w_5) + b\big)$$

$\sigma$ = the activation function, and the inside is the usual $\sum x_i w_i + b$. Its derivative: **0 → 0.25**.

## Multiplying small numbers

Since the derivative range is small (0 → 0.25):

- each term in the chain is between 0 and 0.25
- then it gets smaller and smaller, as 0 → 0.25 is multiplied into the next derivative

$$w_{new} = w_{old} - \underbrace{\alpha}_{\text{small}}\,(\underbrace{\text{small number}}_{\text{tiny}})$$

So $w_{new} \approx w_{old}$, as the change is hardly bigger than 0 = **no change**.

> **Definition:** the **vanishing gradient problem**: the weights aren't updated (no change), because the gradient has shrunk to almost nothing by the time it reaches the early layers.

## The fix

**Use another activation function:**

1. Sigmoid
2. Tanh
3. ReLU
4. Leaky ReLU
5. Softmax

See [Activation functions](#/t/dlactivations).

## Quick reference

| Term | In one line |
|---|---|
| Sigmoid derivative | always 0 → 0.25 |
| Chain rule in a deep NN | many small numbers multiplied together |
| Vanishing gradient | $w_{new} \approx w_{old}$: early weights stop learning |
| Fix | use another activation function |
