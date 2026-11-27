# Back propagation & gradient descent

How the network learns: go backwards from the output and update every weight to lower the loss.

**On this page**

- Forward and back
- The weight update formula
- Which way to move: the slope
- The chain rule

Day 2 of deep learning covered: the **chain rule of derivatives**, the **vanishing gradient problem** and **loss functions**.

## Forward and back

```
          forward →
─O─ w₁ ╲
─O─ w₂ ─ O ─ w₄ ─ O ── O/P ŷ
─O─ w₃ ╱
          ← back
```

Back propagation uses:

- the **weight update formula**
- the **chain rule** of differentiation

## The weight update formula

$$w_{new} = w_{old} - \alpha \frac{dL}{dw_{old}}$$

- $\alpha$ = the learning rate
- $\dfrac{dL}{dw_{old}}$ = the slope of the **loss** against the **old weight**

![Loss against weight is a U-shaped curve. Gradient descent walks downhill to the bottom: the global minimum.](figures/dl-gradient-descent.svg)

## Which way to move: the slope

**If the slope is negative**, the weight must **↑** to get to the minimum:

$$w_{new} = w_{old} - \alpha(-ve) = w_{old} + \alpha \frac{dL}{dw_{old}}$$

$-\alpha \times -ve$ = add, so the weight moves towards the minimum.

**If the slope is positive**, the weight must **↓** to get to the minimum:

$$w_{new} = w_{old} - \alpha(+ve)$$

subtract, so the weight moves down to the minimum.

> **Check:** in my notes the negative-slope case says "add so moves down weights". Adding makes the weight go **up** (to the right on the graph), which is the way to the minimum when the slope is negative.

## The chain rule

$$w_{4\,new} = w_{4\,old} - \alpha \frac{dL}{dw_{4\,old}}$$

To get $\dfrac{dL}{dw_4}$, define it through the output $O_1$:

$$\frac{dL}{dO_1} \times \frac{dO_1}{dw_4} = \frac{dL}{dw_4}$$

- $\dfrac{dL}{dO_1}$: how the loss changes with the **output**
- $\dfrac{dO_1}{dw_4}$: how the output changes with the **weight**

→ **Update incrementally through the layers**, from the output back to the start.

## Quick reference

| Term | In one line |
|---|---|
| Back propagation | update the weights backwards from the output |
| $w_{new} = w_{old} - \alpha \frac{dL}{dw}$ | the weight update formula |
| Slope −ve | weight ↑ |
| Slope +ve | weight ↓ |
| Chain rule | $\frac{dL}{dw} = \frac{dL}{dO} \times \frac{dO}{dw}$ |
| Global minimum | the bottom of the loss curve |
