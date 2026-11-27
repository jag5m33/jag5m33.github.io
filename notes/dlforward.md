# Forward propagation

How the inputs travel forward through the network to make a prediction, and how we tell if it's wrong.

**On this page**

- Inside a neuron: weights, bias, activation
- Sigmoid for binary classification
- The output neuron and the loss
- The whole process

## Inside a neuron

The **weights go into the first NN layer.** Each hidden neuron does two things:

**1) Sum the inputs × weights, plus the bias**

$$\sum x_i w_i = x_1 w_1 + x_2 w_2 + x_3 w_3 + b$$

- $b$ = the **bias**.
- This goes into **each neuron** in the hidden layer = it **creates a linear equation** in each, with its own weights.

**2) Put the sum through the activation function**

$$\sum x_i w_i = z \quad\text{then}\quad a(z) = \hat y$$

![Each hidden neuron is split in two: Σxᵢwᵢ + b, then the activation act( ).](figures/dl-network.svg)

## Sigmoid for binary classification

Activation function for **binary classification: sigmoid**

$$\sigma = \frac{1}{1 + e^{-(\sum x_i w_i + b)}}$$

- Output = between **0 and 1**, with a **threshold** for classification (≥ 0.5 → 1, < 0.5 → 0).

### Perceptron / NN order: forward propagation

1. Input
2. Weights
3. Formula = $\sum x_i w_i$
4. Add bias
5. Activation function
6. Output

## The output neuron and the loss

The output neuron gives $\hat y$ = **predicted**; $y$ = **true**.

- $y - \hat y$ close to **0** = good for prediction accuracy.

> **Definition:** the **loss function** measures how far the prediction is from the truth.

Example: we predict 0 and the true value is 1:

$$y - \hat y = 1 - 0 = 1$$

**Large error**: not close to 0.

> **Say it like this:** "If the loss is high, **back propagate**."

## Back propagation and the optimiser

- **Back propagation:** update the weights → decrease the loss → better prediction.
- **Optimiser:** makes sure all the weights are updated correctly for back propagation.
  - Example of an optimiser: **gradient descent**, changing the weights / coefficients.
- Training is a combination of **forward** propagation (multiply weights and predict) and **back** propagation (update weights).

More in [Back propagation & gradient descent](#/t/dlbackprop).

## The whole process

| Step | | |
|---|---|---|
| 1 | Input layer | **forward propagation** |
| 2 | Weights | |
| 3 | Bias | |
| 4 | Activation function | |
| 5 | Loss function ($y - \hat y$ ↓) | **back propagation** |
| 6 | Optimiser | |
| 7 | Update weights | |

## Quick reference

| Term | In one line |
|---|---|
| Weights | how much each input counts |
| Bias $b$ | added to the sum |
| $z = \sum x_i w_i + b$ | a linear equation in each neuron |
| Activation $a(z) = \hat y$ | turns the sum into the neuron's output |
| Sigmoid | output 0 → 1, threshold for binary classification |
| Loss | how wrong the prediction is: high → back propagate |
