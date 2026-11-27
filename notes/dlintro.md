# Deep learning & neural networks

Where deep learning sits, and what a neural network looks like.

**On this page**

- AI, ML and deep learning
- The perceptron
- Training on a dataset
- Multilayered neural networks

## AI, ML and deep learning

| | What it is |
|---|---|
| **AI** | its own tools that can do their own tasks without human intervention, e.g. Netflix, self-driving cars |
| **ML** | a subset of AI: stats tools to analyse and visualise data |
| **Deep learning** | a subset of ML: focuses on **multi-layer neural networks** (NN) |

> **Definition:** the **aim of deep learning** is to mimic the human brain: we want it to learn like humans.

## The perceptron

> **Definition:** a **perceptron** is a **single layer** neural network.

```
→ O ─┐
→ O ─┼─ O ─── O ── O/P
→ O ─┘
input   hidden    output
layer   layer 1   layer
```

## Training on a dataset

| Study | Play | Sleep | Pass / Fail |
|---|---|---|---|
| 7 | 3 | 7 | 1 |
| 2 | 5 | 8 | 0 |
| 4 | 3 | 7 | P |

How is this dataset used to train a binary classifier?

![Study (x₁), play (x₂) and sleep (x₃) go in, through hidden layer 1, to one output neuron: pass or fail.](figures/dl-network.svg)

- **Features = inputs.** Number of features = number of inputs.
- Train the NN based on the **output, which is known**.

## Multilayered neural networks

> **Definition:** **multilayered** = many hidden layers.

![A multilayered network: an input layer, several hidden layers, and an output layer, every neuron joined to every neuron in the next layer.](figures/dl-deep.svg)

## Quick reference

| Term | In one line |
|---|---|
| Deep learning | multi-layer neural networks; aims to learn like humans |
| Perceptron | single layer NN |
| Features | the inputs: one input per feature |
| Multilayered NN | many hidden layers |
