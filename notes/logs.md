# Logs & exponents

A log answers one question: **what power do I need?** Data scientists use it to tame skewed data and inside log loss.

**On this page**

- Exponents: what a power means
- Logs: the reverse question
- Which base, and Python's default
- Logs of numbers below 1 are negative
- Log transforms on skewed data
- Where you'll meet logs next

## Exponents

> **Definition:** an **exponent** (or **power**) says how many times to multiply a number by itself. 2³ = 2 × 2 × 2 = 8.

- **2** is the **base**: the number being multiplied.
- **3** is the **exponent**: how many copies.
- Read 2³ as "2 to the power 3". 5² = 25 is "5 squared".
- Anything to the power 0 is 1: 2⁰ = 1.

**e ≈ 2.718** is a special constant, like π. It comes from growth that compounds all the time. You'll see it as `np.exp(x)`.

![Growth with bases 2, e and 3: flat at first, then exploding, because each step multiplies. Right: halving each step.](figures/logs-growth-decay.svg)

That multiplying is what logs undo.

## Logs: the reverse question

> **Definition:** a **logarithm** asks the reverse of a power: "the base to the **what** gives this number?" log₂ 8 = 3, because 2³ = 8.

- Power: "2 to the power 3 is what?" → 8.
- Log: "2 to the what is 8?" → 3.

**Worked example.** log₁₀ 1000 = 3, because 10³ = 1000. log₁₀ 1,000,000 = 6. For base 10, the log **counts the zeros**.

## Which base?

> **Definition:** the **natural log**, written **ln**, is the log with base e. It's the one statistics and machine learning use by default.

- `np.log`: natural log (base e). Log loss, log transforms, sklearn.
- `np.log10`: base 10. Log-scale chart axes.
- `np.log2`: base 2. Entropy in decision trees.

> **Where it goes wrong:** in Python, plain `log` means the **natural log**. `np.log(100)` is 4.605, not 2. For base 10, use `np.log10`.

## Logs below 1 are negative

> **Definition:** **ln 1 = 0**, because e⁰ = 1. Any number **between 0 and 1** has a **negative** log, and **ln 0 doesn't exist**.

Example: ln 0.5 ≈ −0.693.

Probabilities live between 0 and 1, so **the log of a probability is always zero or negative**. That's why log loss puts a minus sign in front.

![The e-curve and the ln-curve are mirror images across the dotted line y = x. The pink strip is x between 0 and 1: there, ln x is below zero.](figures/logs-mirror.svg)

These are the three facts to keep: **a log asks "what power?"**, **`np.log` is base e**, and **logs of 0 to 1 are negative (log 0 doesn't exist)**.

## Log transforms on skewed data

> **Definition:** a **log transform** replaces every value with its log, to pull in a long right tail.

A [right-skewed](#/t/descriptive) column (prices, salaries, durations) is mostly small values with a few huge ones. The log squashes big numbers far more than small ones: 10 → 100 and 100 → 1000 are both just +1 on the base-10 scale.

**Worked example.** House prices of £25,000 and £250,000 are £225,000 apart. Their base-10 logs, 4.4 and 5.4, are just 1 apart.

![Left: 1,500 house prices, long right tail, mean pulled above the median. Right: the same prices after a log, nearly symmetric, with mean and median back together.](figures/logs-transform.svg)

**Zeros?** Real columns often have them (rainfall, word counts). Use `np.log1p`, the log of (1 + x), so 0 stays 0.

```python
import numpy as np

np.log(1)        # 0.0     ln 1 = 0
np.log(0.5)      # -0.693  log of a probability is negative
np.log10(1000)   # 3.0     "10 to the what is 1000?"
df["price_log"] = np.log1p(df["price"])   # safe with zeros
```

> **Where it goes wrong:** if you log the target, the model predicts the **log**. Convert back with `np.expm1` before comparing to real prices. And don't bother logging for tree models like [decision trees](#/t/trees): they only care about order, so nothing changes.

> **Say it like this:** "Price was heavily right-skewed, so I log-transformed it with log1p. That pulled in the tail so a few very expensive houses didn't dominate the fit."

## Where you'll meet logs next

> **Definition:** in ML, logs mostly **squash** big numbers or **punish** confident mistakes.

- **Sigmoid** ([logistic regression](#/t/logreg)): uses e to squash any number into a probability between 0 and 1.
- **Log loss** ([logistic regression](#/t/logreg)): the cost is minus the log of the probability you gave the right answer. In Jag's words, "confident and wrong = big cost".
- **TF-IDF** ([TF-IDF](#/t/tfidf)): a log shrinks the weight of words found in nearly every document.
- **Entropy** ([decision trees](#/t/trees)): uses base-2 logs to measure how mixed a node is.

## Quick reference

| Term | In one line |
|---|---|
| Exponent | Multiply the base by itself that many times: 2³ = 8 |
| e | Constant ≈ 2.718, `np.exp(1)` |
| Log | "Base to the what gives this?": log₂ 8 = 3 |
| Natural log (ln) | Base e; plain `np.log` in Python |
| Logs of 0 to 1 | Negative; ln 1 = 0; ln 0 doesn't exist |
| log1p | Log of (1 + x), safe with zeros |
| Log transform | Pulls in right skew: prices, salaries |
