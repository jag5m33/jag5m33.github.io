# Logs & exponents

A gentle start from zero: what a power means, why a logarithm is just the reverse question, and why data scientists use logs on skewed data.

## Exponents: what a power means

An **exponent** (or **power**) says how many times to multiply a number by itself.

$$
2^3 = 2 \times 2 \times 2 = 8
$$

- $2$ is the **base**: the number being multiplied.
- $3$ is the **exponent**: how many copies.

Read $2^3$ as "2 to the power 3". $5^2 = 25$ is "5 squared".

### Three handy facts

| Fact | Example | Why |
|---|---|---|
| Multiplying powers of the same base → **add** the exponents | $2^2 \times 2^3 = 2^5 = 32$ | two 2s and three 2s make five 2s |
| Anything to the power 0 is 1 | $2^0 = 1$ | $2^3 \div 2^3 = 1$ |
| A negative power means "one over" | $2^{-2} = \frac{1}{4} = 0.25$ | it's a small positive number, not a negative one |

### The number e

$e \approx 2.718$ is a special constant, like $\pi$. It comes from growth that compounds continuously (like interest paid every instant). You'll see it written as $e^x$ or `np.exp(x)`.

![Left: growth curves with bases 2, e and 3 all pass through (0, 1) and then shoot up; the e curve sits between the other two. Right: decay, halving at each step.](figures/logs-growth-decay.svg)

In the figure, every growth curve starts flat and then explodes, because each step **multiplies** rather than adds. That multiplying is what logs undo.

## Logarithms: the reverse question

An exponent asks: "2 to the power 3 is what?" (answer 8).

A **logarithm** asks the reverse: "2 to the **what** is 8?" (answer 3).

$$
\log_2 8 = 3 \quad \text{because} \quad 2^3 = 8
$$

| Question | Because | Answer |
|---|---|---|
| $\log_{10} 1000$ | $10^3 = 1000$ | 3 |
| $\log_{10} 0.01$ | $10^{-2} = 0.01$ | −2 |
| $\log_2 1$ | $2^0 = 1$ | 0 |

For base 10, the log is roughly the number of digits: $\log_{10}$ of a million is 6.

### Which base?

| Name | Base | numpy | Where you'll see it |
|---|---|---|---|
| Natural log, $\ln$ | $e$ | `np.log` | log loss, log transforms, sklearn |
| Base 10 | 10 | `np.log10` | log-scale axes |
| Base 2 | 2 | `np.log2` | entropy |

> **Trap:** in Python, plain `log` means the **natural log** (base $e$). `np.log(100)` is 4.605, not 2.

### Two values worth remembering

- $\ln 1 = 0$, because $e^0 = 1$.
- For numbers between 0 and 1, the log is **negative**: $\ln 0.5 \approx -0.693$. And $\ln 0$ doesn't exist.

Probabilities live between 0 and 1, so **the log of a probability is always zero or negative**. Keep that in mind for later.

![The exponential curve (wine) and the natural log curve (teal) reflected in the dotted line y = x. The point (0, 1) on one becomes (1, 0) on the other. The pink strip marks x between 0 and 1, where ln x is negative.](figures/logs-mirror.svg)

In the figure, the two curves are mirror images: $e^x$ shoots up, $\ln x$ climbs ever more slowly. The pink strip is where logs are negative.

## Log transforms in data

A [right-skewed](#/t/descriptive) column (prices, salaries, durations) has mostly small values and a long tail of big ones. The log squashes big numbers much more than small ones: going from 10 to 100 and from 100 to 1000 are both just +1 on the base-10 scale.

![Left: 1,500 simulated house prices, right-skewed (skew 1.88), mean pulled above median. Right: the same data after ln, almost symmetric (skew −0.08), mean and median together.](figures/logs-transform.svg)

In the figure, the long right tail on the left is folded in on the right, and the mean and median come back together.

**Zeros?** $\ln 0$ doesn't exist, and lots of real columns have zeros (rainfall, word counts). Use `np.log1p`, which takes the log of $(1 + x)$, so 0 stays 0.

**When not to log:** negative values (undefined); data that aren't skewed; tree models like [decision trees](#/t/trees), which only care about order, so a log changes nothing.

> **Trap:** if you log the target, your model predicts the log. Convert predictions back with `np.expm1` (or `np.exp`) before comparing to real values.

```python
import numpy as np

print(2 ** 3)            # 8      Python uses ** for powers
print(np.exp(1))         # 2.718  the number e
print(np.log10(1000))    # 3.0
print(np.log(1))         # 0.0
print(np.log(0.5))       # -0.693 log of a probability is negative
print(np.log1p([0, 9]))  # [0, 2.303]  safe with zeros

# Log-transform a skewed pandas column:
# df["price_log"] = np.log1p(df["price"])
```

## Where you'll meet logs next

- **Sigmoid** ([logistic regression](#/t/logreg)): $1/(1 + e^{-z})$ uses $e$ to squash any number into a probability between 0 and 1.
- **Log loss** ([logistic regression](#/t/logreg)): the cost is $-\ln(p)$ for the right answer; in Jag's words, "confident and wrong = big cost".
- **TF-IDF** ([TF-IDF](#/t/tfidf)): a log shrinks the weight of words that appear in nearly every document towards 0.
- **Entropy** ([decision trees](#/t/trees)): uses $\log_2$ of class shares to measure how mixed a node is, in bits.

## Quick reference

| Term | Rule | Example |
|---|---|---|
| Exponent | base multiplied by itself $n$ times | $2^3 = 8$ |
| Zero / negative power | $b^0 = 1$; negative = one over | $2^{-2} = 0.25$ |
| $e$ | constant ≈ 2.718 | `np.exp(1)` |
| Logarithm | "base to the what gives this?" | $\log_2 8 = 3$ |
| Natural log | base $e$; `np.log` in Python | $\ln 1 = 0$ |
| Logs between 0 and 1 | negative | $\ln 0.5 = -0.693$ |
| log1p | log of $(1 + x)$, safe at 0 | log1p(0) = 0 |
| Log transform | pulls in right skew | prices, salaries |
