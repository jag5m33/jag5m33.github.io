# Logs & exponents

Built from zero: what a power means, where the number e comes from, why a logarithm is just the reverse question, and where both turn up in data science.

## Exponents: what a power means

An **exponent** (or **power**) says how many times to multiply a number by itself.

$$
2^3 = 2 \times 2 \times 2 = 8
$$

- $2$ is the **base**: the number being multiplied.
- $3$ is the **exponent**: how many copies.
- $8$ is the result.

Read $2^3$ as "2 to the power 3" or "2 cubed". $5^2 = 25$ is "5 squared".

## Exponent rules, worked

Each rule is just counting copies. Let's see why, with small numbers.

### Product rule: same base, multiply → add the exponents

$$
b^m \times b^n = b^{m+n}
$$

Why: $2^2 \times 2^3 = (2 \times 2) \times (2 \times 2 \times 2)$, which is five 2s. So $2^5 = 32$. Check: $4 \times 8 = 32$.

### Quotient rule: same base, divide → subtract the exponents

$$
\frac{b^m}{b^n} = b^{m-n}
$$

Why: $\frac{2^5}{2^2} = \frac{2 \times 2 \times 2 \times 2 \times 2}{2 \times 2}$. Two pairs cancel, three 2s are left: $2^3 = 8$. Check: $32 / 4 = 8$.

### Power rule: power of a power → multiply the exponents

$$
(b^m)^n = b^{m \times n}
$$

Why: $(2^3)^2 = 2^3 \times 2^3 = 2^6 = 64$. Check: $8^2 = 64$.

### Zero exponent: anything to the power 0 is 1

$$
b^0 = 1 \quad (b \ne 0)
$$

Why: use the quotient rule on something divided by itself. $\frac{2^3}{2^3} = 2^{3-3} = 2^0$, but anything divided by itself is 1. So $2^0 = 1$.

### Negative exponent: means "one over"

$$
b^{-n} = \frac{1}{b^n}
$$

Why: $\frac{2^1}{2^3} = 2^{1-3} = 2^{-2}$, and also $\frac{2}{8} = \frac{1}{4} = \frac{1}{2^2}$. So $2^{-2} = 0.25$.

> **Key idea:** a negative exponent does **not** make the answer negative. $2^{-2} = 0.25$, a small positive number.

### Fractional exponent: means a root

$$
b^{1/n} = \sqrt[n]{b} \qquad b^{m/n} = \left(\sqrt[n]{b}\right)^m
$$

Why: by the power rule, $(9^{1/2})^2 = 9^{1} = 9$. The number that squares to 9 is 3, so $9^{1/2} = \sqrt{9} = 3$. Likewise $8^{1/3} = 2$ (because $2^3 = 8$) and $8^{2/3} = 2^2 = 4$.

### Summary of the rules

| Rule | Formula | Example |
|---|---|---|
| Product | $b^m b^n = b^{m+n}$ | $2^2 \cdot 2^3 = 2^5 = 32$ |
| Quotient | $b^m / b^n = b^{m-n}$ | $2^5 / 2^2 = 2^3 = 8$ |
| Power | $(b^m)^n = b^{mn}$ | $(2^3)^2 = 2^6 = 64$ |
| Zero | $b^0 = 1$ | $7^0 = 1$ |
| Negative | $b^{-n} = 1/b^n$ | $2^{-2} = 0.25$ |
| Fractional | $b^{1/n} = \sqrt[n]{b}$ | $9^{1/2} = 3$ |

> **Trap:** $(2 + 3)^2 = 25$, not $2^2 + 3^2 = 13$. Powers don't split over addition.

## The number e

$e \approx 2.71828$ is a constant like $\pi$. It comes from compound interest.

Put £1 in a bank paying 100% interest a year.

- Paid once a year: $(1 + 1)^1 = £2.00$
- Paid twice (50% each half): $(1 + \tfrac{1}{2})^2 = £2.25$
- Quarterly: $(1 + \tfrac{1}{4})^4 \approx £2.441$
- Monthly: $(1 + \tfrac{1}{12})^{12} \approx £2.613$
- Daily: $(1 + \tfrac{1}{365})^{365} \approx £2.715$

Paying more often helps, but less and less. As the number of payments $n$ goes to infinity it settles on a limit:

$$
e = \lim_{n \to \infty}\left(1 + \frac{1}{n}\right)^n \approx 2.71828
$$

So $e$ is what continuous growth at 100% gives you in one unit of time. Its special property is that the slope of $e^x$ at any point equals its own value: it grows at a rate equal to its current size. That makes calculus with $e^x$ very clean, which is why it appears in the normal distribution, the sigmoid and softmax. $e^x$ is also written $\exp(x)$.

### Exponential growth and decay

$$
y = a \cdot b^t
$$

- $a$: starting amount. $b$: growth factor per step. $t$: number of steps.
- $b > 1$: **growth**. £1,000 at 5% a year for 10 years: $1000 \times 1.05^{10} \approx £1{,}629$.
- $0 < b < 1$: **decay**. Halving each step: $100 \times 0.5^t$ gives $100, 50, 25, 12.5, \dots$

![Left: growth curves with bases 2, e and 3 all pass through (0, 1) and then shoot up; the e curve sits between the other two. Right: decay, halving at each step.](figures/logs-growth-decay.svg)

In the figure, notice every growth curve starts flat and then explodes: each step multiplies, rather than adds. That multiplying is what logs undo.

## Logarithms: the reverse question

An exponent asks: "2 to the power 3 is what?" (answer 8).

A **logarithm** asks the reverse: "2 to the **what** is 8?" (answer 3).

$$
\log_b(y) = x \quad \Longleftrightarrow \quad b^x = y
$$

Say it: "log base 2 of 8 is 3, because $2^3 = 8$."

**Worked:**

| Question | Because | Answer |
|---|---|---|
| $\log_2 8$ | $2^3 = 8$ | 3 |
| $\log_{10} 1000$ | $10^3 = 1000$ | 3 |
| $\log_{10} 0.01$ | $10^{-2} = 0.01$ | −2 |
| $\log_2 1$ | $2^0 = 1$ | 0 |
| $\log_{10} 20$ | between $10^1$ and $10^2$ | about 1.301 |

A log is roughly "how many times do I multiply by the base to get here?" For base 10 that's about the number of digits: $\log_{10}$ of a million is 6.

### Three bases you'll see

| Name | Base | Written | numpy | Where |
|---|---|---|---|---|
| Natural log | $e$ | $\ln x$ (or $\log x$ in most maths and code) | `np.log` | log loss, log-odds, log transforms, sklearn |
| Common log | 10 | $\log_{10} x$ | `np.log10` | orders of magnitude, log-scale axes |
| Binary log | 2 | $\log_2 x$ | `np.log2` | entropy, bits |

> **Trap:** in Python and most ML papers, plain `log` means **natural log** (base $e$). `np.log(100)` is 4.605, not 2.

### Log rules: the exponent rules in reverse

Because logs undo exponents, every exponent rule has a mirror. Multiplying inside becomes adding outside.

| Exponent rule | Log rule | Example |
|---|---|---|
| $b^m b^n = b^{m+n}$ | $\log(xy) = \log x + \log y$ | $\log_{10}(100 \times 1000) = 2 + 3 = 5$ |
| $b^m / b^n = b^{m-n}$ | $\log(x/y) = \log x - \log y$ | $\log_2(32/4) = 5 - 2 = 3$ |
| $(b^m)^n = b^{mn}$ | $\log(x^n) = n \log x$ | $\log_2(8^2) = 2 \times 3 = 6$ |
| $b^0 = 1$ | $\log 1 = 0$ | any base |
| $b^1 = b$ | $\log_b b = 1$ | $\ln e = 1$ |

**Change of base.** Your calculator only has $\ln$ and $\log_{10}$? Use:

$$
\log_b x = \frac{\ln x}{\ln b}
$$

Worked: $\log_2 8 = \ln 8 / \ln 2 = 2.0794 / 0.6931 = 3$. All log bases are the same curve stretched by a constant, which is why the base often doesn't matter for ML (it only rescales).

### log(1) = 0, and logs of numbers between 0 and 1 are negative

This matters for log loss, so slowly:

- $\ln 1 = 0$ because $e^0 = 1$.
- For $0 < x < 1$: what power of $e$ gives a number less than 1? A **negative** one (negative exponents give fractions). So $\ln 0.5 \approx -0.693$, $\ln 0.1 \approx -2.303$, $\ln 0.01 \approx -4.605$.
- As $x \to 0$, $\ln x \to -\infty$. You'd need an ever more negative power.
- $\ln 0$ and $\ln(\text{negative})$ are undefined. No power of $e$ gives 0 or less.

Probabilities live between 0 and 1, so **the log of a probability is always zero or negative**. That's why log loss has a minus sign in front.

### e^x and ln(x) are mirror images

![The exponential curve (wine) and the natural log curve (teal) reflected in the dotted line y = x. The point (0, 1) on one becomes (1, 0) on the other. The pink strip marks x between 0 and 1, where ln x is negative.](figures/logs-mirror.svg)

In the figure, swapping $x$ and $y$ turns one curve into the other: that's what "inverse function" means. $e^x$ explodes upwards; $\ln x$ climbs ever more slowly. It takes $x \approx 22{,}026$ to get $\ln x = 10$.

## Log transforms in data

### Why they pull in right skew

A [right-skewed](#/t/descriptive) column (prices, salaries, durations, counts) has most values small and a long tail of big ones. The log squashes big numbers far more than small ones: going from 10 to 100 and from 100 to 1000 are both just +1 on the $\log_{10}$ scale.

![Left: 1,500 simulated house prices, right-skewed (skew 1.88), mean pulled above median. Right: the same data after ln, almost symmetric (skew −0.08), mean and median together.](figures/logs-transform.svg)

In the figure, the long right tail on the left panel is folded in on the right panel. Data that become normal after a log are called **log-normal**.

### Multiplicative becomes additive

By the product rule, $\log(a \times b) = \log a + \log b$. So:

- A variable that grows by **percentages** (prices up 5%, then 10%) becomes one that moves by **steps** on the log scale. Linear models are built for steps.
- In a regression on $\ln y$, a coefficient of $0.05$ means roughly "+5% in $y$" per unit of $x$.

### log1p for zeros

$\ln 0$ is undefined, and plenty of real columns have zeros (precipitation, word counts). Use:

$$
\text{log1p}(x) = \ln(1 + x)
$$

$\text{log1p}(0) = \ln 1 = 0$, so zeros stay zero. $\text{log1p}(9) = \ln 10 \approx 2.303$. Its inverse is `np.expm1`.

### When not to log

- **Negative values**: log is undefined. Consider shifting (with care) or a Yeo–Johnson transform.
- **Data that aren't skewed**: you'd introduce a left skew.
- **Tree models** ([decision trees](#/t/trees), XGBoost): they split on order, which a log doesn't change, so it rarely helps.
- **When you must report in original units**: remember to transform back with `np.exp` (and note that the mean of logs, transformed back, is the geometric mean, closer to the median than the mean).

> **Trap:** if you log the target $y$, your model predicts $\ln y$. Errors like RMSE are then in log units; convert predictions back before comparing to the real values.

## Where you'll meet them next

![Four places logs and exponents appear: the sigmoid (top left), log loss (top right), log-odds (bottom left) and sublinear TF (bottom right).](figures/logs-next.svg)

### The sigmoid, in [logistic regression](#/t/logreg)

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It squashes any number $z$ into $(0, 1)$ so it can be read as a probability. In the top-left panel: $z = 0$ gives $1/(1+1) = 0.5$. $z = 2$ gives $1/(1 + e^{-2}) \approx 0.881$. Large positive $z$: $e^{-z}$ is tiny so $\sigma \to 1$. Large negative $z$: $e^{-z}$ is huge so $\sigma \to 0$.

### Log loss, also in [logistic regression](#/t/logreg)

The cost for one example is $-\ln(p)$, where $p$ is the probability the model gave to the **correct** answer. In Jag's NLP notes: "Cost = −log(probability given to the right answer). Confident and wrong = big cost."

| $p$ for the right answer | $-\ln p$ |
|---|---|
| 0.9 (confident, right) | 0.11 |
| 0.5 (unsure) | 0.69 |
| 0.01 (confident, wrong) | 4.61 |

In the top-right panel, the curve blows up towards infinity as $p \to 0$. That's the "logs of numbers between 0 and 1 are negative" fact, flipped by the minus sign. A confidently wrong prediction costs about 40 times more than a confident right one, which pushes the model away from overconfidence.

### Log-odds

**Odds** are $p / (1 - p)$: $p = 0.8$ gives odds of 4 (4 to 1). The **log-odds** (or **logit**) is

$$
\text{logit}(p) = \ln\frac{p}{1-p}
$$

$p = 0.8$: $\ln 4 \approx 1.386$. It's the inverse of the sigmoid: logistic regression is a linear model for the log-odds, $\ln\frac{p}{1-p} = w \cdot x + b$. Bottom-left panel: $p = 0.5$ gives 0; probabilities near 0 or 1 stretch out to $\pm\infty$.

### TF-IDF, in [TF-IDF](#/t/tfidf)

Two logs:

- **Sublinear TF**: $1 + \ln(\text{count})$. A word appearing 20 times gets weight $1 + \ln 20 \approx 4.0$, not 20 (bottom-right panel). From Jag's `tf_idf.py`: the $+1$ is there because $\log(1) = 0$, so without it a word appearing once would get weight 0 and vanish. sklearn turns this on with `sublinear_tf=True` and uses the natural log.
- **IDF**: $\ln(N / \text{df})$, where $N$ is the number of documents and df the number containing the word. In 1,000 documents, a word in 10 of them: $\ln 100 \approx 4.6$. A word in all 1,000: $\ln 1 = 0$, so it carries no information. (sklearn's default adds smoothing: $\ln\frac{1+N}{1+\text{df}} + 1$.)

([`tf_idf.py`](https://github.com/jag5m33/py_legal_data/blob/main/src/tf_idf.py))

### Entropy, in [decision trees](#/t/trees)

$$
H = -\sum_k p_k \log_2 p_k
$$

$\log_2$ gives the answer in **bits**. A 50/50 node has $H = 1$ bit (maximally mixed); a 90/10 node has $H \approx 0.47$; a pure node has $H = 0$. The minus sign is there for the same reason as in log loss: logs of probabilities are negative.

![Entropy of a two-class node against the share of class 1. It peaks at 1 bit for a 50/50 split and drops to 0 for a pure node.](figures/logs-entropy.svg)

In the figure, a tree picks the split that moves its nodes furthest down this curve.

## Code

```python
import numpy as np

# Exponents
print(2 ** 3)                 # 8      Python uses ** for powers (not ^)
print(2 ** -2)                # 0.25   negative exponent = one over
print(9 ** 0.5)               # 3.0    fractional exponent = root
print(np.exp(1))              # 2.718... the number e
print(np.exp([0, 1, 2]))      # [1, 2.718, 7.389]  works on whole arrays

# Logs: natural, base 10, base 2
print(np.log(np.e))           # 1.0    np.log is the NATURAL log
print(np.log10(1000))         # 3.0
print(np.log2(8))             # 3.0
print(np.log(8) / np.log(2))  # 3.0    change of base by hand
print(np.log(0.5))            # -0.693 log of a probability is negative
print(np.log(1))              # 0.0

# log1p / expm1: safe with zeros
x = np.array([0, 9, 99])
print(np.log1p(x))            # [0, 2.303, 4.605]  ln(1 + x)
print(np.expm1(np.log1p(x)))  # [0, 9, 99]          back to the original

# Log-transform a skewed pandas column (zeros allowed)
# df["price_log"] = np.log1p(df["price"])

# The ML functions from this chapter
z = np.array([-2, 0, 2])
print(1 / (1 + np.exp(-z)))   # sigmoid: [0.119, 0.5, 0.881]
print(-np.log([0.9, 0.5, 0.01]))  # log loss: [0.105, 0.693, 4.605]
```

> **Trap:** `np.log(0)` returns `-inf` with a warning, and `np.log(-1)` returns `nan`. If either shows up, you've logged a zero or negative value. Use `log1p`, or clip probabilities away from 0 (sklearn's `log_loss` does this for you).

## Quick reference

| Term | Formula / rule | Example |
|---|---|---|
| Exponent | $b^n$ = $b$ multiplied $n$ times | $2^3 = 8$ |
| Product / quotient | $b^m b^n = b^{m+n}$; $b^m/b^n = b^{m-n}$ | $2^2 \cdot 2^3 = 32$ |
| Power | $(b^m)^n = b^{mn}$ | $(2^3)^2 = 64$ |
| Zero / negative | $b^0 = 1$; $b^{-n} = 1/b^n$ | $2^{-2} = 0.25$ |
| Fractional | $b^{1/n} = \sqrt[n]{b}$ | $9^{1/2} = 3$ |
| $e$ | $\lim (1 + 1/n)^n \approx 2.718$ | continuous growth |
| Logarithm | $\log_b y = x \iff b^x = y$ | $\log_2 8 = 3$ |
| Log rules | $\log xy = \log x + \log y$; $\log x^n = n\log x$ | $\log_{10}10^5 = 5$ |
| Change of base | $\log_b x = \ln x / \ln b$ | $\ln 8/\ln 2 = 3$ |
| Key values | $\ln 1 = 0$; $\ln x < 0$ for $0 < x < 1$ | $\ln 0.5 = -0.693$ |
| log1p | $\ln(1 + x)$, safe at 0 | log1p(0) = 0 |
| Sigmoid | $1/(1 + e^{-z})$ | $\sigma(2) = 0.881$ |
| Log loss | $-\ln p$ for the true class | $p = 0.01$ → 4.61 |
| Log-odds | $\ln(p/(1-p))$ | $p = 0.8$ → 1.386 |
| TF-IDF logs | $1 + \ln(\text{count})$, $\ln(N/\text{df})$ | 20 → 4.0 |
| Entropy | $-\sum p \log_2 p$ | 50/50 → 1 bit |
