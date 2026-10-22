# Distributions & the CLT

What a probability distribution is, the four you'll meet most, and the Central Limit Theorem, which is why hypothesis tests and confidence intervals work.

## Random variables

A **random variable** is a number whose value comes from a random process. Written with a capital letter, $X$. A particular value it takes is lower-case, $x$.

- **Discrete**: countable values, usually whole numbers. Number of heads in 10 flips, number of accidents per day.
- **Continuous**: any value in a range. Height, temperature, time until a server fails.

A **distribution** is the full description of which values $X$ can take and how likely each one is.

## PMF, PDF and CDF

| Name | For | Gives you | Rule |
|---|---|---|---|
| **PMF** (probability mass function) | discrete | $P(X = k)$, the probability of exactly $k$ | all bars sum to 1 |
| **PDF** (probability density function) | continuous | a **density** $f(x)$. Probability is **area** under the curve | total area is 1 |
| **CDF** (cumulative distribution function) | both | $F(x) = P(X \le x)$, the probability of $x$ or less | rises from 0 to 1 |

![Left: a PMF, where each bar's height is a probability. Middle: a PDF, where only areas are probabilities (the gold area between 0 and 1 is 0.341). Right: the CDF, which adds everything up to a point.](figures/distributions-pmf-pdf-cdf.svg)

In the figure, look at the middle panel: for a continuous variable, $P(X = \text{exactly } 1) = 0$ because a single line has no area. You can only ask about ranges: $P(a \le X \le b) = F(b) - F(a)$.

> **Trap:** a PDF's height can be bigger than 1 (a very narrow curve is very tall). It's a density, not a probability.

The CDF is the one Z-tables give you: in Jag's words, "a table gives the area to the left of a single point. Never an interval."

## The normal distribution

The bell curve. Two parameters:

- $\mu$ (mu): the mean, where the centre sits.
- $\sigma$ (sigma): the standard deviation, how wide it is.

Written $X \sim N(\mu, \sigma^2)$. Symmetric, so mean = median = mode. Its PDF is

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$

You don't need to memorise this. Read it as: the further $x$ is from $\mu$ (measured in $\sigma$s), the faster the density falls off, and the $\frac{1}{\sigma\sqrt{2\pi}}$ just makes the total area equal 1. ($e$ is explained in [Logs & exponents](#/t/logs).)

### The 68–95–99.7 rule

For **any** normal distribution:

![Within 1σ of the mean: 68.3% of values. Within 2σ: 95.4%. Within 3σ: 99.7%.](figures/distributions-empirical-rule.svg)

In the figure, each band adds another $\sigma$ either side. Beyond $3\sigma$ there's only 0.3% left, which is why $\lvert z \rvert > 3$ is a common [outlier rule](#/t/quartiles).

### The standard normal and the z-score

The **standard normal** is $N(0, 1)$: mean 0, SD 1. Any normal value can be converted to it with a **z-score**:

$$
z = \frac{x - \mu}{\sigma}
$$

Where it comes from: subtract $\mu$ to move the centre to 0, divide by $\sigma$ to make the spread 1. The z-score is "how many standard deviations from the mean".

**Worked.** Let's say adult heights are $N(175, 7^2)$ cm. How unusual is 185 cm?

1. $z = (185 - 175) / 7 \approx 1.43$
2. Look up $\Phi(1.43) \approx 0.923$ (the CDF of the standard normal, written $\Phi$)
3. So about 92.3% of people are shorter, and $1 - 0.923 \approx 7.7\%$ are taller.

> **Say it like this:** "185 cm is 1.43 standard deviations above the mean, taller than about 92% of adults."

## Binomial distribution

**Use when:** you count successes in a fixed number of independent yes/no trials, each with the same success probability.

- $n$: number of trials
- $p$: probability of success on each trial

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad \binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

Where it comes from: one particular sequence with $k$ successes and $n-k$ failures has probability $p^k(1-p)^{n-k}$. $\binom{n}{k}$ ("n choose k") counts how many such sequences there are.

Mean $= np$, variance $= np(1-p)$.

**Worked.** 20% of emails are spam. In a batch of 10, what's the chance exactly 3 are spam?

1. $\binom{10}{3} = 120$ ways to choose which 3
2. $0.2^3 \times 0.8^7 \approx 0.008 \times 0.2097 \approx 0.001678$
3. $120 \times 0.001678 \approx 0.201$

So about 20%. Expected number of spam emails: $np = 10 \times 0.2 = 2$.

## Poisson distribution

**Use when:** you count events in a fixed interval of time or space, events happen independently at a constant average rate.

- $\lambda$ (lambda): the average number of events per interval. It is both the mean **and** the variance.

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

**Worked.** A support desk gets on average 3 tickets an hour. Chance of exactly 5 in the next hour?

$$
P(X = 5) = \frac{3^5 e^{-3}}{5!} = \frac{243 \times 0.0498}{120} \approx 0.101
$$

And the chance of none at all: $P(X = 0) = e^{-3} \approx 0.050$.

![Left: Binomial with n = 10, p = 0.2, peaking at 2 with P(X = 3) highlighted. Right: Poisson with λ = 3, with P(X = 5) highlighted.](figures/distributions-binom-poisson.svg)

In the figure, both are right-skewed: counts can't go below 0 but can stretch upwards.

## Uniform distribution

Every value in a range $[a, b]$ is equally likely. A flat line.

$$
f(x) = \frac{1}{b - a} \text{ for } a \le x \le b, \qquad \text{mean} = \frac{a+b}{2}
$$

Example: a random number between 0 and 10. $f(x) = 0.1$, and $P(X \le 3) = 3 \times 0.1 = 0.3$. Its discrete cousin is a fair die. Uniform is what `np.random.default_rng().uniform()` draws from, and it's often the starting point for simulations and random hyperparameter search.

## The sampling distribution

Here is the bridge from describing data to testing it. From Jag's notes:

> Take a different sample and you get a different estimate. Plot every estimate you could have got and they form a bell curve centred on the true parameter. That curve is the **sampling distribution**. Everything that follows is measured on it.

The data have a distribution. The **estimate** (e.g. the sample mean $\bar{x}$) has a distribution too, because it would change if you resampled. That second distribution is the sampling distribution.

## The Central Limit Theorem

**Statement:** take samples of size $n$ from **any** population with mean $\mu$ and finite SD $\sigma$. As $n$ grows, the distribution of the sample mean $\bar{x}$ becomes approximately normal:

$$
\bar{x} \;\approx\; N\!\left(\mu,\ \frac{\sigma^2}{n}\right)
$$

whatever shape the population has.

Three things in that line:

1. Centred on $\mu$: the sample mean is right on average.
2. Spread $\sigma / \sqrt{n}$: it gets tighter as $n$ grows.
3. Normal shape: even if the population is skewed.

Why it happens, intuitively: in a sample mean, unusually large and unusually small values tend to cancel. Extreme means need many extreme values at once, which is rare.

![Far left: a strongly right-skewed population. Then the histograms of 4,000 sample means for n = 2, 5 and 30. By n = 30 the histogram sits on the gold normal curve.](figures/distributions-clt.svg)

In the figure, watch two things move left to right: the skew disappears, and the spread narrows from $2/\sqrt{2} \approx 1.41$ to $2/\sqrt{30} \approx 0.37$.

> **Trap:** the CLT is about the **mean of the sample**, not the data. A big sample of skewed data is still skewed. Its *mean* is what becomes normal.

"Large enough" is often quoted as $n \ge 30$. That's a rule of thumb, not a law: very skewed populations need more, nearly symmetric ones less.

## Standard deviation vs standard error

From Jag's notes, kept as he wrote them: *These get confused constantly. They measure different things.*

| | Standard deviation ($s$ or $\sigma$) | Standard error (SE) |
|---|---|---|
| Measures spread of | Individual data points | The estimate itself, across repeated samples |
| Answers | How varied are the people in my sample? | How much would my answer wobble if I resampled? |
| Effect of larger $n$ | Doesn't shrink: it's a property of the population | **Shrinks**, as $\sqrt{n}$ |
| Used for | Describing the data | Building intervals and tests |

$$
\text{SE (mean)} = \frac{\sigma}{\sqrt{n}} \approx \frac{s}{\sqrt{n}} \qquad \text{SE (proportion)} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

> **Key idea:** the standard error is the standard deviation of the sampling distribution. That single sentence is the link between the two.

Where $\sigma/\sqrt{n}$ comes from: the variance of a sum of $n$ independent values is $n\sigma^2$. Divide the sum by $n$ to get the mean, and the variance divides by $n^2$: $n\sigma^2 / n^2 = \sigma^2/n$. Square-root it: $\sigma/\sqrt{n}$.

**Worked.** Cereal packets weigh on average 500 g with $\sigma = 10$ g. Take 25 packets: SE $= 10 / \sqrt{25} = 2$ g. Individual packets vary by about 10 g; the average of 25 varies by only about 2 g.

![Grey: the spread of individual packets (SD 10). Teal: the spread of averages of 25 packets (SE 2), five times narrower.](figures/distributions-sd-se.svg)

In the figure, both curves are centred at 500. To halve the SE you need four times the data, since $\sqrt{4} = 2$.

## How this feeds hypothesis testing

Everything in [Hypothesis testing](#/t/hypothesis) and [Confidence intervals](#/t/ci) is the z-score idea applied to the sampling distribution instead of to raw data.

**A test statistic is a z-score of your estimate on its sampling distribution:**

$$
z = \frac{\text{estimate} - \text{hypothesised value}}{\text{SE}} = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

Compare with $z = (x - \mu)/\sigma$ for a single data point: same shape, but the data point is replaced by the estimate and $\sigma$ by the SE.

**Worked, continuing the cereal.** The factory claims $\mu_0 = 500$ g. Your 25 packets average $\bar{x} = 504$ g. Then

$$
z = \frac{504 - 500}{2} = 2.0
$$

Your average is 2 standard errors above the claim.

**Critical values** are the z-scores that cut off the tails you decide are "too unusual". For a two-sided test at $\alpha = 0.05$, each tail holds $0.025$, so you look up $1 - 0.025 = 0.975$ in the CDF: $z^* = 1.96$.

![The sampling distribution of z if the claim is true. The wine tails beyond ±1.96 hold 5% in total. Our z = 2.0 (gold) lands just inside the right tail.](figures/distributions-test-stat.svg)

In the figure, 2.0 is past 1.96, so at the 5% level the result counts as significant. The two-sided p-value is the area beyond $\pm 2.0$, about 0.046.

The same machinery builds a confidence interval by running it backwards: $\bar{x} \pm z^* \times \text{SE} = 504 \pm 1.96 \times 2$, roughly 500.1 to 507.9 g.

> **Key idea:** the CLT is what lets you use the normal curve here even when the raw data aren't normal. Without it, none of these z-based tests would be justified.

## Code

```python
import numpy as np
from scipy import stats

# Normal: heights N(175, 7^2)
print(stats.norm.cdf(185, loc=175, scale=7))   # 0.923  P(X <= 185)
print(stats.norm.sf(185, loc=175, scale=7))    # 0.077  P(X > 185) = 1 - cdf
print(stats.norm.pdf(0))                       # 0.399  peak density of N(0,1)
print(stats.norm.ppf(0.975))                   # 1.96   inverse CDF: critical value

# Binomial: n = 10 trials, p = 0.2
print(stats.binom.pmf(3, n=10, p=0.2))         # 0.201  P(X = 3)
print(stats.binom.cdf(2, n=10, p=0.2))         # 0.678  P(X <= 2)

# Poisson: lambda = 3 events per hour
print(stats.poisson.pmf(5, mu=3))              # 0.101  P(X = 5)
print(stats.poisson.pmf(0, mu=3))              # 0.050  P(X = 0)

# Uniform on [0, 10]: scipy uses loc = a, scale = b - a
print(stats.uniform.cdf(3, loc=0, scale=10))   # 0.3

# Simulating the CLT
rng = np.random.default_rng(0)                 # seeded so results repeat
for n in [2, 5, 30]:
    samples = rng.exponential(2, size=(4000, n))  # 4000 samples of size n
    means = samples.mean(axis=1)               # one mean per sample (row)
    print(n, means.mean().round(2),            # stays near the true mean 2
          means.std().round(3),                # close to 2 / sqrt(n)
          round(2 / np.sqrt(n), 3),            # theoretical SE
          round(stats.skew(means), 2))         # skew shrinks towards 0
```

> **Trap:** `pmf` for discrete distributions, `pdf` for continuous. `binom.pdf` doesn't exist. `ppf` is the inverse of `cdf` (probability in, value out).

## Quick reference

| Term | Formula / rule | Example |
|---|---|---|
| PMF | $P(X = k)$, discrete | binom pmf(3) = 0.201 |
| PDF | density; probability = area | norm area 0 to 1 = 0.341 |
| CDF | $F(x) = P(X \le x)$ | $\Phi(1) = 0.841$ |
| Normal | $N(\mu, \sigma^2)$, 68–95–99.7 | heights $N(175, 7^2)$ |
| z-score | $z = (x - \mu)/\sigma$ | 185 cm → 1.43 |
| Binomial | $\binom{n}{k}p^k(1-p)^{n-k}$, mean $np$ | 10 emails, p = 0.2 |
| Poisson | $\lambda^k e^{-\lambda}/k!$, mean = var = $\lambda$ | 3 tickets/hour |
| Uniform | $1/(b-a)$ on $[a, b]$ | $P(X \le 3) = 0.3$ on [0, 10] |
| Sampling distribution | distribution of an estimate over repeated samples | — |
| CLT | $\bar{x} \approx N(\mu, \sigma^2/n)$ for large $n$ | any population shape |
| Standard error | $\sigma/\sqrt{n}$ = SD of the sampling distribution | 10/√25 = 2 |
| Test statistic | $(\bar{x} - \mu_0)/\text{SE}$ | (504 − 500)/2 = 2.0 |
| Critical value (95%, two-sided) | $z^* = 1.96$ | look up 0.975 |
