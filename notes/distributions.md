# Distributions & the CLT

What a probability distribution is, the ones you'll meet most, and the Central Limit Theorem, which is why hypothesis tests and confidence intervals work.

## Random variables and distributions

A **random variable** is a number whose value comes from a random process.

- **Discrete**: countable values. Number of heads in 10 flips, accidents per day.
- **Continuous**: any value in a range. Height, temperature, time until a server fails.

A **distribution** describes which values are possible and how likely each one is.

## PMF, PDF and CDF

| Name | For | Gives you |
|---|---|---|
| **PMF** (probability mass function) | discrete | the probability of exactly one value. Bars add up to 1 |
| **PDF** (probability density function) | continuous | a curve where **area** under it is probability. Total area is 1 |
| **CDF** (cumulative distribution function) | both | the probability of that value **or less**. Rises from 0 to 1 |

![Left: a PMF, where each bar's height is a probability. Middle: a PDF, where only areas are probabilities (the gold area between 0 and 1 is 0.341). Right: the CDF, which adds everything up to a point.](figures/distributions-pmf-pdf-cdf.svg)

In the figure, look at the middle panel: for a continuous variable, the chance of *exactly* 1 is zero because a single line has no area. You can only ask about ranges.

The CDF is what Z-tables give you: in Jag's words, "a table gives the area to the left of a single point. Never an interval."

## The normal distribution

The bell curve. Two numbers describe it completely:

- $\mu$ (mu): the mean, where the centre sits.
- $\sigma$ (sigma): the standard deviation, how wide it is.

It is symmetric, so mean = median = mode.

### The 68–95–99.7 rule

For **any** normal distribution:

![Within 1σ of the mean: 68.3% of values. Within 2σ: 95.4%. Within 3σ: 99.7%.](figures/distributions-empirical-rule.svg)

In the figure, each band adds another $\sigma$ either side. Beyond $3\sigma$ only 0.3% is left, which is why "more than 3 SDs away" is a common [outlier rule](#/t/quartiles).

### The z-score

$$
z = \frac{x - \mu}{\sigma}
$$

- $x$: your value. $\mu$: the mean. $\sigma$: the SD.
- $z$: how many standard deviations $x$ is from the mean.

Subtract the mean to move the centre to 0, divide by the SD to make the spread 1. The result is the **standard normal**: mean 0, SD 1.

**Worked.** Adult heights have mean 175 cm, SD 7 cm. How unusual is 185 cm?

1. $z = (185 - 175) / 7 \approx 1.43$
2. The CDF at 1.43 is about 0.923
3. So about 92.3% of people are shorter, and about 7.7% are taller.

> **Say it like this:** "185 cm is 1.43 standard deviations above the mean, taller than about 92% of adults."

## Other distributions you'll meet

| Distribution | Use when | Example |
|---|---|---|
| **Binomial** | counting successes in a fixed number of yes/no trials | 20% of emails are spam; in a batch of 10, the chance exactly 3 are spam is about 0.20. Expected number: $10 \times 0.2 = 2$ |
| **Poisson** | counting events in a fixed time window at a steady average rate | a desk gets 3 tickets an hour on average; the chance of exactly 5 next hour is about 0.10 |
| **Uniform** | every value in a range equally likely | a random number between 0 and 10; chance it's 3 or less is 0.3 |

![Left: Binomial with 10 trials and p = 0.2, peaking at 2 with P(X = 3) highlighted. Right: Poisson with an average of 3, with P(X = 5) highlighted.](figures/distributions-binom-poisson.svg)

In the figure, both are right-skewed: counts can't go below 0 but can stretch upwards.

## The sampling distribution

Here is the bridge from describing data to testing it. From Jag's notes:

> Take a different sample and you get a different estimate. Plot every estimate you could have got and they form a bell curve centred on the true parameter. That curve is the **sampling distribution**. Everything that follows is measured on it.

## The Central Limit Theorem

**In words:** take lots of samples of size $n$ from **any** population and work out each sample's mean. As $n$ grows, those means form a normal (bell-shaped) curve:

1. centred on the true mean,
2. with spread $\sigma / \sqrt{n}$, so tighter as $n$ grows,
3. bell-shaped even if the population is skewed.

Why: in an average, unusually big and unusually small values tend to cancel out.

![Far left: a strongly right-skewed population. Then the histograms of 4,000 sample means for n = 2, 5 and 30. By n = 30 the histogram sits on the gold normal curve.](figures/distributions-clt.svg)

In the figure, watch two things left to right: the skew disappears, and the spread narrows.

> **Trap:** the CLT is about the **mean of the sample**, not the data. A big sample of skewed data is still skewed. Its *mean* is what becomes normal.

"Large enough" is often quoted as $n \ge 30$. That's a rule of thumb: very skewed data need more.

## Standard deviation vs standard error

From Jag's notes, kept as he wrote them: *These get confused constantly. They measure different things.*

| | Standard deviation (SD) | Standard error (SE) |
|---|---|---|
| Measures spread of | Individual data points | The estimate itself, across repeated samples |
| Answers | How varied are the people in my sample? | How much would my answer wobble if I resampled? |
| Bigger sample | Doesn't shrink | **Shrinks** |
| Used for | Describing the data | Building intervals and tests |

$$
\text{SE} = \frac{s}{\sqrt{n}}
$$

- $s$: the sample SD. $n$: sample size.

> **Key idea:** the standard error is the standard deviation of the sampling distribution. That single sentence links the two.

**Worked.** Cereal packets weigh 500 g on average with SD 10 g. Take 25 packets: SE $= 10 / \sqrt{25} = 2$ g. Single packets vary by about 10 g; the average of 25 varies by only about 2 g.

![Grey: the spread of individual packets (SD 10). Teal: the spread of averages of 25 packets (SE 2), five times narrower.](figures/distributions-sd-se.svg)

In the figure, both curves are centred at 500. To halve the SE you need four times the data, since $\sqrt{4} = 2$.

## How this feeds hypothesis testing

A **test statistic** is just a z-score for your estimate, using the SE instead of the SD:

$$
z = \frac{\text{estimate} - \text{claimed value}}{\text{SE}}
$$

**Worked, continuing the cereal.** The factory claims 500 g. Your 25 packets average 504 g. $z = (504 - 500)/2 = 2.0$: your average is 2 standard errors above the claim.

A **critical value** is the cut-off for "too unusual". For 5% split across both tails, it is $1.96$.

![The sampling distribution of z if the claim is true. The wine tails beyond ±1.96 hold 5% in total. Our z = 2.0 (gold) lands just inside the right tail.](figures/distributions-test-stat.svg)

In the figure, 2.0 is past 1.96, so at the 5% level the result counts as significant (p ≈ 0.046). Run it backwards and you get a [confidence interval](#/t/ci): $504 \pm 1.96 \times 2$, roughly 500.1 to 507.9 g.

> **Key idea:** the CLT is what lets you use the normal curve here even when the raw data aren't normal.

## Code

```python
from scipy import stats

print(stats.norm.cdf(185, loc=175, scale=7))   # 0.923  share shorter than 185 cm
print(stats.norm.ppf(0.975))                   # 1.96   critical value for 95%
print(stats.binom.pmf(3, n=10, p=0.2))         # 0.201  exactly 3 spam in 10
print(stats.poisson.pmf(5, mu=3))              # 0.101  exactly 5 tickets
print(stats.uniform.cdf(3, loc=0, scale=10))   # 0.3    3 or less on [0, 10]
```

> **Trap:** `pmf` for discrete distributions, `pdf` for continuous. `ppf` is the reverse of `cdf` (probability in, value out).

## Quick reference

| Term | Rule | Example |
|---|---|---|
| PMF / PDF / CDF | exact value / area under curve / value or less | — |
| Normal | bell curve; 68–95–99.7 rule | heights, mean 175, SD 7 |
| z-score | $(x - \mu)/\sigma$ | 185 cm → 1.43 |
| Binomial | successes in $n$ yes/no trials | 3 spam in 10 → 0.20 |
| Poisson | events per time window | 5 tickets when average is 3 → 0.10 |
| Sampling distribution | spread of an estimate over repeated samples | — |
| CLT | sample means are bell-shaped for large $n$ | any population shape |
| Standard error | $s/\sqrt{n}$ | 10/√25 = 2 |
| Test statistic | (estimate − claim) / SE | (504 − 500)/2 = 2.0 |
| Critical value (95%, two-sided) | 1.96 | — |
