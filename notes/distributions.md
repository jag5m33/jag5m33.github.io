# Distributions & the CLT

A distribution answers one question: **which values can turn up, and how often?** The Central Limit Theorem is why the bell curve shows up in every test.

**On this page**

- What a distribution is
- The normal curve and the 68–95–99.7 rule
- The z-score: how far from the middle
- Counts: binomial and Poisson
- The sampling distribution and the CLT
- SD vs SE: the one everyone mixes up
- How it all feeds a test

## What a distribution is

> **Definition:** a **distribution** shows which values a random number can take and how likely each one is.

- **Discrete:** countable values. Heads in 10 flips, accidents per day. Each bar is a probability.
- **Continuous:** any value in a range. Height, temperature. Only **areas** under the curve are probabilities.

![Left: discrete bars, each height a probability. Middle: a continuous curve, where the gold area between 0 and 1 is the probability (0.341). Right: the running total, which climbs from 0 to 1.](figures/distributions-pmf-pdf-cdf.svg)

The right-hand panel (the running total, or CDF) is what a z-table gives you: **a table gives the area to the left of a single point. Never an interval.**

## The normal distribution

> **Definition:** the **normal distribution** is the symmetric bell curve. Two numbers describe it completely: the **mean** (where the centre sits) and the **SD** (how wide it is).

![For any normal curve: 68% of values within 1 SD of the mean, 95% within 2, 99.7% within 3. Beyond 3 SDs only 0.3% is left.](figures/distributions-empirical-rule.svg)

That last 0.3% is why "more than 3 SDs away" is a common [outlier rule](#/t/quartiles).

## The z-score

> **Definition:** a **z-score** is how many standard deviations a value sits from the mean.

$$
z = (\text{value} - \text{mean}) \div \text{SD}
$$

- Subtracting the mean moves the centre to 0. Dividing by the SD makes the spread 1. That's the **standard normal**.

**Worked example.** Adult heights: mean 175 cm, SD 7 cm. How unusual is 185 cm?

1. z = (185 − 175) ÷ 7 ≈ **1.43**
2. Area to the left of 1.43 ≈ **0.923**
3. So about 92% of people are shorter, 8% taller.

> **Say it like this:** "185 cm is 1.43 standard deviations above the mean, taller than about 92% of adults."

## Counts: binomial and Poisson

> **Definition:** **binomial** counts successes in a fixed number of yes/no tries. **Poisson** counts events in a fixed time window at a steady average rate.

- Binomial: 20% of emails are spam. In 10 emails, expect 2; the chance of exactly 3 is about **0.20**.
- Poisson: a help desk gets 3 tickets an hour on average. The chance of exactly 5 next hour is about **0.10**.

![Left: binomial, 10 tries at 20%, peaking at 2. Right: Poisson with an average of 3. Both lean right: counts can't go below 0 but can stretch upwards.](figures/distributions-binom-poisson.svg)

## The sampling distribution and the CLT

> **Definition:** the **sampling distribution** is the curve of every estimate you could have got from different samples. The **Central Limit Theorem (CLT)** says that curve, for a mean, is bell-shaped once n is big enough, whatever shape the data have.

Take a different sample and you get a different estimate. Plot every estimate you could have got and they form a bell curve centred on the true value. Everything in tests and intervals is measured on that curve. It's a bell because in an average, big and small values tend to cancel out.

![Far left: a strongly right-skewed population. Then 4,000 sample means for n = 2, 5 and 30. Left to right, the skew disappears and the spread narrows; by n = 30 they sit on the gold bell.](figures/distributions-clt.svg)

> **Where it goes wrong:** the CLT is about the **mean of the sample**, not the data. A big sample of skewed data is still skewed; its *mean* is what becomes bell-shaped. "Big enough" is usually n ≥ 30, more for very skewed data.

## SD vs SE

> **Definition:** the **standard deviation (SD)** is the spread of individual data points. The **standard error (SE)** is the spread of the estimate itself, across repeated samples.

| | SD | SE |
|---|---|---|
| Answers | How varied are the people in my sample? | How much would my answer wobble if I resampled? |
| Bigger sample | Doesn't shrink | Shrinks |
| Used for | Describing the data | Building intervals and tests |

$$
\text{SE} = \text{SD} \div \sqrt{n}
$$

- n: the sample size. Four times the data halves the SE, since √4 = 2.

**Worked example.** Cereal packets: mean 500 g, SD 10 g. Average 25 packets: SE = 10 ÷ √25 = **2 g**. Single packets vary by about 10 g; the average of 25 varies by only about 2 g.

![Grey: individual packets (SD 10). Teal: averages of 25 packets (SE 2), five times narrower. Both centred at 500.](figures/distributions-sd-se.svg)

The link in one sentence: **the standard error is the standard deviation of the sampling distribution.**

> **Where it goes wrong:** using the SD where the SE belongs. Intervals and tests are built from the SE, never the SD directly. Divide by √n first.

## How it all feeds a test

> **Definition:** a **test statistic** is just a z-score for your estimate, using the SE instead of the SD.

**Worked example.** The factory claims 500 g. Your 25 packets average 504 g. z = (504 − 500) ÷ 2 = **2.0**: 2 standard errors above the claim.

![If the claim is true, z follows this bell. The tails beyond ±1.96 hold 5% in total. Our z = 2.0 (gold) lands just inside the right tail.](figures/distributions-test-stat.svg)

2.0 is past the 1.96 cutoff, so it's significant at 5% (p ≈ 0.046). More in [Hypothesis testing](#/t/hypothesis) and [Confidence intervals](#/t/ci). The CLT is what lets you use the bell here even when the raw data aren't bell-shaped.

```python
from scipy import stats
stats.norm.cdf(185, loc=175, scale=7)   # 0.923  share shorter than 185 cm
stats.norm.ppf(0.975)                   # 1.96   cutoff for 95%
stats.binom.pmf(3, n=10, p=0.2)         # 0.20   exactly 3 spam in 10
```

## Quick reference

| Term | In one line |
|---|---|
| Distribution | Which values can turn up, and how often |
| Normal | Symmetric bell. 68% / 95% / 99.7% within 1 / 2 / 3 SDs |
| z-score | How many SDs from the mean |
| Binomial / Poisson | Successes in n tries / events per time window |
| Sampling distribution | Curve of every estimate you could have got |
| CLT | Sample means are bell-shaped for big n, whatever the data look like |
| SD / SE | Spread of the people / wobble of the estimate (SD ÷ √n) |

**Order:** mean and SD → SE = SD ÷ √n → z = (estimate − claim) ÷ SE → compare with the cutoff.
