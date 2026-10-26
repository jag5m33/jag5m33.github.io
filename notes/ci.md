# Confidence intervals

A confidence interval turns one guess from a sample into a range of plausible values for the true population number, with a stated level of confidence in the method.

## The setting

| Term | Meaning |
|---|---|
| **Parameter** | The true value in the population. Fixed, unknown (e.g. the true mean $\mu$, the true proportion $p$) |
| **Point estimate** | The value you calculate from your sample. Your single best guess ($\bar{x}$ for a mean, $\hat{p}$ for a proportion) |
| **Confidence interval** | A range of plausible values for the parameter, built around the point estimate |

## The building blocks

### Sampling distribution and standard error

Take a different sample and you get a different estimate. Plot every estimate you could have got and they form a bell curve centred on the true parameter. That curve is the **sampling distribution**. Everything that follows is measured on it.

The **standard error (SE)** is the spread of that curve: how much your estimate would wobble if you resampled. It is not the same as the SD, which is the spread of individual data points (see [Distributions & the CLT](#/t/distributions)).

$$
SE_{\text{mean}} = \frac{s}{\sqrt{n}} \qquad SE_{\text{proportion}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

- $s$: sample SD. $n$: sample size. $\hat{p}$: sample proportion.

### Confidence level and alpha

- **Confidence level**: the share of the curve you choose to capture in the middle. 90%, 95%, 99%.
- **Alpha (α)**: what's left over, split between the two tails. $\alpha = 1 - \text{confidence level}$, so each tail holds $\alpha/2$.

For 99%: α = 0.01, and **0.005** in each tail.

![The standard normal curve with the middle 99% shaded teal, the two 0.005 tails in red and dashed cutoffs at ±2.576.](figures/ci-tails.svg)

In the figure, the wide middle band is the 99%; the two red slivers are the 0.005 tails; the dashed lines are the cutoffs you're about to look up.

### Critical value

**Critical value** (written $z^*$ or $t^*$): the score marking the edge of the middle band.

**The one rule about z-tables:** a table gives the area to the **left** of a single point. Never an interval. Below the upper cutoff sits the middle band **plus** the lower tail, so you look up $1 - \alpha/2$. For 99%: **0.995**, not 0.99.

| Confidence | Look up | $z^*$ |
|---|---|---|
| 90% | 0.950 | 1.645 |
| 95% | 0.975 | 1.960 |
| 99% | 0.995 | 2.576 |

### Z or t: one plain rule

> **Key idea:** for a **proportion**, use $z^*$. For a **mean**, use $t^*$ with $n - 1$ degrees of freedom (because you don't know the true SD and have to estimate it). $t^*$ is a bit bigger than $z^*$ for small samples, and almost the same past about 30.

**Check before you trust it:** random sample; for a proportion, at least 10 "yes" and 10 "no" in the sample; for a mean, $n \ge 30$ or roughly bell-shaped data.

## The formula

$$
\text{CI} = \text{point estimate} \pm \text{critical value} \times SE
$$

The bit after the $\pm$ is the **margin of error (MoE)**: how far you reach either side.

| Change | Effect on the interval |
|---|---|
| Higher confidence (95% → 99%) | Wider (bigger critical value) |
| Larger sample | Narrower (SE shrinks) |

Because of the $\sqrt{n}$, **halving the margin of error needs 4× the sample.**

## Worked example: proportion

200 people sampled, 120 recycle, 99% confidence.

| Step | Working | Result |
|---|---|---|
| 1. Point estimate | 120 / 200 | $\hat{p}$ = 0.60 |
| 2. Check | 120 yes, 80 no | both ≥ 10, fine |
| 3. Standard error | √(0.6 × 0.4 / 200) | SE = 0.0346 |
| 4. Critical value | look up 0.995 | $z^*$ = 2.576 |
| 5. Margin of error | 2.576 × 0.0346 | MoE = 0.0892 |
| 6. Interval | 0.60 ± 0.0892 | **(0.511, 0.689)** |

## Worked example: mean

n = 25, $\bar{x}$ = 68, s = 10, 95% confidence. It's a mean, so use t.

| Step | Working | Result |
|---|---|---|
| 1. Standard error | 10 / √25 | SE = 2.0 |
| 2. Degrees of freedom | 25 − 1 | df = 24 |
| 3. Critical value | t table, df 24, 95% | $t^*$ = 2.064 |
| 4. Margin of error | 2.064 × 2.0 | MoE = 4.13 |
| 5. Interval | 68 ± 4.13 | **(63.87, 72.13)** |

```python
import numpy as np
from scipy import stats

# Proportion: 120 of 200 recycle, 99%
p_hat = 120 / 200                               # 0.60
se = np.sqrt(p_hat * (1 - p_hat) / 200)         # 0.0346
z_star = stats.norm.ppf(0.995)                  # 2.576 (ppf = z-table backwards)
print(p_hat - z_star*se, p_hat + z_star*se)     # (0.511, 0.689)

# Mean: n = 25, mean 68, s = 10, 95%
se = 10 / np.sqrt(25)                           # 2.0
print(stats.t.interval(0.95, df=24, loc=68, scale=se))   # (63.87, 72.13)
```

## Sample size

Work backwards: pick the margin of error you want, then solve for $n$. For a proportion:

$$
n = \frac{z^{*2} \times \hat{p}(1-\hat{p})}{\text{MoE}^2}
$$

No prior guess for $\hat{p}$? Use 0.5: it gives the largest, safest $n$. Always **round up**.

**Example.** MoE = 0.03 at 95%: $n = 1.96^2 \times 0.25 / 0.03^2 = 1067.1$, so **1068** people.

![Margin of error against sample size at 90%, 95% and 99% confidence, with n = 1068 marked at MoE 0.03.](figures/ci-moe-n.svg)

In the figure, each curve flattens out: the first few hundred people buy a lot of precision, the next thousand buy very little.

> **Key idea:** precision, confidence, sample size: pick two, the third follows.

## Saying it in words

Two different sentences, and they are not interchangeable.

> **Say it like this (the interval):** "We are 99% confident that the true proportion of all residents who recycle is between 51.1% and 68.9%."

> **Say it like this (the method):** "If we repeated this sampling many times and built an interval each time, about 99% of those intervals would contain the true proportion."

**Using an interval to decide:** if a claimed value falls outside the interval, it is implausible at this confidence level. For a difference, check whether the interval contains **0**. This is the bridge to [Hypothesis testing](#/t/hypothesis).

## Traps

The true value is fixed. The **interval** is the thing that moves every time you resample. "95%" describes the method, not one interval.

![Fifty 95% intervals from fifty simulated samples (n = 25, true mean 68). 47 cross the dashed true-mean line; 3, in red, miss it.](figures/ci-coverage.svg)

In the figure, every bar is one sample's interval. Most cross the dashed true line; three (red) miss, about the 1-in-20 you'd expect. From inside one study you cannot tell whether yours is a red one.

| Wrong | Right |
|---|---|
| "99% chance the true value is in this interval" | The true value is fixed; the interval varies |
| Looking up 0.99 for a 99% interval | Look up 0.995 |
| Using the SD where the SE belongs | Divide the SD by √n first |
| Rounding sample size down | Always round up |
| "95% of the data lie in the interval" | The CI is about the mean, not individual values |

## Quick reference

| Term | One line |
|---|---|
| Point estimate | Your sample's best guess ($\hat{p}$ or $\bar{x}$) |
| Standard error | Spread of the estimate: SD / √n |
| Alpha (α) | 1 − confidence level, split across two tails |
| Critical value | Look up 1 − α/2. Proportion → z, mean → t |
| Margin of error | critical value × SE |
| Confidence interval | point estimate ± margin of error |
| Sample size | $z^{*2}\hat{p}(1-\hat{p})/\text{MoE}^2$, round up |

**Order of calculation:** point estimate → check → SE → α/2 → critical value → MoE → interval → sentence.
