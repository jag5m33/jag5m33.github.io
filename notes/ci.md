# Confidence intervals

A confidence interval turns one guess from a sample into a range of plausible values for the true population number, with a stated level of confidence in the method.

## 1. The setting

| Term | Meaning | Symbol |
|---|---|---|
| Population | Everyone you care about | — |
| Parameter | True value in the population. Fixed, unknown | μ (mean), p (proportion) |
| Sample | The subset you measured | n = how many |
| Point estimate | Value calculated from the sample. Your single best guess | x̄ (mean), p̂ (proportion) |

A confidence interval turns that single guess into a **range of plausible values** for the parameter.

## 2. The sampling distribution

Take a different sample and you get a different estimate. Plot every estimate you could have got and they form a bell curve centred on the true parameter. That curve is the **sampling distribution**. Everything that follows is measured on it.

It is bell-shaped because of the **Central Limit Theorem**: for a large enough sample, the distribution of the estimate is approximately normal regardless of the shape of the underlying data (see [Distributions & the CLT](#/t/distributions)).

## 3. Standard deviation vs standard error

These get confused constantly. They measure different things.

| | Standard deviation (s or σ) | Standard error (SE) |
|---|---|---|
| Measures spread of | Individual data points | The estimate itself, across repeated samples |
| Answers | How varied are the people in my sample? | How much would my answer wobble if I resampled? |
| Effect of larger n | Doesn't shrink — it's a property of the population | **Shrinks**, as √n |
| Used for | Describing the data | Building intervals and tests |

$$
SE_{\text{mean}} = \frac{s}{\sqrt{n}} \qquad SE_{\text{proportion}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

**Where the √n comes from.** The variance of a sum of n independent values is n times the variance of one, so the variance of their average is $n\sigma^2 / n^2 = \sigma^2/n$. Take the square root and the spread of the average is $\sigma/\sqrt{n}$. For a proportion, each person is a 0/1 value with variance $p(1-p)$, so the same logic gives $\sqrt{p(1-p)/n}$.

> **Key idea:** The standard error is the standard deviation of the sampling distribution. That single sentence is the link between the two.

![Gold: 4,000 individual scores with SD about 10. Teal: 2,000 sample means (n = 25 each), squeezed to an SE of about 2.](figures/ci-sd-vs-se.svg)

In the figure, the wide gold histogram is the people; the narrow teal one is the averages. Same centre, five times narrower, because √25 = 5.

## 4. Confidence level and alpha

- **Confidence level** — the share of the sampling distribution you choose to capture in the middle. 90%, 95%, 99%.
- **Alpha (α)** — everything you don't capture. The leftover, split between the two tails.

$$
\alpha = 1 - \text{confidence level} \qquad \text{each tail} = \frac{\alpha}{2}
$$

For 99%: α = 0.01, and α/2 = **0.005** in each tail.

![The standard normal curve with the middle 99% shaded teal, the two 0.005 tails in red and dashed cutoffs at ±2.576.](figures/ci-tails.svg)

In the figure, the wide middle band is the 99%; the two red slivers at the edges are the 0.005 tails; the dashed lines are the cutoffs you're about to look up.

## 5. Critical value

**Critical value** — the score marking the edge of the middle band. Written z* or t*.

**The one rule about z-tables:** a table gives the area to the **left** of a single point. Never an interval. So to find the upper cutoff, ask how much area sits below it. Below it = the middle band **+** the lower tail:

$$
\text{area to look up} = 1 - \frac{\alpha}{2}
$$

For 99%: 1 − 0.005 = **0.995**. You account for the lower tail yourself when you choose 0.995 rather than 0.99. The table just returns a position.

| Confidence | α | α/2 | Look up | z* |
|---|---|---|---|---|
| 90% | 0.10 | 0.05 | 0.950 | 1.645 |
| 95% | 0.05 | 0.025 | 0.975 | 1.960 |
| 99% | 0.01 | 0.005 | 0.995 | 2.576 |

The lower cutoff is the mirror image, −z*. One lookup gives both bounds — in the figure above, ±2.576 are the dashed lines.

**Z-table walkthrough (99%).** Scan the body of the table for the area closest to 0.9950. It sits between 0.9949 (row 2.5, column 0.07) and 0.9951 (row 2.5, column 0.08). Row + column gives z ≈ 2.57–2.58; the precise value is 2.576.

```python
from scipy import stats

conf = 0.99
alpha = 1 - conf                      # leftover: 0.01
z_star = stats.norm.ppf(1 - alpha/2)  # ppf = inverse of the table: area 0.995 -> z
print(round(z_star, 3))               # 2.576

t_star = stats.t.ppf(0.975, df=24)    # same idea for t: 95%, df = 24
print(round(t_star, 3))               # 2.064
```

`ppf` ("percent point function") is the z-table run backwards: you give it the area to the left, it gives you the position.

## 6. Z or t — which one, and the assumptions

| | Z | t |
|---|---|---|
| Use for | Proportions; means when σ is known | Means when σ is unknown (almost always) |
| Shape | Fixed normal curve | Normal but fatter tails |
| Needs | — | Degrees of freedom, df = n − 1 |
| As n grows | — | Converges to Z (near-identical past n ≈ 30) |

**Degrees of freedom** — how many independent pieces of information are left to estimate the spread. You used one up computing x̄, so df = n − 1.

Fatter tails mean a larger critical value, so a t interval is **wider** than the Z equivalent — the price of not knowing σ and having to estimate it with s.

![Z curve and t curves for df = 2, 5 and 24. Lower df gives fatter tails and a larger 95% critical value.](figures/ci-z-vs-t.svg)

In the figure, look at the tails: df = 2 needs t* = 4.303 to capture 95%, df = 24 needs 2.064, and Z needs 1.960.

**Assumptions that must hold either way:**

| Assumption | Mean | Proportion |
|---|---|---|
| Random sample | Required | Required |
| Independence | 10% rule: n < 10% of population | 10% rule |
| Approximate normality | n ≥ 30, or the data are roughly normal | **Success–failure: np̂ ≥ 10 and n(1−p̂) ≥ 10** |

If the success–failure condition fails, the normal approximation is invalid and the interval shouldn't be trusted.

## 7. Margin of error

**Margin of error (MoE)** — how far you reach either side of the point estimate.

$$
\text{MoE} = \text{critical value} \times SE
$$

| Change | Effect on MoE | Why |
|---|---|---|
| Higher confidence (95% → 99%) | Wider | Bigger z* |
| Larger n | Narrower | SE shrinks as √n |
| p̂ nearer 0.5 | Wider | p̂(1−p̂) is largest at 0.5 |

Because of the √n, **halving the margin of error requires 4× the sample; quartering it requires 16×.**

## 8. The confidence interval

**Confidence interval** — point estimate ± margin of error.

$$
\text{CI} = \text{point estimate} \pm (\text{critical value} \times SE)
$$

| Parameter | Formula |
|---|---|
| Proportion | p̂ ± z* √( p̂(1−p̂) / n ) |
| Mean, σ unknown | x̄ ± t* ( s / √n ) |
| Mean, σ known | x̄ ± z* ( σ / √n ) |

### Worked example — proportion

200 people sampled, 120 recycle, 99% confidence.

| Step | Working | Result |
|---|---|---|
| 1. Point estimate | 120 / 200 | p̂ = 0.60 |
| 2. Check conditions | np̂ = 120, n(1−p̂) = 80 | Both ≥ 10 ✓ |
| 3. Standard error | √(0.6 × 0.4 / 200) | SE = 0.0346 |
| 4. α and tail | 1 − 0.99 = 0.01, halved | α/2 = 0.005 |
| 5. Look up 1 − α/2 | 0.995 in the table | z* = 2.576 |
| 6. Margin of error | 2.576 × 0.0346 | MoE = 0.0892 |
| 7. Interval | 0.60 ± 0.0892 | **(0.511, 0.689)** |

### Worked example — mean

n = 25 sampled, x̄ = 68, s = 10, 95% confidence, σ unknown so use t.

| Step | Working | Result |
|---|---|---|
| 1. Standard error | 10 / √25 | SE = 2.0 |
| 2. Degrees of freedom | 25 − 1 | df = 24 |
| 3. Critical value | t table, df 24, 95% | t* = 2.064 |
| 4. Margin of error | 2.064 × 2.0 | MoE = 4.13 |
| 5. Interval | 68 ± 4.13 | **(63.87, 72.13)** |

### Both in code

```python
import numpy as np
from scipy import stats

# --- proportion: 120 of 200 recycle, 99% ---
n, successes, conf = 200, 120, 0.99
p_hat = successes / n                           # point estimate 0.60
se = np.sqrt(p_hat * (1 - p_hat) / n)           # standard error 0.0346
z_star = stats.norm.ppf(1 - (1 - conf) / 2)     # look up 0.995 -> 2.576
moe = z_star * se                               # margin of error 0.0892
print(p_hat - moe, p_hat + moe)                 # (0.511, 0.689)

# --- mean: n = 25, x-bar = 68, s = 10, 95%, sigma unknown -> t ---
n, xbar, s = 25, 68, 10
se = s / np.sqrt(n)                             # 2.0
lo, hi = stats.t.interval(0.95, df=n - 1, loc=xbar, scale=se)  # t* x SE either side
print(lo, hi)                                   # (63.87, 72.13)

# --- from raw data instead of summary numbers ---
data = np.array([61, 72, 65, 70, 74, 66, 69, 63])
ci = stats.t.interval(0.95, df=len(data) - 1,
                      loc=data.mean(), scale=stats.sem(data))  # sem = s / sqrt(n)
```

## 9. Picking the right procedure

Run this in order before calculating anything:

1. **Proportion or mean?** Counting yes/no → proportion. Measuring a quantity → mean.
2. **Proportion** → use z*. Check np̂ ≥ 10 and n(1−p̂) ≥ 10.
3. **Mean, σ known** → use z*. Rare; usually only in textbook questions.
4. **Mean, σ unknown** → use t*, df = n − 1. This is the default in practice.
5. **Either** → random sample, and n < 10% of the population.

## 10. Determining sample size

Works backwards: you fix the margin of error you're willing to accept, then solve for n. Start from MoE = z* √(p̂(1−p̂)/n), square both sides and rearrange:

$$
n = \frac{z^{*2}\,\hat{p}(1-\hat{p})}{\text{MoE}^2} \qquad\qquad n = \left(\frac{z^{*}\sigma}{\text{MoE}}\right)^2
$$

(left: proportion, right: mean).

- Always **round up** — a fractional person doesn't exist, and rounding down misses your target margin.
- **No prior estimate of p̂?** Use p̂ = 0.5. It maximises p̂(1−p̂), so it gives the largest, safest n.

**Example.** Want MoE = 0.03 at 95% confidence, no prior estimate.

$$
n = \frac{1.96^2 \times 0.25}{0.03^2} = \frac{3.8416 \times 0.25}{0.0009} = 1067.1 \;\rightarrow\; n = 1068
$$

```python
import math
from scipy import stats
z = stats.norm.ppf(0.975)                 # 1.96 for 95%
n = z**2 * 0.5 * 0.5 / 0.03**2            # p-hat = 0.5 is the safe worst case
print(math.ceil(n))                       # always round UP -> 1068
```

## 11. How they all move together

The three quantities are locked in one equation, so fixing any two sets the third.

$$
\text{MoE} = z^{*} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

| If you… | And hold… | Then… |
|---|---|---|
| Raise confidence | n fixed | MoE grows — wider interval |
| Raise n | Confidence fixed | MoE shrinks — tighter interval |
| Demand a smaller MoE | Confidence fixed | n must grow, by the square |
| Raise confidence and hold MoE | — | n must grow |

![Margin of error against sample size at 90%, 95% and 99% confidence, with n = 1068 marked at MoE 0.03.](figures/ci-moe-n.svg)

In the figure, each curve flattens out: the first few hundred people buy a lot of precision, the next thousand buy very little. Higher confidence lifts the whole curve.

> **Key idea:** Precision, confidence, sample size — pick two, the third follows.

## 12. Saying it in words

Two different sentences, and they are not interchangeable.

**Interpreting the confidence interval** (about this one interval):

> **Say it like this:** "We are 99% confident that the true [parameter in context] is between [lower] and [upper]."

**Interpreting the confidence level** (about the method):

> **Say it like this:** "If we repeated this sampling many times and built an interval each time, about 99% of those intervals would contain the true [parameter]."

| Result | Sentence |
|---|---|
| Proportion: 0.60, 99% CI (0.511, 0.689) | "We are 99% confident that the true proportion of all residents who recycle is between 51.1% and 68.9%." |
| Mean: x̄ = 68, s = 10, n = 25, 95% CI (63.87, 72.13) | "We are 95% confident that the true mean score for the population is between 63.87 and 72.13." |

**Reading a result stated as "mean 68, sd 10, n = 25":** the sd describes the spread of the 25 individual scores. Divide by √n to get the standard error (2.0), which is the wobble in the mean. The interval is built from the standard error, never from the sd directly.

**Using an interval to decide:** if a claimed value falls outside the interval, that value is implausible at this confidence level. For a difference or a change, check whether the interval contains **0**; for a ratio, whether it contains **1**. This is the bridge to [Hypothesis testing](#/t/hypothesis).

## 13. Traps

### What "95%" really refers to

The true value is fixed. The **interval** is the random thing — it moves every time you resample. "95%" is a property of the method: across many samples, 95% of the intervals it builds will catch the truth. Any single interval either contains μ or it doesn't.

![Fifty 95% intervals from fifty simulated samples (n = 25, true mean 68). 47 cross the dashed true-mean line; 3, in red, miss it.](figures/ci-coverage.svg)

In the figure, every horizontal bar is one sample's interval. The dashed line never moves. Most bars cross it; three (red) miss — about the 1-in-20 you'd expect. From inside one study you cannot tell whether yours is a red one.

| Wrong | Right |
|---|---|
| "There's a 99% chance the true value is in this interval" | The true value is fixed. The interval is what varies between samples |
| Looking up 0.99 for a 99% interval | Look up 0.995 — that's 1 − α/2 |
| Using the sd where the SE belongs | Divide the sd by √n first |
| Using z* for a mean with unknown σ | Use t*, df = n − 1 |
| Rounding n down | Always round sample size up |
| Reporting an interval without checking conditions | Check success–failure or n ≥ 30 first |
| "95% of the data lie in the interval" | The CI is about the parameter (e.g. the mean), not individual values |

## Quick reference

| Term | One line |
|---|---|
| Parameter | True population value. Fixed, unknown |
| Point estimate | Your sample's single best guess (p̂ or x̄) |
| Sampling distribution | Bell curve of all estimates you could have got |
| Standard deviation | Spread of individual data points |
| Standard error | Spread of the estimate. sd / √n |
| Confidence level | Middle share of the curve you capture |
| Alpha (α) | 1 − confidence level. The leftover |
| α/2 | The leftover in one tail |
| Critical value (z*, t*) | Score at the edge of the middle band. Look up 1 − α/2 |
| Margin of error | critical value × SE |
| Confidence interval | point estimate ± margin of error |
| Sample size (proportion) | n = z*² p̂(1−p̂) / MoE², round up |

**The order of calculation:** point estimate → check conditions → SE → α → α/2 → critical value → MoE → interval → sentence.
