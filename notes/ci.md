# Confidence intervals

A confidence interval answers one question: **my sample gave one number, so what range could the true number plausibly be in?**

**On this page**

- Parameter and point estimate: the true number vs your guess
- Standard error: how much your guess would wobble
- Confidence level and α: how much you capture
- The critical value: where the edges are
- Building the interval, step by step
- Precision, confidence, sample size: pick two
- Saying it in words

## Parameter and point estimate

> **Definition:** the **parameter** is the true value in the whole population: fixed, but unknown. The **point estimate** is the value you calculate from your sample: your single best guess at it.

True share of all residents who recycle: the parameter (p). Share in your sample: the point estimate (p̂). A confidence interval turns that guess into a **range of plausible values**.

## Standard error

> **Definition:** the **standard error (SE)** is how much your estimate would wobble if you took a fresh sample and recalculated.

Every sample gives a slightly different estimate. Plotted, they form a bell curve around the true value (the **sampling distribution**). The SE is its spread.

$$
SE = \frac{SD}{\sqrt{n}}
$$

SD is the spread of individual data points; n is the sample size.

![Grey: single packets, SD 10. Teal: averages of 25 packets, SE 10 / √25 = 2. Five times narrower.](figures/distributions-sd-se.svg)

> **Where it goes wrong:** SD and SE are not the same thing. The **SD** answers "how varied are the people?" and doesn't shrink with more data. The **SE** answers "how much would my answer wobble if I resampled?" and shrinks as √n. Build the interval from the SE, never the SD.

## Confidence level and α

> **Definition:** the **confidence level** is the share of the curve you choose to capture in the middle (90%, 95%, 99%). **α** is everything you don't capture: the leftover, split between the two tails.

For 99%: α = 1 − 0.99 = 0.01, so **0.005 in each tail**.

![Teal: the middle 99%. Red slivers: the 0.005 tails. Dashed lines at ±2.576: the cutoffs you look up next.](figures/ci-tails.svg)

## The critical value

> **Definition:** the **critical value** (z* or t*) is the score marking the edge of the middle band.

The one rule about z-tables: a table gives the area to the **left** of a single point, never an interval. Below the upper cutoff sits the middle **plus** the lower tail, so look up 1 − α/2.

| Confidence | Look up | z* |
|---|---|---|
| 90% | 0.950 | 1.645 |
| 95% | 0.975 | 1.960 |
| 99% | 0.995 | 2.576 |

- **Proportion** (counting yes/no): use z*.
- **Mean** (measuring a quantity): use t*, with n − 1 degrees of freedom. Slightly bigger than z*; near-identical past n ≈ 30.

> **Where it goes wrong:** for a 99% interval, look up **0.995**, not 0.99. Choosing 0.995 is how you account for the lower tail; the table just returns a position.

## Building the interval

> **Definition:** the **margin of error (MoE)** is how far you reach either side of your estimate. The **confidence interval** is estimate ± margin of error.

$$
CI = \text{estimate} \pm \text{critical value} \times SE
$$

**Worked example.** 200 people sampled, 120 recycle, 99% confidence.

| Step | Working | Result |
|---|---|---|
| 1. Estimate | 120 / 200 | p̂ = 0.60 |
| 2. Check | 120 yes, 80 no: both at least 10 | fine |
| 3. SE | √(0.6 × 0.4 / 200) | 0.0346 |
| 4. Critical value | look up 0.995 | z* = 2.576 |
| 5. MoE | 2.576 × 0.0346 | 0.0892 |
| 6. Interval | 0.60 ± 0.0892 | **(0.511, 0.689)** |

For yes/no data, √(0.6 × 0.4) plays the part of the SD.

**For a mean**, same steps: n = 25, x̄ = 68, SD = 10, 95%. SE = 2.0, t* (df 24) = 2.064, MoE = 4.13, so **(63.87, 72.13)**.

```python
import numpy as np
from scipy import stats

p_hat, n = 120 / 200, 200
se = np.sqrt(p_hat * (1 - p_hat) / n)        # 0.0346
z_star = stats.norm.ppf(0.995)               # 2.576 (the z-table, backwards)
print(p_hat - z_star*se, p_hat + z_star*se)  # 0.511, 0.689
```

## Precision, confidence, sample size

> **Definition:** these three are locked together: **precision, confidence, sample size: pick two, the third follows.**

- Raise confidence (95% → 99%): bigger critical value, **wider** interval.
- Raise the sample size: smaller SE, **narrower** interval.
- Because SE shrinks with √n, **halving the margin of error needs 4× the sample.**

![Margin of error against sample size. Each curve flattens: the first few hundred people buy a lot of precision, the next thousand very little.](figures/ci-moe-n.svg)

## Saying it in words

> **Definition:** the interval is about **this one result**; the confidence level is about **the method**. Two different sentences.

> **Say it like this:** "We are 99% confident that the true proportion of all residents who recycle is between 51.1% and 68.9%. If we repeated this sampling many times, about 99% of the intervals would contain the true proportion."

![Fifty 95% intervals, one per sample. 3 (red) miss the dashed true mean, about 1 in 20. From inside one study you can't tell if yours is red.](figures/ci-coverage.svg)

> **Where it goes wrong:** "there's a 99% chance the true value is in this interval" is wrong. The true value is fixed; the **interval** moves each time you resample.

**Using it to decide:** 50% sits outside (51.1%, 68.9%), so 50% is implausible. That's the bridge to [Hypothesis testing](#/t/hypothesis).

## Quick reference

| Term | In one line |
|---|---|
| Parameter | True population value. Fixed, unknown |
| Point estimate | Your sample's best guess (p̂ or x̄) |
| Standard error | How much the estimate wobbles: SD / √n |
| α | 1 − confidence level, split across two tails |
| Critical value | Look up 1 − α/2. Proportion → z, mean → t |
| Margin of error | Critical value × SE |
| Confidence interval | Estimate ± margin of error |

**Order:** point estimate → check → SE → α → α/2 → critical value → MoE → interval → sentence.
