# Effect size

An effect size says how big an effect is, in a number that does not grow just because you collected more data.

## The plainest version

**Effect size**: how big the effect actually is, in a number that does **not** grow with your sample size. A p-value tells you whether an effect is there; an effect size tells you how big it is.

The raw difference is already an effect size. A drug that lowers blood pressure by 8 points: that 8 is an effect size.

## Why divide by the spread

A raw difference on its own can't be compared across different measurements (8 mmHg vs 8 exam marks vs 8 seconds). So you measure the effect against the natural person-to-person spread.

**Two worlds.** An 8-point shift is trivial if people naturally vary by 20–30 points, but enormous if they vary by 2. Same 8 points, completely different meaning. So you always judge the effect against the background variation.

![Left: drug and placebo bell curves 8 points apart with SD 25 almost completely overlap (d = 0.32). Right: the same 8-point gap with SD 2 gives two separate peaks (d = 4).](figures/effectsize-two-worlds.svg)

In the figure, the gap between the peaks is the same 8 points in both panels. On the left you could not tell a drug patient from a placebo patient; on the right you could tell them apart almost every time.

## Cohen's d

$$
d = \frac{\text{gap between the two group means}}{\text{SD of individuals}}
$$

Read it as: **how many person-to-person spreads wide is my effect.** It has no units, so you can compare it across any measurement.

The bottom is the **pooled SD**: one combined spread for both groups. With equal group sizes it's simply the square root of the average of the two groups' variances.

> **Trap:** divide by the SD of individuals, not the standard error. d must not shrink when the sample grows.

### Worked example

Let's say we have 5 patients on the drug and 5 on placebo, and we record how much each person's blood pressure dropped.

| | Drops | Mean | Variance |
|---|---|---|---|
| Drug | 1, 23, 7, 15, 9 | 11.0 | 70.0 |
| Placebo | −2, 13, −7, 8, 3 | 3.0 | 62.5 |

| Step | Working | Result |
|---|---|---|
| 1. Gap | 11.0 − 3.0 | 8.0 |
| 2. Average variance | (70.0 + 62.5) / 2 | 66.25 |
| 3. Pooled SD | √66.25 | 8.14 |
| 4. Cohen's d | 8.0 / 8.14 | **d = 0.98** |

A large effect: the drug group's mean sits about one person-to-person spread above placebo. Yet a t-test on these 10 people gives p = 0.16, not significant. Big effect, too little data to be sure. That's why you report both.

```python
import numpy as np
from scipy import stats

drug    = np.array([1, 23, 7, 15, 9])     # BP drop per patient on the drug
placebo = np.array([-2, 13, -7, 8, 3])    # BP drop per patient on placebo

gap = drug.mean() - placebo.mean()                          # 8.0
pooled_sd = np.sqrt((drug.var(ddof=1) + placebo.var(ddof=1)) / 2)  # 8.14 (equal group sizes)
print(round(gap / pooled_sd, 2))                            # d = 0.98
print(round(stats.ttest_ind(drug, placebo).pvalue, 2))      # p = 0.16
```

### The bands

| d | Band |
|---|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

These are rough rules of thumb: in some fields a d of 0.2 is a big deal.

![Three pairs of bell curves at d = 0.2, 0.5 and 0.8, with shared area shaded grey: 92%, 80% and 69% overlap.](figures/effectsize-d-bands.svg)

In the figure, the grey shared area shrinks as d grows: 92% overlap at d = 0.2, 69% at 0.8. Even a "large" effect leaves most of the two groups overlapping.

## The effect-size family

| Flavour | Question | Measures |
|---|---|---|
| **Differences** | How far apart are two groups? | Cohen's d, risk ratio |
| **Associations** | How strongly do two things move together? | r, r² (see [Correlation](#/t/correlation)) |

**Risk ratio**: risk in one group divided by risk in the other (6% vs 3% → 2.0, "twice the risk").

## The anchor

> **Key idea:** The p-value tells you whether the effect is there. The effect size tells you how big it is.

A tiny p with a huge sample can still sit on a trivial effect.

![Typical p-value against sample size for a trivial true effect of d = 0.05: around 0.8 at n = 50, crossing 0.05 near n = 3,000 and reaching 0.0004 at n = 10,000.](figures/effectsize-p-vs-n.svg)

In the figure, the effect never changes (d stays at a trivial 0.05), but the p-value keeps falling as the sample grows, sliding under the dashed 0.05 line past about 3,000 per group. Significance tracks sample size; effect size doesn't.

## Saying it in words

> **Say it like this:** "The drug lowered blood pressure by 8 points more than placebo, a large effect (Cohen's d = 0.98), though with only 5 patients per group the difference was not statistically significant (p = 0.16)."

## Traps

| Wrong | Right |
|---|---|
| "p < 0.001, so the effect is huge" | p measures evidence; report d for size |
| Dividing the gap by the standard error | Divide by the SD of individuals |
| Treating 0.2 / 0.5 / 0.8 as fixed rules | They're rough guides; context decides |
| Not significant, so no effect | A big d with a small sample can miss significance; see [Power & sample size](#/t/power) |

## Quick reference

| Term | One line |
|---|---|
| Effect size | How big the effect is; doesn't grow with n |
| Cohen's d | Gap between means ÷ pooled SD |
| Pooled SD | Combined spread of both groups |
| Bands | 0.2 small, 0.5 medium, 0.8 large |
| Anchor | p: is it there? Effect size: how big? |

**Order of calculation:** group means → gap → pooled SD → d → band → sentence.
