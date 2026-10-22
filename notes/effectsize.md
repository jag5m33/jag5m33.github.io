# Effect size

An effect size says how big an effect is, in a number that does not grow just because you collected more data.

## 1. The plainest version

**Effect size** — how big the effect actually is, in a number that does **not** grow with your sample size. That last part is the whole point: a p-value tells you whether an effect is there; an effect size tells you how big it is.

The raw difference is already an effect size. A drug that lowers blood pressure by 8 points — that 8 is an effect size. Simple.

## 2. Why divide by the spread

A raw difference on its own can't be compared across different measurements (8 mmHg vs 8 exam marks vs 8 seconds). So the standard move is to measure the effect against the natural person-to-person spread.

**Two worlds.** An 8-point shift is trivial if people naturally vary by 20–30 points, but enormous if they vary by 2. Same 8 points, completely different meaning. So you always judge the effect against the background variation.

![Left: drug and placebo bell curves 8 points apart with SD 25 almost completely overlap (d = 0.32). Right: the same 8-point gap with SD 2 gives two separate peaks (d = 4).](figures/effectsize-two-worlds.svg)

In the figure, the gap between the peaks is the same 8 points in both panels. On the left you could not tell a drug patient from a placebo patient; on the right you could tell them apart almost every time.

## 3. Cohen's d

$$
d = \frac{\bar{x}_1 - \bar{x}_2}{s_p} = \frac{\text{gap between the two means}}{\text{standard deviation of individuals}}
$$

Read it as: **how many person-to-person spreads wide is my effect.** Unit-free, so it's comparable across any measurement at all.

Gap of 8, individual spread of 8 → d = 1. Gap of 8, spread of 16 → d = 0.5.

### The denominator: pooled standard deviation

**Pooled SD (s_p)** — one combined estimate of the spread within the groups, weighting each group's variance by its degrees of freedom (n − 1):

$$
s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}
$$

Where it comes from: each group gives an estimate of the same underlying variance. Averaging the **variances** (weighted by how much information each has) and then taking the square root gives the best single estimate. With equal group sizes it is just the square root of the average of the two variances.

> **Trap:** The denominator is the pooled **standard deviation**, not the variance. And it's the SD of individuals, not the standard error — d must not shrink when n grows.

### Worked example

Let's say we have 5 patients on the drug and 5 on placebo, and we record how much each person's blood pressure dropped.

| | Drops | Mean | Variance s² |
|---|---|---|---|
| Drug | 1, 23, 7, 15, 9 | 11.0 | 70.0 |
| Placebo | −2, 13, −7, 8, 3 | 3.0 | 62.5 |

| Step | Working | Result |
|---|---|---|
| 1. Gap | 11.0 − 3.0 | 8.0 |
| 2. Pooled variance | (4 × 70.0 + 4 × 62.5) / (5 + 5 − 2) = 530 / 8 | 66.25 |
| 3. Pooled SD | √66.25 | 8.14 |
| 4. Cohen's d | 8.0 / 8.14 | **d = 0.98** |

A large effect: the drug group's mean sits about one person-to-person spread above placebo. Yet a t-test on these 10 people gives p = 0.16 — not significant. Big effect, too little data to be sure. That is exactly why you report both.

### In code

```python
import numpy as np
from scipy import stats

drug    = np.array([1, 23, 7, 15, 9])     # BP drop per patient on the drug
placebo = np.array([-2, 13, -7, 8, 3])    # BP drop per patient on placebo

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)            # sample variances (divide by n-1)
    sp = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)) # pooled SD
    return (a.mean() - b.mean()) / sp                # gap in units of spread

print(round(cohens_d(drug, placebo), 2))             # 0.98
print(round(stats.ttest_ind(drug, placebo).pvalue, 2))  # 0.16 -- big effect, small n
```

`ddof=1` makes numpy divide by n − 1 (the sample variance); its default divides by n.

### The bands

| d | Band |
|---|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

These are Cohen's rough rules of thumb, not laws — in some fields a d of 0.2 is a big deal.

## 4. The picture

Draw the two groups as two bell curves.

- **d = 0.2** — the bells almost entirely overlap. You can't tell a treated person from a placebo person by looking.
- **d = 0.8** — the bells are visibly separated. The effect stands clearly above the noise.

![Three pairs of bell curves at d = 0.2, 0.5 and 0.8, with shared area shaded grey: 92%, 80% and 69% overlap.](figures/effectsize-d-bands.svg)

In the figure, the grey shared area shrinks as d grows: 92% overlap at d = 0.2, 80% at 0.5, 69% at 0.8. Even a "large" effect leaves most of the two groups overlapping.

> **Key idea:** The bigger the spread between the two distributions, the more easily you can tell them apart — so the more apparent the effect.

## 5. The effect-size family

Two flavours, depending on what you're measuring.

| Flavour | Question | Measures |
|---|---|---|
| **Differences** | How far apart are two groups? | Cohen's d, risk ratio, odds ratio |
| **Associations** | How strongly do two things move together? | r, r² (see [Correlation](#/t/correlation)) |

- **Risk ratio** — risk in one group divided by risk in the other (e.g. 6% vs 3% → 2.0).
- **Odds ratio** — the same idea using odds, p / (1 − p), instead of probabilities; it is what [Logistic regression](#/t/logreg) coefficients turn into.

## 6. The anchor

> **Key idea:** The p-value tells you whether the effect is there. The effect size tells you how big it is.

A tiny p with a huge n can still sit on a trivial effect — which is exactly why you always report both.

![Typical p-value against sample size for a trivial true effect of d = 0.05: around 0.8 at n = 50, crossing 0.05 near n = 3,000 and reaching 0.0004 at n = 10,000.](figures/effectsize-p-vs-n.svg)

In the figure, the effect never changes — d stays at a trivial 0.05 — but the p-value keeps falling as n grows, sliding under the dashed α line somewhere past n ≈ 3,000 per group. Significance tracks sample size; effect size doesn't.

## 7. Saying it in words

> **Say it like this:** "The drug lowered blood pressure by 8 points more than placebo, a large effect (Cohen's d = 0.98), though with only 5 patients per group the difference was not statistically significant (p = 0.16)."

## 8. Traps

| Wrong | Right |
|---|---|
| "p < 0.001, so the effect is huge" | p measures evidence; report d for size |
| Dividing the gap by the standard error | Divide by the SD of individuals (pooled) |
| Dividing by the pooled variance | Take the square root first |
| Treating 0.2 / 0.5 / 0.8 as fixed rules | They're rough guides; context decides what matters |
| Not significant, so no effect | A big d with small n can miss significance — see [Power & sample size](#/t/power) |

## Quick reference

| Term | One line |
|---|---|
| Effect size | How big the effect is; doesn't grow with n |
| Raw difference | The simplest effect size, in original units |
| Cohen's d | Gap between means ÷ pooled SD |
| Pooled SD | √( ((n₁−1)s₁² + (n₂−1)s₂²) / (n₁+n₂−2) ) |
| Bands | 0.2 small, 0.5 medium, 0.8 large |
| Family | Differences (d, risk ratio, odds ratio) vs associations (r, r²) |
| Anchor | p: is it there? Effect size: how big? |

**The order of calculation:** group means → gap → each group's variance → pooled variance → pooled SD → d → band → sentence.
