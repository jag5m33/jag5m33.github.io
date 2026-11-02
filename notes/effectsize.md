# Effect size

An effect size answers the question a p-value can't: **how big is the effect, really?**

**On this page**

- Effect size: size, not evidence
- Why you divide by the spread
- Cohen's d, worked through
- Small, medium, large: what the bands look like
- The effect-size family
- Why you always report both p and size

## Effect size

> **Definition:** an **effect size** is how big the effect actually is, in a number that does **not** grow with your sample size.

That last part is the whole point. A p-value tells you whether an effect is there; an effect size tells you how big it is.

The raw difference is already an effect size. A drug lowers blood pressure by 8 points more than placebo: that 8 is an effect size. Simple.

## Why divide by the spread

> **Definition:** a **standardised** effect size measures the gap against the natural **person-to-person spread**, so it has no units.

A raw 8 can't be compared across measurements (8 mmHg, 8 exam marks, 8 seconds). And even on one scale it can mislead.

**Two worlds.** An 8-point shift is trivial if people naturally vary by 20 to 30 points, but enormous if they vary by 2. Same 8 points, completely different meaning. So you always judge the effect against the background variation.

![The same 8-point gap in both panels. Left: people vary a lot (SD 25), the curves almost completely overlap and you couldn't tell a drug patient from a placebo one (d = 0.32). Right: people vary by 2, the peaks separate (d = 4).](figures/effectsize-two-worlds.svg)

## Cohen's d

> **Definition:** **Cohen's d** is the gap between two group means divided by the spread of individuals. Read it as: **how many person-to-person spreads wide is my effect.**

$$
d = \frac{\text{gap between the means}}{\text{spread (SD) of individuals}}
$$

The gap is one group's mean minus the other's. The spread is the **pooled SD**: one combined SD for both groups.

Gap of 8, spread of 8 → d = 1. Gap of 8, spread of 16 → d = 0.5.

**Worked example.** 5 patients on the drug, 5 on placebo. How much did each person's blood pressure drop?

| | Drops | Mean | SD |
|---|---|---|---|
| Drug | 1, 23, 7, 15, 9 | 11.0 | 8.4 |
| Placebo | −2, 13, −7, 8, 3 | 3.0 | 7.9 |

1. Gap: 11.0 − 3.0 = **8.0**
2. Pooled SD (both groups combined): **8.14**
3. d = 8.0 / 8.14 = **0.98**, a large effect.

Yet a t-test on these 10 people gives **p = 0.16**: not significant. Big effect, too little data to be sure.

```python
import numpy as np
drug    = np.array([1, 23, 7, 15, 9])     # BP drop on the drug
placebo = np.array([-2, 13, -7, 8, 3])    # BP drop on placebo
gap = drug.mean() - placebo.mean()                                 # 8.0
pooled_sd = np.sqrt((drug.var(ddof=1) + placebo.var(ddof=1)) / 2)  # 8.14
print(round(gap / pooled_sd, 2))                                   # d = 0.98
```

> **Where it goes wrong:** divide by the **SD of individuals**, not the standard error. The SE shrinks as the sample grows, so d would grow with n, which is exactly what an effect size must not do. And it's the SD, not the variance.

## The bands

> **Definition:** the rough **bands** for d are **0.2 small, 0.5 medium, 0.8 large**.

The picture: the bigger the spread between the two distributions, the more easily you can tell them apart, so the more apparent the effect.

- d = 0.2: the bells almost entirely overlap. You can't tell a treated person from a placebo person by looking.
- d = 0.8: the bells are visibly separated. The effect stands clearly above the noise.

![Three pairs of bell curves at d = 0.2, 0.5 and 0.8. The grey shared area shrinks from 92% to 80% to 69%: even a "large" effect leaves most of the two groups overlapping.](figures/effectsize-d-bands.svg)

> **Where it goes wrong:** the bands are rules of thumb, not laws. In some fields a d of 0.2 is a big deal.

## The effect-size family

> **Definition:** effect sizes come in two flavours: **differences** (how far apart are two groups?) and **associations** (how strongly do two things move together?).

| Flavour | Examples |
|---|---|
| Differences | Cohen's d; risk ratio (6% vs 3% risk → 2.0, "twice the risk") |
| Associations | r and r², see [Correlation](#/t/correlation) |

## Report both

> **Definition:** the anchor: **the p-value tells you whether the effect is there. The effect size tells you how big it is.**

A tiny p with a huge sample can still sit on a trivial effect. That's exactly why you always report both.

![The effect never changes (a trivial d = 0.05), but the p-value keeps falling as the sample grows, sliding under the dashed 0.05 line past about 3,000 per group. Significance tracks sample size; effect size doesn't.](figures/effectsize-p-vs-n.svg)

> **Where it goes wrong:** "p < 0.001, so the effect is huge" is wrong: p measures evidence, d measures size. The reverse is wrong too: "not significant, so no effect". The drug example has a big d and p = 0.16, the sample was just too small. That's [Power & sample size](#/t/power).

> **Say it like this:** "The drug lowered blood pressure by 8 points more than placebo, a large effect (Cohen's d = 0.98), though with only 5 patients per group the difference wasn't statistically significant (p = 0.16)."

## Quick reference

| Term | In one line |
|---|---|
| Effect size | How big the effect is. Doesn't grow with n |
| Cohen's d | Gap ÷ spread: how many person-to-person spreads wide the effect is |
| Pooled SD | One combined spread for both groups |
| Bands | 0.2 small, 0.5 medium, 0.8 large (rough) |
| Anchor | p: is it there? Effect size: how big? |

**Order:** group means → gap → pooled SD → d → band → sentence.
