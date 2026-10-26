# Hypothesis testing

A hypothesis test asks one question of your data: does a result like this turn up easily by chance, or not?

## The two hypotheses

| | Meaning | Example |
|---|---|---|
| **Null (H₀)** | No effect, no difference. The default you assume true | p = 0.5 |
| **Alternative (H₁)** | What you're testing for | p ≠ 0.5 |

- **Two-sided**: you care about a difference in either direction. The default.
- **One-sided**: you only care about one direction, decided **before** you look at the data.

> **Key idea:** You never prove H₀. You either reject it or fail to reject it.

## Test statistic

**Test statistic**: one number that says how far your sample result sits from what H₀ predicts, measured in standard errors.

$$
\text{test statistic} = \frac{\text{estimate} - \text{null value}}{SE}
$$

- estimate: what your sample gave. null value: what H₀ says. SE: the [standard error](#/t/distributions), how much the estimate wobbles between samples.

It's the same "distance in standard errors" idea as a z-score.

**Worked example.** 120 of 200 people recycle ($\hat{p}$ = 0.60). H₀: p = 0.5.

| Step | Working | Result |
|---|---|---|
| 1. SE under H₀ | √(0.5 × 0.5 / 200), using the **null** value because you're assuming H₀ | 0.0354 |
| 2. Test statistic | (0.60 − 0.50) / 0.0354 | z = 2.83 |
| 3. p-value (two-sided) | area beyond ±2.83 | p = 0.0047 |
| 4. Decide at α = 0.05 | 0.0047 ≤ 0.05 | Reject H₀ |

## Significance level (α)

**α**: the threshold you set **before** testing, usually 0.05. It's the risk of a false positive you're willing to accept. It is the same α as in [Confidence intervals](#/t/ci): 95% confidence and α = 0.05 are the same choice.

## The p-value

**p-value**: the probability of seeing a result at least as extreme as yours **if H₀ were true**.

- **Small p**: a result this size almost never turns up by chance, so chance is a poor explanation. Reject the null, call it significant.
- **Large p**: a result this size turns up easily by chance, so you can't rule chance out. Keep the null: not proof there's no effect, just too little evidence against it.

The rule: $p \le \alpha$ → reject H₀. $p > \alpha$ → fail to reject H₀.

![Standard normal curve under H0 with the observed z = 2.83 marked; the two red tails beyond plus and minus 2.83 add up to p = 0.0047.](figures/hypothesis-pvalue.svg)

In the figure, the curve is the "no effect" world. Your result sits at z = 2.83. The red tails are every result **at least that extreme** in either direction; their total area is the p-value. They are tiny, so chance is a poor explanation.

| Wrong reading | Correct |
|---|---|
| The probability H₀ is true | It assumes H₀ is true, so it can't measure that |
| A measure of how big the effect is | A tiny p can come from a trivial effect with big n; see [Effect size](#/t/effectsize) |

## The two errors

| | H₀ actually true | H₀ actually false |
|---|---|---|
| **Reject H₀** | **Type I: false positive.** Chance = α | Correct (power) |
| **Fail to reject H₀** | Correct | **Type II: false negative.** Chance = β |

- **Type I**: you claim an effect that isn't there. A healthy patient told they're ill.
- **Type II**: you miss an effect that is there. A sick patient told they're fine.
- **Power = 1 − β**: the chance of catching a real effect. 80% is the usual target. More in [Power & sample size](#/t/power).

Which error is worse depends on context. A missed cancer is usually worse than a false alarm, so screening tests accept more false alarms.

![Two bell curves: H0 (no effect) and H1 (real effect). Right of the dashed cutoff, the red sliver of H0 is alpha and the teal part of H1 is power; left of the cutoff, the gold part of H1 is beta.](figures/hypothesis-errors.svg)

In the figure, the left curve is the "no effect" world, the right curve the "real effect" world. The dashed line is your cutoff. Move it right and α (red) shrinks while β (gold) grows: **the two trade off, and only a larger sample shrinks both.**

## The link to confidence intervals

> **Key idea:** If the null value falls **outside** the confidence interval, reject H₀.

Recycling example: the 99% CI is (0.511, 0.689). It doesn't contain 0.5, so H₀ is rejected at α = 0.01, matching p = 0.0047.

## Which test?

| Test | Use it for | scipy |
|---|---|---|
| One-sample t | One mean vs a fixed value | `stats.ttest_1samp(a, popmean=5)` |
| Two-sample t | Two separate groups' means | `stats.ttest_ind(a, b, equal_var=False)` |
| Paired t | Same subjects measured twice | `stats.ttest_rel(before, after)` |
| z-test (proportion) | A proportion vs a value | `proportions_ztest(...)` (statsmodels) |
| Chi-square | Are two categorical columns related? | `stats.chi2_contingency(table)` |
| ANOVA | Three or more group means | `stats.f_oneway(a, b, c)` |

`equal_var=False` gives **Welch's t-test**, which doesn't assume the two groups have the same spread. It's the safer default.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)            # seeded so results repeat
a = rng.normal(5.0, 1, 20)                # group A: mean 5
b = rng.normal(5.5, 1, 20)                # group B: mean 5.5

alpha = 0.05                              # chosen BEFORE looking
t, p = stats.ttest_ind(a, b, equal_var=False)   # Welch's t-test
print("reject H0" if p <= alpha else "fail to reject H0")
```

## Saying it in words

> **Say it like this:** "With p = 0.0047, below α = 0.05, we reject the null. There is significant evidence that the proportion of residents who recycle differs from 50%."

> **Trap:** Never write "we accept the null" or "we proved there is no effect." Absence of evidence is not evidence of absence: the test may simply have had too little power.

## Traps

| Wrong | Right |
|---|---|
| "We accept H₀" | "We fail to reject H₀" |
| "p = 0.03 means a 3% chance H₀ is true" | p assumes H₀ is true |
| "Significant, so the effect is big" | Significance is about evidence, not size. Report the effect size |
| Choosing α or one-sided after seeing the data | Fix them before testing |
| Running many tests and reporting the one that crossed | That's p-hacking; see [Power & sample size](#/t/power) |

## Quick reference

| Term | One line |
|---|---|
| H₀ | No effect. Assumed true |
| Test statistic | (estimate − null value) / SE |
| α | False positive rate you accept. Set before testing |
| p-value | Chance of data this extreme if H₀ is true |
| Significant | p ≤ α |
| Type I / Type II | False positive (α) / false negative (β) |
| Power | 1 − β. Chance of catching a real effect |
| CI link | Null value outside the CI → reject |

**Order of calculation:** H₀ and H₁ → choose α → pick the test → test statistic → p-value → compare with α → sentence in context.
