# Hypothesis testing

A hypothesis test asks one question of your data: **does a result like this turn up easily by chance, or not?**

**On this page**

- The two hypotheses: what you assume, and what you're looking for
- α: the line you draw before you start
- The p-value: how surprising your result is
- The two mistakes you can make
- The shortcut through confidence intervals
- Which test to use, and how to say the result

## The two hypotheses

> **Definition:** the **null hypothesis (H₀)** says "nothing is going on": no effect, no difference. The **alternative (H₁)** is the thing you're testing for.

You start by assuming H₀ is true, then ask whether your data makes that hard to believe.

Example: is this coin fair?

- H₀: it's a fair coin, heads = 50%.
- H₁: heads ≠ 50%.

> **Where it goes wrong:** you never *prove* H₀. You either **reject** it or **fail to reject** it. "Not enough evidence of an effect" is not the same as "no effect".

## α (the significance level)

> **Definition:** **α** is the line you draw **before** testing: how big a risk of a false alarm you'll accept. Usually 0.05 (5%).

It's the same α as in [Confidence intervals](#/t/ci): 95% confidence and α = 0.05 are the same choice, seen from two sides.

## The p-value

> **Definition:** the **p-value** is how easily a result at least this extreme would turn up by chance alone, **if H₀ were true**.

- **Small p:** a result this size almost never turns up by chance, so chance is a poor explanation. Reject H₀, call it **significant**.
- **Large p:** a result this size turns up easily by chance, so you can't rule chance out. Keep H₀.

The rule: **p ≤ α → reject H₀. p > α → fail to reject H₀.**

![The "no effect" world as a bell curve. Your result sits far out at z = 2.83; the two red tails (everything at least that extreme) add up to p = 0.0047.](figures/hypothesis-pvalue.svg)

**Worked example.** 120 out of 200 people recycle (60%). H₀ says 50%. Is 60% just luck?

1. Work out how far 60% sits from 50%, in standard errors: **z = 2.83** (the tool does this for you).
2. The chance of landing that far out by luck: **p = 0.0047**.
3. 0.0047 is below 0.05 → **reject H₀**. 60% is very unlikely to be luck.

> **Where it goes wrong:** the p-value is **not** "the chance H₀ is true". It *assumes* H₀ is true, so it can't measure that. And it's **not** how big the effect is: a huge sample can make a tiny, useless difference significant. For size, see [Effect size](#/t/effectsize).

## The two mistakes

> **Definition:** a **Type I error** is a false positive: you claim an effect that isn't there. A **Type II error** is a false negative: you miss an effect that is there.

| | Really no effect | Really an effect |
|---|---|---|
| **You reject H₀** | **Type I** (false alarm), chance = α | Correct: this is **power** |
| **You keep H₀** | Correct | **Type II** (miss), chance = β |

- Type I: a healthy patient told they're ill.
- Type II: a sick patient told they're fine.
- **Power = 1 − β**: the chance of catching a real effect. Aim for 80%. More in [Power & sample size](#/t/power).

![Two worlds side by side: "no effect" (left) and "real effect" (right). Right of the dashed cutoff, the red sliver of the left curve is α and the teal part of the right curve is power; the gold part of the right curve, left of the cutoff, is β.](figures/hypothesis-errors.svg)

Move the cutoff right and α shrinks but β grows. **They trade off. Only a bigger sample shrinks both.**

## The shortcut through confidence intervals

> **Definition:** if the H₀ value sits **outside** the confidence interval, reject H₀.

Recycling: the 99% confidence interval is 51% to 69%. 50% isn't in it → reject H₀ at α = 0.01. Same answer as the p-value.

## Which test?

| You want to compare | Test | scipy |
|---|---|---|
| One mean vs a fixed number | One-sample t | `stats.ttest_1samp` |
| Two separate groups | Two-sample t (Welch) | `stats.ttest_ind(a, b, equal_var=False)` |
| Same people, before vs after | Paired t | `stats.ttest_rel` |
| Two categorical columns | Chi-square | `stats.chi2_contingency` |
| Three or more groups | ANOVA | `stats.f_oneway` |

```python
from scipy import stats

alpha = 0.05                                        # decided BEFORE looking
t, p = stats.ttest_ind(group_a, group_b, equal_var=False)
print("reject H0" if p <= alpha else "fail to reject H0")
```

## Saying it in words

> **Say it like this:** "p = 0.0047 is below 0.05, so we reject the null. There's significant evidence that the share of people who recycle isn't 50%."

> **Where it goes wrong:** never say "we accept the null" or "we proved there's no effect". Absence of evidence isn't evidence of absence. The test may just have been too small to see it.

## Quick reference

| Term | In one line |
|---|---|
| H₀ | "Nothing is going on." Assumed true |
| α | False-alarm risk you accept. Set before testing |
| p-value | How easily a result this extreme turns up by chance, if H₀ is true |
| Significant | p ≤ α |
| Type I / Type II | False alarm (α) / miss (β) |
| Power | 1 − β: chance of catching a real effect |

**Order:** H₀ and H₁ → pick α → pick the test → p-value → compare with α → say it in words.
