# Hypothesis testing

A hypothesis test asks one question of your data: does a result like this turn up easily by chance, or not?

## 1. The two hypotheses

| | Meaning | Example |
|---|---|---|
| **Null (H₀)** | No effect, no difference. The default you assume true | p = 0.5 |
| **Alternative (H₁ or Hₐ)** | What you're testing for | p ≠ 0.5 (two-sided), p > 0.5 (one-sided) |

- **Two-sided** — you care about a difference in either direction. The default.
- **One-sided** — you only care about one direction, decided **before** you look at the data.

> **Key idea:** You never prove H₀. You either reject it or fail to reject it.

## 2. Test statistic

**Test statistic** — one number that says how far your sample result sits from what H₀ predicts, measured in standard errors.

$$
\text{test statistic} = \frac{\text{estimate} - \text{null value}}{SE}
$$

It is the same "distance in standard errors" idea as a z-score in [Distributions & the CLT](#/t/distributions). Because of the Central Limit Theorem, under H₀ this number follows a known curve (Z, t, χ², F), so you can ask how unusual it is.

**Worked example.** 120 of 200 people recycle (p̂ = 0.60). H₀: p = 0.5.

| Step | Working | Result |
|---|---|---|
| 1. SE under H₀ | √(0.5 × 0.5 / 200) — use the **null** value, because you're assuming H₀ | 0.0354 |
| 2. Test statistic | (0.60 − 0.50) / 0.0354 | z = 2.83 |
| 3. p-value (two-sided) | 2 × area beyond 2.83 | p = 0.0047 |
| 4. Decide at α = 0.05 | 0.0047 ≤ 0.05 | Reject H₀ |

## 3. Significance level (α)

**α** — the threshold you set **before** testing. The same α as in [Confidence intervals](#/t/ci): a 95% confidence level and α = 0.05 are the same choice.

It is the risk of a false positive you're willing to accept.

## 4. The p-value

**p-value** — the probability of seeing a result at least as extreme as yours **if H₀ were true**.

$$
p = P(\text{data this extreme} \mid H_0 \text{ true})
$$

- **What it is** — how easily a result this extreme turns up by chance alone, if there's no real effect.
- **Small p** — a result this size almost never turns up by chance, so chance is a poor explanation. Reject the null, call it significant.
- **Large p** — a result this size turns up easily by chance, so you can't rule chance out. Keep the null — not proof there's no effect, just too little evidence against it.

$$
p \le \alpha \;\Rightarrow\; \text{reject } H_0 \qquad\qquad p > \alpha \;\Rightarrow\; \text{fail to reject } H_0
$$

![Standard normal curve under H0 with the observed z = 2.83 marked; the two red tails beyond plus and minus 2.83 add up to p = 0.0047.](figures/hypothesis-pvalue.svg)

In the figure, the curve is the H₀ world. Your result sits at z = 2.83. The red tails are every result **at least that extreme** in either direction — their total area is the p-value. They are tiny, so chance is a poor explanation.

**What it is not:**

| Wrong reading | Correct |
|---|---|
| The probability H₀ is true | It assumes H₀ is true, so it can't measure that |
| The probability your result was a fluke | It's the chance of this data given H₀ |
| A measure of effect size | A tiny p can come from a trivial effect with big n — see [Effect size](#/t/effectsize) |

## 5. The two errors

| | H₀ actually true | H₀ actually false |
|---|---|---|
| **Reject H₀** | **Type I — false positive.** Probability = α | Correct. Probability = power |
| **Fail to reject H₀** | Correct. Probability = 1 − α | **Type II — false negative.** Probability = β |

- **Type I error (α)** — you claim an effect that isn't there. A healthy patient told they're ill.
- **Type II error (β)** — you miss an effect that is there. A sick patient told they're fine.
- **β (beta)** — the probability of a Type II error, for a given true effect size.
- **Power = 1 − β** — the chance of detecting a real effect. 80% is the usual target. Full treatment in [Power & sample size](#/t/power).

**Medical example.** A screening test for a disease. H₀: "the patient is healthy".

| | Patient really healthy | Patient really ill |
|---|---|---|
| Test says ill | Type I: needless worry, more tests | Correct: caught it (power) |
| Test says healthy | Correct | Type II: missed disease, no treatment |

Which error is worse depends on the context — a missed cancer is usually worse than a false alarm, so screening tests accept more Type I errors to cut Type II.

![Two bell curves: H0 (no effect) and H1 (real effect). Right of the dashed cutoff, the red sliver of H0 is alpha and the teal part of H1 is power; left of the cutoff, the gold part of H1 is beta.](figures/hypothesis-errors.svg)

In the figure, the left curve is the world where H₀ is true, the right curve where H₁ is true. The dashed line is your cutoff. The slice of the H₀ curve to its right is α (red). The slice of the H₁ curve to its left is β (gold). Move the line right and α shrinks while β grows — **the two trade off directly, and only a larger n shrinks both.**

## 6. The link to confidence intervals

A two-sided test at α and a confidence interval at (1 − α) give the same verdict:

> **Key idea:** If the null value falls **outside** the interval, reject H₀ at that α.

Recycling example: the 99% CI is (0.511, 0.689). It excludes 0.5, so H₀: p = 0.5 is rejected at α = 0.01. The 95% CI, (0.532, 0.668), also excludes 0.5, so it is rejected at α = 0.05 too. That matches p = 0.0047, which is below both.

(The test uses the null value in its SE and the CI uses p̂, so in borderline cases the two can disagree slightly for proportions; for means they match exactly.)

## 7. Which test?

| Test | Use it for | Example question | scipy one-liner |
|---|---|---|---|
| One-sample t | One mean vs a fixed value | Is mean delivery time 5 days? | `stats.ttest_1samp(a, popmean=5)` |
| Two-sample t (independent) | Two separate groups' means | Do group A and B differ? | `stats.ttest_ind(a, b, equal_var=False)` |
| Paired t | Same subjects measured twice | Did each patient's score change? | `stats.ttest_rel(before, after)` |
| z-test (proportion) | A proportion vs a value, large n | Is the recycling rate 50%? | `proportions_ztest(120, 200, value=0.5)` (statsmodels) |
| Chi-square independence | Two categorical variables | Is churn related to plan type? | `stats.chi2_contingency(table)` |
| Chi-square goodness of fit | Observed counts vs expected | Are the four outcomes equally likely? | `stats.chisquare(observed, f_exp=expected)` |
| One-way ANOVA | Three or more group means | Do three teaching methods differ? | `stats.f_oneway(a, b, c)` |

- `equal_var=False` gives **Welch's t-test**, which does not assume the two groups have the same spread. It is the safer default.
- A significant ANOVA tells you *some* group differs, not *which* — follow up with pairwise tests (and correct for multiple testing).
- t-tests assume roughly normal data or n ≥ 30 per group; chi-square needs expected counts of at least 5 per cell.

```python
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# z-test for the recycling example: 120 of 200, H0: p = 0.5
z, p = proportions_ztest(count=120, nobs=200, value=0.5,
                         prop_var=0.5)    # use the null p in the SE, as in the hand calculation
print(round(z, 2), round(p, 4))           # 2.83 0.0047

# two-sample t-test on simulated data
rng = np.random.default_rng(0)            # seeded so results repeat
a = rng.normal(5.0, 1, 20)                # group A: mean 5
b = rng.normal(5.5, 1, 20)                # group B: mean 5.5
t, p = stats.ttest_ind(a, b, equal_var=False)   # Welch's t-test
alpha = 0.05                              # chosen BEFORE looking
print("reject H0" if p <= alpha else "fail to reject H0")

# chi-square test of independence on a 2x2 table of counts
table = np.array([[30, 10],               # rows: plan A / plan B
                  [20, 40]])              # cols: churned / stayed
chi2, p, dof, expected = stats.chi2_contingency(table)
```

## 8. Saying it in words

| Outcome | Sentence |
|---|---|
| p = 0.003, α = 0.05 | "p = 0.003 is below 0.05, so we reject the null. There is significant evidence that [effect] exists." |
| p = 0.21, α = 0.05 | "p = 0.21 exceeds 0.05, so we fail to reject the null. There is insufficient evidence of [effect]." |

> **Say it like this:** "With p = 0.0047, below α = 0.05, we reject the null. There is significant evidence that the proportion of residents who recycle differs from 50%."

> **Trap:** Never write "we accept the null" or "we proved there is no effect." Absence of evidence is not evidence of absence — the test may simply have had too little power.

## 9. Traps

| Wrong | Right |
|---|---|
| "We accept H₀" | "We fail to reject H₀" |
| "p = 0.03 means a 3% chance H₀ is true" | p assumes H₀ is true; it is P(data this extreme given H₀) |
| "Significant, so the effect is big" | Significance is about evidence, not size. Report the effect size |
| Choosing α or one-sided after seeing the data | Fix α and direction before testing |
| Running many tests and reporting the one that crossed | That is p-hacking; see [Power & sample size](#/t/power) |
| Using the SE from p̂ in a proportion z-test | Under H₀, use the null value: √(p₀(1−p₀)/n) |

## Quick reference

| Term | One line |
|---|---|
| H₀ | No effect. Assumed true |
| H₁ | The effect you're testing for |
| Test statistic | (estimate − null value) / SE |
| α | False positive rate you accept. Set before testing |
| p-value | P(data this extreme given H₀ true) |
| Significant | p ≤ α |
| Type I error | False positive. Probability α |
| Type II error | False negative. Probability β |
| Power | 1 − β. Chance of catching a real effect |
| CI link | Null value outside the (1 − α) CI means reject at α |

**The order of calculation:** state H₀ and H₁ → choose α → pick the test → check assumptions → test statistic → p-value → compare with α → sentence in context.
