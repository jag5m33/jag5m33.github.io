# Correlation

Correlation puts one number on how strongly two variables move together, and in which direction.

## 1. Pearson vs Spearman

Both measure a **linear-ish** association. This is the correction to watch: it is not "Pearson linear, Spearman non-linear."

| | Works on | Catches |
|---|---|---|
| **Pearson** | The raw values | Straight-line relationships |
| **Spearman** | The **ranks** of the values | Any consistently-increasing (or decreasing) relationship, even if it curves; robust to outliers |

- **Monotonic** — always moving in the same direction (never turning back), though not necessarily in a straight line.
- **Rank** — a value's position when sorted: smallest = 1, next = 2, and so on.

> **Key idea:** Spearman is just Pearson run on the ranks instead of the raw numbers.

## 2. r — the Pearson correlation coefficient

**r** — runs from −1 to +1. It captures the **strength and direction** of a straight-line link.

| r | Meaning |
|---|---|
| +1 | Perfect upward line |
| 0 | No linear link — a shapeless cloud |
| −1 | Perfect downward line |

![Five scatter plots with correlations of about −0.9, −0.5, 0, +0.5 and +0.9.](figures/correlation-grid.svg)

In the figure, the sign sets the direction of the slope and the size sets how tightly the cloud hugs a line. Note that r is **not** the slope: a shallow line and a steep line can both have r = 0.9.

### The formula

$$
r = \frac{\text{cov}(x, y)}{s_x \, s_y}, \qquad \text{cov}(x, y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{n - 1}
$$

**Where it comes from.**

- **Covariance** — the average of (how far x is from its mean) × (how far y is from its mean). If points that are above average in x tend to be above average in y, those products are positive and add up. If they sit on opposite sides, the products are negative.
- Covariance's size depends on the units (hours × marks), so you can't judge it on its own. Dividing by both standard deviations strips the units out and squeezes the result into −1 to +1.

The n − 1 cancels top and bottom, so the hand-calculation version is:

$$
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \; \sum (y_i - \bar{y})^2}}
$$

### Worked example

Let's say we have 6 students: hours studied (x) and exam score (y).

![The six points with dotted lines at the two means; vertical lines show each point's deviation from the mean score. Five points are teal (positive product), one is red (negative).](figures/correlation-worked.svg)

In the figure, the dotted lines split the plot into four quarters. Points in the bottom-left and top-right (teal) add to the covariance; the one red point, above-average hours but below-average score, subtracts a little.

Means: x̄ = 21 / 6 = **3.5**, ȳ = 368 / 6 = **61.33**.

| x | y | x − x̄ | y − ȳ | product | (x − x̄)² | (y − ȳ)² |
|---|---|---|---|---|---|---|
| 1 | 52 | −2.5 | −9.33 | 23.33 | 6.25 | 87.11 |
| 2 | 55 | −1.5 | −6.33 | 9.50 | 2.25 | 40.11 |
| 3 | 61 | −0.5 | −0.33 | 0.17 | 0.25 | 0.11 |
| 4 | 60 | 0.5 | −1.33 | −0.67 | 0.25 | 1.78 |
| 5 | 68 | 1.5 | 6.67 | 10.00 | 2.25 | 44.44 |
| 6 | 72 | 2.5 | 10.67 | 26.67 | 6.25 | 113.78 |
| **Sum** | | 0 | 0 | **69.0** | **17.5** | **287.33** |

| Step | Working | Result |
|---|---|---|
| 1. Sum of products | from the table | 69.0 |
| 2. Spread terms | √(17.5 × 287.33) | 70.91 |
| 3. r | 69.0 / 70.91 | **r = 0.973** |

(Equivalently: covariance = 69.0 / 5 = 13.8, $s_x$ = 1.871, $s_y$ = 7.581, and 13.8 / (1.871 × 7.581) = 0.973.)

**Spearman on the same data.** Rank each column. x ranks are 1–6. The y values 52, 55, 61, 60, 68, 72 rank as 1, 2, **4, 3**, 5, 6 — only students 3 and 4 swap. Pearson on those ranks gives **ρ = 0.943**. (Shortcut when there are no ties: ρ = 1 − 6Σd² / (n(n² − 1)), with d the rank difference: 1 − 6 × 2 / 210 = 0.943.)

## 3. When to reach for Spearman

- **Monotonic but curved** — e.g. income rising with experience but levelling off.
- **Outliers** — ranks cap how far one extreme value can pull.
- **Ordinal data** — ratings like 1–5 stars, where the gaps between levels aren't equal.

![Left: 12 points on a clean line, Pearson 0.98 and Spearman 0.97. Right: one extra point typed as 60 instead of about 13 drags Pearson to 0.67 while Spearman stays at 0.97.](figures/correlation-outlier.svg)

In the figure, the right panel adds one bad reading. Pearson uses the raw value 60, so it is dragged from 0.98 to 0.67. Spearman only sees that it is the largest value at the largest x — which fits the trend — so it stays at 0.97.

**Neither catches a relationship that turns back on itself.**

![A U-shaped scatter where y = x squared plus noise: Pearson r is 0.02 and Spearman is 0.00.](figures/correlation-curve.svg)

In the figure, y depends completely on x, yet both coefficients are about zero, because the left half goes down and the right half goes up and they cancel. **r = 0 means no linear link, not no link.** Always plot the data.

## 4. r² — the one to phrase carefully

> **Key idea:** r² is the share of the variation in the outcome that the other variable explains. NOT "the share of the correlation," and NOT "how related they are as a percent."

The frame is **directional**: pick one variable as the outcome, and r² is the slice of its wobble that the other one explains — it is not a quantity floating between the pair. (The number itself is the same whichever variable you call the outcome; it's the sentence you write that has a direction.)

**Height and weight.** r = 0.7 → r² = 0.49 → height accounts for about **49% of why people's weights differ**. The other 51% is diet, genes, and noise.

**Why square it:** it turns a −1-to-1 correlation into a clean 0-to-1 proportion.

> **Trap:** r = 0.7 is NOT "70% related." And halving r quarters r² — because you're squaring it. r = 0.5 → r² = 0.25, so 25% explained, 75% down to everything else.

In the worked example, r = 0.973 → r² = 0.947: hours studied account for about 95% of the variation in these six scores (a tiny made-up sample — real data are rarely this tidy).

## 5. Adjusted R² — the noise tax

When you have several predictors, R² is the same idea from a regression model. This section is the short version; the full treatment is in [Regression metrics](#/t/regmetrics).

**Plain R² is a one-way ratchet.** Add any predictor and it rises or stays flat — never falls. A useless variable can always fit a scrap of random noise by luck, and plain R² counts that scrap as real explanation.

**Adjusted R² charges every predictor the same "noise tax".** It never inspects the features. It knows only two numbers: how much data you have (n) and how many predictors you used (p).

$$
R^2_{\text{adj}} = 1 - (1 - R^2)\,\frac{n - 1}{n - p - 1}
$$

The fraction is always above 1 and grows as p approaches n — that is the tax getting heavier.

- **A junk variable** raises plain R² by about the lucky amount → the tax cancels it → adjusted R² stays flat or dips. Correctly ignored.
- **A real variable** raises plain R² by far more than the lucky amount → adjusted R² still rises. Correctly rewarded.

> **Key idea:** Plain R² asks "how well did I fit?" Adjusted R² asks "how well did I fit, given how many chances I gave myself?"

**Demonstration (50 points, one real variable, then pure-noise columns added):**

| Junk added | Predictors | Plain R² | Adjusted R² |
|---|---|---|---|
| 0 | 1 | 0.494 | 0.483 |
| 2 | 3 | 0.510 | 0.478 |
| 4 | 5 | 0.548 | 0.497 |
| 8 | 9 | 0.587 | 0.494 |

Plain R² climbed 0.49 → 0.59 on nothing but noise. Adjusted R² stayed roughly flat. (Check one row: 1 − (1 − 0.587) × 49 / 40 = 0.494.)

## 6. Correlation is not causation

A strong r says two things move together. It does not say one causes the other. Three common reasons:

- **Confounding** — a third variable drives both. Ice-cream sales and drownings correlate because hot weather increases both. A **confounder** is that hidden third variable.
- **Reverse causation** — the arrow points the other way.
- **Chance** — test enough pairs and some will correlate by luck (the same multiple-testing problem as p-hacking in [Power & sample size](#/t/power)).

**Simpson's paradox** — a trend that appears in every group reverses (or vanishes) when the groups are pooled, because a confounder sits behind the grouping.

![Three coloured groups each slope downwards (r about −0.8), but the dashed line through all points pooled slopes upwards (r = 0.86).](figures/correlation-simpson.svg)

In the figure, within each colour y falls as x rises. But the groups themselves sit higher and further right in turn, so the pooled cloud slopes up. Which answer is right depends on the question — and you only see the trap if you split by the group.

## 7. In code

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([1, 2, 3, 4, 5, 6])            # hours studied
y = np.array([52, 55, 61, 60, 68, 72])      # exam score

r, p = stats.pearsonr(x, y)                 # r and the p-value for H0: r = 0
rho, p_s = stats.spearmanr(x, y)            # Spearman: Pearson on the ranks
print(round(r, 3), round(rho, 3))           # 0.973 0.943
print(round(r**2, 3))                       # r-squared: 0.947

# correlation matrix for a whole DataFrame
rng = np.random.default_rng(0)
df = pd.DataFrame({"hours": rng.normal(5, 2, 100)})
df["score"] = 50 + 3 * df["hours"] + rng.normal(0, 5, 100)   # depends on hours
df["sleep"] = rng.normal(7, 1, 100)                           # unrelated noise

corr = df.corr(method="pearson")            # or method="spearman"
sns.heatmap(corr, annot=True, fmt=".2f",    # write the value in each cell
            cmap="vlag", vmin=-1, vmax=1,   # diverging colours centred on 0
            square=True)
plt.title("Correlation matrix")
plt.show()
```

- `vmin=-1, vmax=1` pins the colour scale so 0 is always the neutral colour; without it seaborn stretches the colours to your data's range and weak correlations can look strong.
- The p-value from `pearsonr` tests whether r differs from 0. With large n, even r = 0.05 can be "significant" — check the size of r, not just p.

**From his US Accidents EDA.** Jag used `.corr()` to spot redundant features: wind chill and temperature were highly correlated, so one was dropped ([dcdm.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/dcdm.ipynb)):

```python
df_cleaned[['wind_chill(f)', 'temperature(f)']].corr()
# drop one column they are highly correlated
df_cleaned.drop(columns = ['wind_chill(f)'], inplace = True)
```

## 8. Do I need the equations?

Not to memorise. As a data scientist you call them (`scipy.stats.pearsonr`, `.spearmanr`, sklearn's `r2_score`). What you must own is the **interpretation** — the phrasing above — because that's what an interview and a stakeholder actually test. Working one small example by hand, as above, is what makes the phrasing stick.

## 9. Saying it in words

> **Say it like this:** "Hours studied and exam score are strongly positively correlated (r = 0.97); hours studied explain about 95% of the variation in scores in this sample. That is an association, not proof that studying more caused the higher scores."

## 10. Traps

| Wrong | Right |
|---|---|
| "Pearson is linear, Spearman is non-linear" | Both measure association in one direction; Spearman handles curves that keep rising, and outliers |
| r = 0 means no relationship | No **linear** relationship; plot it |
| r = 0.7 means "70% related" | r² = 0.49: 49% of the variation in the outcome explained |
| r is the slope | r is tightness to a line, not steepness |
| More predictors, higher R², better model | Check adjusted R² or a validation set |
| Strong correlation, so X causes Y | Could be confounding, reverse causation or chance |
| Trusting a pooled trend | Split by group to check for Simpson's paradox |

## Quick reference

| Term | One line |
|---|---|
| Covariance | Average product of deviations from both means; unit-dependent |
| Pearson r | cov(x, y) / ($s_x s_y$); strength and direction of a straight-line link, −1 to +1 |
| Spearman ρ | Pearson on the ranks; monotonic trends, outliers, ordinal data |
| r² | Share of the variation in the outcome explained by the other variable |
| Halving r | Quarters r² |
| Adjusted R² | 1 − (1 − R²)(n − 1)/(n − p − 1); R² minus a noise tax per predictor |
| Confounder | Hidden third variable driving both |
| Simpson's paradox | Group-level trend reverses when groups are pooled |

**The order of calculation:** plot it → means → deviations → products and squares → sum → r → r² → sentence (with "association, not causation").
