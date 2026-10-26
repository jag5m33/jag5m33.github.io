# Correlation

Correlation puts one number on how strongly two variables move together, and in which direction.

## Pearson vs Spearman

| | Works on | Catches |
|---|---|---|
| **Pearson** | The raw values | Straight-line relationships |
| **Spearman** | The **ranks** of the values | Any relationship that keeps going up (or keeps going down), even if it curves; not fooled by outliers |

- **Rank**: a value's position when sorted: smallest = 1, next = 2, and so on.

> **Key idea:** Spearman is just Pearson run on the ranks instead of the raw numbers. It is not "Pearson linear, Spearman non-linear".

## r, the Pearson correlation

**r** runs from −1 to +1 and captures the **strength and direction** of a straight-line link.

| r | Meaning |
|---|---|
| +1 | Perfect upward line |
| 0 | No straight-line link: a shapeless cloud |
| −1 | Perfect downward line |

![Five scatter plots with correlations of about −0.9, −0.5, 0, +0.5 and +0.9.](figures/correlation-grid.svg)

In the figure, the sign sets the direction and the size sets how tightly the cloud hugs a line. r is **not** the slope: a shallow line and a steep line can both have r = 0.9.

### How it works, in words

1. For each point, find how far x is from its mean and how far y is from its mean.
2. Multiply those two distances. If a point is above average on both (or below on both), the product is positive. If it's above on one and below on the other, it's negative.
3. Add up the products. Big positive total → they move together.
4. Divide by the spread of x and the spread of y, which strips out the units and squeezes the answer into −1 to +1.

$$
r = \frac{\text{sum of products}}{\sqrt{\text{sum of squared x-distances} \times \text{sum of squared y-distances}}}
$$

### Worked example

Let's say we have 6 students: hours studied (x) and exam score (y).

![The six points with dotted lines at the two means; vertical lines show each point's deviation from the mean score. Five points are teal (positive product), one is red (negative).](figures/correlation-worked.svg)

In the figure, the dotted lines are the two means. Teal points add to the total; the one red point (above-average hours, below-average score) subtracts a little.

Means: hours = **3.5**, score = **61.33**.

| x | y | x distance | y distance | product |
|---|---|---|---|---|
| 1 | 52 | −2.5 | −9.33 | 23.33 |
| 2 | 55 | −1.5 | −6.33 | 9.50 |
| 3 | 61 | −0.5 | −0.33 | 0.17 |
| 4 | 60 | 0.5 | −1.33 | −0.67 |
| 5 | 68 | 1.5 | 6.67 | 10.00 |
| 6 | 72 | 2.5 | 10.67 | 26.67 |
| **Sum** | | | | **69.0** |

| Step | Working | Result |
|---|---|---|
| 1. Sum of products | from the table | 69.0 |
| 2. Squared distances | x: 17.5, y: 287.33 | √(17.5 × 287.33) = 70.91 |
| 3. r | 69.0 / 70.91 | **r = 0.973** |

**Spearman on the same data.** Rank the scores: 52, 55, 61, 60, 68, 72 become 1, 2, **4, 3**, 5, 6. Only students 3 and 4 swap. Pearson on the ranks gives **0.943**.

## When to reach for Spearman

- **Curved but always rising**: e.g. income rising with experience but levelling off.
- **Outliers**: a rank can't be more extreme than "the biggest".
- **Ordered categories**: ratings like 1–5 stars.

![Left: 12 points on a clean line, Pearson 0.98 and Spearman 0.97. Right: one extra point typed as 60 instead of about 13 drags Pearson to 0.67 while Spearman stays at 0.97.](figures/correlation-outlier.svg)

In the figure, the right panel adds one bad reading. Pearson is dragged from 0.98 to 0.67. Spearman only sees "largest value at the largest x", which fits the trend, so it stays at 0.97.

**Neither catches a relationship that turns back on itself.**

![A U-shaped scatter where y = x squared plus noise: Pearson r is 0.02 and Spearman is 0.00.](figures/correlation-curve.svg)

In the figure, y depends completely on x, yet both numbers are about zero, because the left half goes down and the right half goes up. **r = 0 means no straight-line link, not no link.** Always plot the data.

## r², the one to phrase carefully

> **Key idea:** r² is the share of the variation in the outcome that the other variable explains. NOT "the share of the correlation," and NOT "how related they are as a percent."

**Height and weight.** r = 0.7 → r² = 0.49 → height accounts for about **49% of why people's weights differ**. The other 51% is diet, genes, and noise.

**Why square it:** it turns a −1-to-1 correlation into a clean 0-to-1 share.

> **Trap:** r = 0.7 is NOT "70% related." And halving r quarters r²: r = 0.5 → r² = 0.25, so 25% explained, 75% down to everything else.

In the worked example, r = 0.973 → r² = 0.947: hours studied account for about 95% of the variation in these six scores (a tiny made-up sample; real data are rarely this tidy).

## Adjusted R², the noise tax

With several predictors in a model, R² is the same idea (more in [Regression metrics](#/t/regmetrics)).

**Plain R² only ever goes up.** Add any column, even pure noise, and it rises a little, because a useless column can always fit a scrap of noise by luck.

**Adjusted R² charges every predictor a "noise tax".** A junk column raises plain R² by about the tax, so adjusted R² stays flat. A real column raises it by much more, so adjusted R² still rises.

| Junk columns added | Plain R² | Adjusted R² |
|---|---|---|
| 0 | 0.494 | 0.483 |
| 4 | 0.548 | 0.497 |
| 8 | 0.587 | 0.494 |

(50 points, one real variable, then pure-noise columns.) Plain R² climbed on nothing but noise; adjusted R² stayed flat.

## Correlation is not causation

A strong r says two things move together, not that one causes the other.

- **Confounding**: a hidden third variable (a **confounder**) drives both. Ice-cream sales and drownings correlate because hot weather increases both.
- **Reverse causation**: the arrow points the other way.
- **Chance**: test enough pairs and some will correlate by luck.

**Simpson's paradox**: a trend inside every group reverses when the groups are pooled together.

![Three coloured groups each slope downwards (r about −0.8), but the dashed line through all points pooled slopes upwards (r = 0.86).](figures/correlation-simpson.svg)

In the figure, within each colour y falls as x rises, but the pooled cloud slopes up. You only see the trap if you split by group.

## In code

```python
import numpy as np
from scipy import stats

x = np.array([1, 2, 3, 4, 5, 6])            # hours studied
y = np.array([52, 55, 61, 60, 68, 72])      # exam score

r, p = stats.pearsonr(x, y)                 # r and a p-value for "r = 0"
rho, _ = stats.spearmanr(x, y)              # Pearson on the ranks
print(round(r, 3), round(rho, 3), round(r**2, 3))   # 0.973 0.943 0.947

# Whole DataFrame as a heatmap:
# sns.heatmap(df.corr(), annot=True, cmap="vlag", vmin=-1, vmax=1)
```

`vmin=-1, vmax=1` pins the colours so 0 is always neutral; otherwise weak correlations can look strong.

**From his US Accidents EDA.** Jag used `.corr()` to spot redundant features: wind chill and temperature were highly correlated, so one was dropped ([dcdm.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/dcdm.ipynb)):

```python
df_cleaned[['wind_chill(f)', 'temperature(f)']].corr()
# drop one column they are highly correlated
df_cleaned.drop(columns = ['wind_chill(f)'], inplace = True)
```

## Saying it in words

> **Say it like this:** "Hours studied and exam score are strongly positively correlated (r = 0.97); hours studied explain about 95% of the variation in scores in this sample. That is an association, not proof that studying more caused the higher scores."

## Traps

| Wrong | Right |
|---|---|
| "Pearson linear, Spearman non-linear" | Spearman handles curves that keep rising, and outliers |
| r = 0 means no relationship | No **straight-line** relationship; plot it |
| r = 0.7 means "70% related" | r² = 0.49: 49% of the variation explained |
| r is the slope | r is tightness to a line, not steepness |
| More columns, higher R², better model | Check adjusted R² |
| Strong correlation, so X causes Y | Could be confounding, reverse causation or chance |

## Quick reference

| Term | One line |
|---|---|
| Pearson r | Strength and direction of a straight-line link, −1 to +1 |
| Spearman | Pearson on the ranks; curves that keep rising, outliers, ratings |
| r² | Share of the variation in the outcome explained by the other variable |
| Halving r | Quarters r² |
| Adjusted R² | R² minus a noise tax per predictor |
| Confounder | Hidden third variable driving both |
| Simpson's paradox | Group trend reverses when groups are pooled |

**Order of calculation:** plot it → means → distances → products → sum → r → r² → sentence (with "association, not causation").
