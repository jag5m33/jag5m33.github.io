# Descriptive statistics

The handful of numbers that summarise a column: where its centre is, how spread out it is, and what shape it has.

## Population vs sample

| Term | Meaning | Symbols |
|---|---|---|
| Population | Everyone (or everything) you care about | mean $\mu$, SD $\sigma$ |
| Sample | The subset you actually measured | mean $\bar{x}$, SD $s$, size $n$ |

> **Key idea:** Greek letters belong to the population, ordinary letters to the sample. You almost never have the whole population, so you calculate sample numbers and use them as estimates.

**Descriptive statistics** summarise the data you have. **Inferential statistics** ([Confidence intervals](#/t/ci), [Hypothesis testing](#/t/hypothesis)) use those summaries to say something about the population.

## The worked dataset

Let's say we have 8 values (already sorted):

$$
2,\ 4,\ 4,\ 4,\ 5,\ 5,\ 7,\ 9
$$

Every number below comes from these eight.

## Measures of centre

### Mean

**Definition:** the average. Add everything up, divide by how many there are.

$$
\bar{x} = \frac{\text{sum of values}}{n}
$$

**Worked:** $2+4+4+4+5+5+7+9 = 40$, and $40 / 8 = 5$. So $\bar{x} = 5$.

The mean is the **balance point**: the distances below it and above it cancel out exactly.

![The eight values as dots, with the mean (gold) and median (blue dashed). Each horizontal line is a deviation from the mean; the negatives (wine) and positives (teal) cancel exactly.](figures/descriptive-deviations.svg)

In the figure, add the little signed numbers: $-3-1-1-1+0+0+2+4 = 0$. That is why you can't measure spread by just averaging distances from the mean, and why variance squares them (below).

### Median

**Definition:** the middle value once the data are sorted. Half the values are below it, half above. With an even count, average the two middle values.

**Worked:** 8 values, so the middle two are the 4th and 5th: $4$ and $5$. Median $= (4+5)/2 = 4.5$.

### Mode

**Definition:** the most frequent value.

**Worked:** $4$ appears three times. Mode $= 4$.

The mode is the only centre that works for **categorical** data ("the most common accident severity is 2").

### When the median is the honest one

Let's say eight people work in a small office. Salaries in £ thousands:

$$
24,\ 26,\ 27,\ 28,\ 30,\ 31,\ 32,\ 150
$$

- Mean $= 348/8 = 43.5$
- Median $= (28+30)/2 = 29$

![One very large salary (the director) drags the mean to £43.5k, higher than seven of the eight people earn. The median stays at £29k.](figures/descriptive-salaries.svg)

In the figure, the mean (gold) sits in empty space where nobody actually earns. The median is **robust**: it only cares about order, not how extreme the extremes are.

> **Say it like this:** "Salaries are right-skewed by a few high earners, so I report the median as the typical value. The mean is pulled up by the tail."

This is the rule in Jag's US-accidents EDA: his `audit_cols` function flags a column as "median must be used, mean is likely biased" when the mean sits above the 75th percentile ([py_acc eda.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb)). The same logic decides missing-value filling: fill skewed columns with the median.

## Measures of spread

### Range

**Definition:** largest minus smallest: $9 - 2 = 7$.

Quick, but one outlier controls it completely. The [IQR](#/t/quartiles) is the robust alternative.

### Variance and standard deviation

**Variance** is the average squared distance from the mean. **Standard deviation (SD)** is its square root, which puts spread back into the original units.

Why square? Plain distances cancel to zero (see the figure above). Squaring makes every distance positive.

$$
s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1} \qquad s = \sqrt{s^2}
$$

- $x_i$: each value. $\bar{x}$: the mean. $n$: how many values.
- $s^2$: sample variance. $s$: sample SD.

**Worked, step by step:**

| Step | What you do | Result |
|---|---|---|
| 1 | Mean | $5$ |
| 2 | Distances from the mean | $-3, -1, -1, -1, 0, 0, 2, 4$ |
| 3 | Square them | $9, 1, 1, 1, 0, 0, 4, 16$ |
| 4 | Add them up | $32$ |
| 5 | Divide by $n - 1 = 7$ | $s^2 \approx 4.571$ |
| 6 | Square root | $s \approx 2.138$ |

> **Say it like this:** "The values typically sit about 2.1 units from the mean of 5."

**Why $n-1$?** The distances are measured from the sample's own mean, which sits a little closer to the data than the true mean would. Dividing by $n-1$ instead of $n$ nudges the answer up to correct for that. If you had the whole population you would divide by $n$ (here: $32/8 = 4$, SD $= 2$).

> **Trap:** pandas and numpy disagree by default. `pd.Series.std()` divides by $n-1$; `np.std()` divides by $n$. Set `ddof=1` (for $n-1$) or `ddof=0` (for $n$) explicitly.

## Shape: skew

**Skewness** measures lopsidedness: which side has the longer tail.

| Shape | Tail points | Mean vs median |
|---|---|---|
| Right (positive) skew | Right, towards large values | mean > median |
| Symmetric | Both equally | equal |
| Left (negative) skew | Left, towards small values | mean < median |

![Three distributions. In each skewed one the mode sits at the peak, the mean is dragged furthest into the tail, and the median sits between them.](figures/descriptive-skew.svg)

In the figure, watch the gold mean line: it always moves towards the tail. Skew is named after the **tail**, not the peak, so a right-skewed distribution has its hump on the left.

> **Trap:** "Left-skewed" means the tail is on the left. Jag corrected exactly this in his accidents EDA: humidity (skew $-0.72$) has its mass at 90–100% and a thin tail down to 0, so it is **left**-skewed.

A big mean–median gap means the mean is not a typical value: report the median, and consider a [log transform](#/t/logs) for right skew. Real right-skewed examples: salaries, house prices, incident durations. Jag's accident `Distance(mi)` column had skew around 1,649.

Rough rule for the skew number pandas gives you: below 0.5 (either sign) fairly symmetric, 0.5 to 1 moderate, above 1 strong. Our 8 values give about 0.82: mildly right-skewed, matching mean $5 >$ median $4.5$.

## Code

```python
import numpy as np
import pandas as pd

s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])   # the worked dataset

print(s.describe())        # count, mean, std, min, quartiles, max
print(s.mean(), s.median(), s.mode()[0])   # 5.0  4.5  4
print(s.std(ddof=1))       # 2.138  sample SD (pandas default)
print(np.std(s))           # 2.0    numpy divides by n, careful
print(s.max() - s.min())   # 7      range
print(s.skew())            # 0.818  positive: mild right skew

# On a whole DataFrame: skew of every numeric column
# df.select_dtypes("number").skew().round(2)
```

## Quick reference

| Term | Rule | Worked value |
|---|---|---|
| Mean | sum ÷ count | 5 |
| Median | middle of sorted data | 4.5 |
| Mode | most frequent value | 4 |
| Range | max − min | 7 |
| Sample variance | squared distances summed, ÷ $(n-1)$ | 4.571 |
| SD | square root of variance | 2.138 |
| Right skew | tail right, mean > median | report median |
| Left skew | tail left, mean < median | report median |
| pandas vs numpy | `.std()` uses $n-1$, `np.std` uses $n$ | set `ddof` |
