# Descriptive statistics

The handful of numbers that summarise a column: where its centre is, how spread out it is, and what shape it has.

## Population vs sample

Everything in statistics starts with this split.

| Term | Meaning | Symbols |
|---|---|---|
| Population | Everyone (or everything) you care about | size $N$, mean $\mu$, SD $\sigma$ |
| Sample | The subset you actually measured | size $n$, mean $\bar{x}$, SD $s$ |
| Parameter | A true value describing the population. Fixed, usually unknown | $\mu$, $\sigma$ |
| Statistic | A value you calculate from the sample. Your estimate of the parameter | $\bar{x}$, $s$ |

> **Key idea:** Greek letters belong to the population, Roman letters to the sample. You almost never have the population, so you calculate sample statistics and use them to estimate population parameters.

**Descriptive statistics** summarise the data you have. **Inferential statistics** ([Confidence intervals](#/t/ci), [Hypothesis testing](#/t/hypothesis)) use those summaries to say something about the population.

## The worked dataset

Let's say we have 8 values (already sorted):

$$
2,\ 4,\ 4,\ 4,\ 5,\ 5,\ 7,\ 9
$$

Every number in this chapter is calculated from these eight.

## Measures of centre

### Mean

**Definition:** the arithmetic average. Add everything up, divide by how many there are.

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad \text{(population: } \mu = \tfrac{1}{N}\textstyle\sum x_i\text{)}
$$

**Worked:** $2+4+4+4+5+5+7+9 = 40$, and $40 / 8 = 5$. So $\bar{x} = 5$.

Where it comes from: the mean is the **balance point**. If you put the values on a see-saw, it balances at the mean, which is the same as saying the deviations $x_i - \bar{x}$ add up to zero.

![The eight values as dots, with the mean (gold) and median (blue dashed). Each horizontal line is a deviation from the mean; the negatives (wine) and positives (teal) cancel exactly.](figures/descriptive-deviations.svg)

In the figure, read the little signed numbers: $-3-1-1-1+0+0+2+4 = 0$. That cancelling is why you can't measure spread by averaging raw deviations, and why variance squares them (below).

### Median

**Definition:** the middle value once the data are sorted. Half the values are below it, half above.

- $n$ odd: the value at position $\frac{n+1}{2}$.
- $n$ even: the average of the two middle values, positions $\frac{n}{2}$ and $\frac{n}{2}+1$.

**Worked:** $n = 8$ is even, so take positions 4 and 5: values $4$ and $5$. Median $= (4+5)/2 = 4.5$.

### Mode

**Definition:** the most frequent value. A dataset can have one mode, several (bimodal, multimodal), or none worth reporting (every value appears once).

**Worked:** $4$ appears three times, more than anything else. Mode $= 4$.

The mode is the only centre that works for **categorical** data ("the most common accident severity is 2").

### When the median is the honest one

Let's say eight people work in a small office. Salaries in £ thousands:

$$
24,\ 26,\ 27,\ 28,\ 30,\ 31,\ 32,\ 150
$$

- Mean $= 348/8 = 43.5$
- Median $= (28+30)/2 = 29$

![One very large salary (the director) drags the mean to £43.5k, higher than seven of the eight people earn. The median stays at £29k.](figures/descriptive-salaries.svg)

In the figure, the mean (gold) sits in empty space where nobody actually earns. Remove the director and the mean drops to about £28.3k, while the median barely moves. The median is **robust**: it only cares about order, not about how extreme the extremes are.

> **Say it like this:** "Salaries are right-skewed by a few high earners, so I report the median as the typical value. The mean is pulled up by the tail."

This is exactly the rule in Jag's US-accidents EDA: his `audit_cols` function flags a column as "median must be used, mean is likely biased" when the mean sits above the 75th percentile ([py_acc eda.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb)). The same logic decides imputation: fill skewed columns with the median.

## Measures of spread

### Range

**Definition:** largest minus smallest.

$$
\text{range} = \max - \min = 9 - 2 = 7
$$

Quick, but it uses only two points, so a single outlier controls it completely. The [IQR](#/t/quartiles) is the robust alternative.

### Variance and standard deviation, in calculation order

**Variance** is the average squared distance from the mean. **Standard deviation (SD)** is its square root, which puts spread back into the original units.

Why square? Raw deviations cancel to zero (see the figure above). Squaring makes every deviation positive and punishes big deviations more than small ones.

**Population formulas** (you have everyone):

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2 \qquad \sigma = \sqrt{\sigma^2}
$$

**Sample formulas** (you have a subset, which is the usual case):

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2 \qquad s = \sqrt{s^2}
$$

The only difference is the denominator: $N$ versus $n-1$.

**Worked, step by step:**

| Step | What you do | Result |
|---|---|---|
| 1 | Mean | $\bar{x} = 5$ |
| 2 | Deviations $x_i - \bar{x}$ | $-3, -1, -1, -1, 0, 0, 2, 4$ |
| 3 | Square them | $9, 1, 1, 1, 0, 0, 4, 16$ |
| 4 | Sum of squares | $32$ |
| 5a | Population variance: divide by $n = 8$ | $\sigma^2 = 4$ |
| 6a | Population SD | $\sigma = 2$ |
| 5b | Sample variance: divide by $n - 1 = 7$ | $s^2 \approx 4.571$ |
| 6b | Sample SD | $s \approx 2.138$ |

> **Say it like this:** "The values typically sit about 2.1 units from the mean of 5."

### Why n − 1 (Bessel's correction), simply

The deviations in step 2 are measured from $\bar{x}$, the **sample's own** mean, not the true $\mu$. The sample mean is, by construction, the point that makes the sum of squared deviations as small as it can possibly be for that sample. So deviations from $\bar{x}$ are, on average, a little smaller than deviations from the true $\mu$ would be. Divide by $n$ and you **underestimate** the true variance.

A second way to see it: once you know $\bar{x}$ and any $n-1$ of the deviations, the last one is forced (they must sum to zero). Only $n-1$ deviations are free to vary. That count is called the **degrees of freedom**.

Dividing by $n-1$ instead of $n$ inflates the answer by exactly the right amount to remove the bias on average. With small $n$ this matters; with $n = 10{,}000$ the difference is negligible.

![Simulation: 20,000 samples of size 5 from a population with variance 4. Dividing by n (wine) averages about 3.2; dividing by n − 1 (teal) averages about 4.0.](figures/descriptive-bessel.svg)

In the figure, the dashed lines are the average of each method across all 20,000 samples. The wine line misses the true value by the factor $\frac{n-1}{n} = \frac{4}{5}$, which is exactly $3.2$. The teal line lands on 4.

> **Trap:** pandas and numpy disagree by default. `pd.Series.var()` and `.std()` use $n-1$ (`ddof=1`). `np.var()` and `np.std()` use $n$ (`ddof=0`). Always set `ddof` explicitly.

`ddof` means "delta degrees of freedom": the denominator is $n - \text{ddof}$.

## Shape: skew

**Skewness** measures asymmetry: which side has the longer tail.

| Shape | Tail points | Order of the three centres | Mean vs median |
|---|---|---|---|
| Right (positive) skew | Right, towards large values | mode < median < mean | mean > median |
| Symmetric | Both equally | mode = median = mean | equal |
| Left (negative) skew | Left, towards small values | mean < median < mode | mean < median |

![Three distributions. In each skewed one the mode sits at the peak, the mean is dragged furthest into the tail, and the median sits between them.](figures/descriptive-skew.svg)

In the figure, look at which way the gold mean line moves: always towards the tail. The skew is named after the **tail**, not the peak. A right-skewed distribution has its hump on the left.

> **Trap:** "Left-skewed" means the tail is on the left. Jag corrected exactly this in his accidents EDA: humidity (skew $-0.72$) has its mass at 90–100% and a thin tail down to 0, so it is **left**-skewed.

**What a mean–median gap tells you.** If mean $>$ median, there is a right tail or some large outliers pulling the mean up. If mean $<$ median, a left tail. A big gap means the mean is not a typical value, so report the median (and consider a [log transform](#/t/logs) for right skew).

Real examples of right skew: salaries, house prices, incident durations, word counts. Jag's accident `Distance(mi)` column had skew around 1,649: a handful of huge values dragging the mean far from anything typical.

The usual sample skewness is the average cubed z-score (with a small-sample correction in pandas):

$$
\text{skew} \approx \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3
$$

Cubing keeps the sign, so a long right tail gives big positive values. For our 8 numbers the formula above gives about 0.54 (0.66 if you use the population SD instead of $s$); pandas gives skew $\approx 0.82$ because it applies an adjustment factor $\sqrt{n(n-1)}/(n-2)$ that is large for small $n$: mildly right-skewed (the 9 and 7 stretch the right side), consistent with mean $5 >$ median $4.5$.

Rough rule of thumb: $\lvert \text{skew} \rvert < 0.5$ fairly symmetric, $0.5$ to $1$ moderate, $> 1$ strong.

## Shape: kurtosis (briefly)

**Kurtosis** describes the tails: how often extreme values happen and how far out they go. It uses the fourth power of z-scores, so a few extreme points dominate it.

pandas reports **excess kurtosis** (normal distribution $= 0$):

- $> 0$: heavy tails, more extreme values than a normal (e.g. financial returns).
- $= 0$: normal-like tails.
- $< 0$: light tails, flatter (e.g. uniform, which is $-1.2$).

Jag's accident durations had kurtosis around 3,660, a sign of a long tail of absurd values (some lasting years).

## Code

```python
import numpy as np
import pandas as pd

s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])   # the worked dataset

print(s.describe())        # count, mean, std (n-1), min, 25%, 50%, 75%, max
print(s.mean())            # 5.0
print(s.median())          # 4.5
print(s.mode())            # 0    4   (a Series: there can be several modes)

print(s.var(ddof=1))       # 4.571...  sample variance (pandas default)
print(s.var(ddof=0))       # 4.0       population variance
print(s.std(ddof=1))       # 2.138...  sample SD
print(np.std(s))           # 2.0       numpy default is ddof=0, careful

print(s.max() - s.min())   # 7         range
print(s.skew())            # 0.818...  positive: mild right skew
print(s.kurtosis())        # 0.94...   excess kurtosis (normal = 0)

# On a whole DataFrame: every numeric column at once
# df.describe(include="number", percentiles=[0.5, 0.75, 0.99])
# df.select_dtypes("number").skew().round(2)
```

`describe()` on our data shows `std 2.13809`, confirming it uses $n-1$.

## Quick reference

| Term | Formula / rule | Worked value |
|---|---|---|
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ | 5 |
| Median | middle of sorted data (average the two middles if $n$ even) | 4.5 |
| Mode | most frequent value | 4 |
| Range | $\max - \min$ | 7 |
| Population variance | $\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2$ | 4 |
| Sample variance | $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$ | 4.571 |
| SD | square root of variance | 2 (pop), 2.138 (sample) |
| Bessel's correction | divide by $n-1$ because deviations are measured from $\bar{x}$ | removes underestimate |
| Right skew | tail right; mean > median > mode | report median |
| Left skew | tail left; mean < median < mode | report median |
| Excess kurtosis | 0 normal, > 0 heavy tails, < 0 light tails | 0.94 |
| pandas vs numpy | `.std()` uses ddof=1, `np.std` uses ddof=0 | set `ddof` |
