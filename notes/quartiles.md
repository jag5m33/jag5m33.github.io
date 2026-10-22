# Quartiles, IQR & outliers

How to cut sorted data into quarters, measure the spread of its middle half, and use that to flag values that look out of place.

## Definitions first

| Term | Plain meaning |
|---|---|
| **Percentile** | The $p$th percentile is the value with $p\%$ of the data at or below it. The 90th percentile: 90% of values are no bigger. |
| **Quantile** | The same idea written as a fraction instead of a percent: the 0.9 quantile = the 90th percentile. pandas calls it `quantile`. |
| **Quartile** | The three cut points that split sorted data into four equal-sized groups. |
| **Q1** (lower quartile) | 25th percentile. A quarter of the data sits below it. |
| **Q2** | 50th percentile, which is the **median**. |
| **Q3** (upper quartile) | 75th percentile. Three quarters of the data sit below it. |
| **IQR** (interquartile range) | $Q3 - Q1$: the width of the middle 50% of the data. |
| **Outlier** | A value unusually far from the rest. "Unusually far" needs a rule, and the rules are below. |

> **Key idea:** quartiles depend only on the **order** of the data, not on how extreme the extremes are. Make the largest value a million and Q1, Q2, Q3 do not move. That is what makes them robust.

## Computing quartiles by hand

Let's say we have 11 values, already sorted:

$$
3,\ 5,\ 7,\ 8,\ 9,\ 10,\ 11,\ 12,\ 13,\ 15,\ 40
$$

### Step 1: find the position

numpy's default method ("linear") puts the $p$ quantile at position (counting from 1):

$$
\text{position} = 1 + (n - 1)\,p
$$

Where it comes from: the smallest value sits at position 1 (that's $p = 0$) and the largest at position $n$ (that's $p = 1$). Spread $p$ evenly along the $n-1$ gaps between them.

With $n = 11$:

| Quartile | $p$ | Position $1 + 10p$ |
|---|---|---|
| Q1 | 0.25 | 3.5 |
| Q2 | 0.50 | 6 |
| Q3 | 0.75 | 8.5 |

### Step 2: read off the value, interpolating if the position is fractional

If the position is a whole number, take that value. If it falls between two values, go that fraction of the way from the lower to the upper one (**linear interpolation**):

$$
Q = x_{(k)} + f \cdot \big(x_{(k+1)} - x_{(k)}\big)
$$

where $k$ is the whole part of the position, $f$ the fractional part, and $x_{(k)}$ is the $k$th smallest value.

- **Q1**, position 3.5: between value #3 $= 7$ and #4 $= 8$. $\;7 + 0.5 \times (8 - 7) = 7.5$
- **Q2**, position 6: value #6 $= 10$
- **Q3**, position 8.5: between #8 $= 12$ and #9 $= 13$. $\;12 + 0.5 \times (13 - 12) = 12.5$

![The 11 sorted values with their position numbers. The arrows show where Q1, the median and Q3 land: two of them fall halfway between neighbours.](figures/quartiles-positions.svg)

In the figure, follow each arrow down: Q1 lands between the 3rd and 4th circles, so its value is between 7 and 8.

**A less tidy fraction.** Take the 8 values from [Descriptive statistics](#/t/descriptive): $2, 4, 4, 4, 5, 5, 7, 9$. Q3 position $= 1 + 7 \times 0.75 = 6.25$. Value #6 is $5$, #7 is $7$, so Q3 $= 5 + 0.25 \times (7 - 5) = 5.5$. Q1 position $= 2.75$, between $4$ and $4$, so Q1 $= 4$. This matches the `25%` and `75%` rows of `describe()`.

### Different methods, slightly different answers

There is no single agreed definition of a quartile for a finite sample. A common "by hand" textbook method (also used by many calculators) splits the data at the median, leaves the median itself out, and takes the median of each half:

- lower half $3, 5, 7, 8, 9$ → Q1 $= 7$
- upper half $11, 12, 13, 15, 40$ → Q3 $= 13$

That gives 7 and 13 rather than numpy's 7.5 and 12.5. numpy offers several methods through `np.percentile(x, 25, method=...)` (`"linear"` is the default; others include `"lower"`, `"higher"`, `"midpoint"`, `"nearest"`, `"hazen"`, `"weibull"`). Excel's `QUARTILE.INC` matches numpy's linear; `QUARTILE.EXC` does not.

> **Trap:** don't panic if your hand calculation differs from Python by a fraction. Small datasets expose the method choice; on thousands of rows the methods agree almost exactly. State the method if it matters.

## IQR

$$
\text{IQR} = Q3 - Q1 = 12.5 - 7.5 = 5
$$

> **Say it like this:** "The middle half of the values span 5 units, from 7.5 to 12.5."

The IQR is the robust partner of the median, the same way the SD is the partner of the mean. Compare with the range: $40 - 3 = 37$, controlled entirely by the 40.

## Tukey's fences

John Tukey's rule: anything more than $1.5 \times \text{IQR}$ beyond the box is a potential outlier.

$$
\text{lower fence} = Q1 - 1.5 \times \text{IQR} \qquad \text{upper fence} = Q3 + 1.5 \times \text{IQR}
$$

**Worked:**

| Step | Calculation | Result |
|---|---|---|
| 1 | $1.5 \times \text{IQR} = 1.5 \times 5$ | 7.5 |
| 2 | Lower fence $= 7.5 - 7.5$ | 0 |
| 3 | Upper fence $= 12.5 + 7.5$ | 20 |
| 4 | Any value $< 0$ or $> 20$? | 40 is flagged |

### Why 1.5?

It is a convention Tukey chose, but it has a sensible meaning on normal data. For a normal distribution, Q1 and Q3 sit at $\mu \pm 0.674\sigma$, so $\text{IQR} \approx 1.349\sigma$. Then:

$$
Q3 + 1.5 \times \text{IQR} \approx 0.674\sigma + 1.5 \times 1.349\sigma \approx 2.698\sigma
$$

So the fences sit at about $\pm 2.7\sigma$, and only about **0.7%** of perfectly normal data falls outside them. A "real" normal dataset will still produce a few flags, especially if it is large: 0.7% of 7.7 million rows is about 54,000 points.

![On a normal curve, the teal box is the middle 50% (±0.674σ). The fences land at about ±2.7σ and the two wine slivers beyond them hold about 0.7% of the data.](figures/quartiles-normal-fences.svg)

In the figure, the wine tails are what Tukey's rule would call outliers even though nothing is wrong with them. Using $3 \times \text{IQR}$ instead gives "far out" fences (about $\pm 4.7\sigma$) for flagging only extreme points.

## Box plot anatomy

A box plot draws all of the above in one picture.

![Every part of a box plot for the 11-value dataset: box from Q1 to Q3, median line inside, whiskers reaching to the most extreme points still inside the fences, and the 40 drawn on its own as an outlier. The dotted lines are the fences, which are normally not drawn.](figures/quartiles-boxplot.svg)

In the figure, note the whiskers do **not** go to the fences. They stop at the last real data point inside the fences (3 and 15). Anything beyond is drawn as a separate dot.

| Part | Shows |
|---|---|
| Box edges | Q1 and Q3 |
| Line in the box | Median |
| Box width | IQR, the middle 50% |
| Whiskers | Furthest data points within the fences |
| Dots | Points beyond the fences |

How to read one, from Jag's [US-accidents EDA](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb):

- A box sitting hard against one whisker means skew.
- A box collapsed to a line means one value dominates.
- Box plots flag individual extreme points; histograms show shape but bury anything rare. Run both.

## The alternative: the z-score rule

A **z-score** says how many standard deviations a value is from the mean (more in [Distributions](#/t/distributions)):

$$
z = \frac{x - \bar{x}}{s}
$$

The rule: flag $\lvert z \rvert > 3$. On normal data that flags about 0.27%.

**Worked on the same 11 values:** $\bar{x} \approx 12.09$, $s \approx 9.89$. For the 40:

$$
z = \frac{40 - 12.09}{9.89} \approx 2.82
$$

That is under 3, so the z-score rule says the 40 is **not** an outlier.

### Why IQR is more robust

The mean and SD are pulled by the very outliers they're trying to find. Without the 40, the other ten values have mean 9.3 and SD about 3.7. Adding the 40 drags the mean up to 12.1 and nearly triples the SD to 9.9, which stretches the "normal" band to cover the 40 itself. This is called **masking**.

![Top: mean ± 3 SD (gold band) is so inflated by the 40 that it contains it. Bottom: the IQR fences (teal band) are built from quartiles, which barely notice the 40, so it lands outside.](figures/quartiles-zscore-vs-iqr.svg)

In the figure, compare the widths of the two shaded bands. Same data, same 40; only the IQR rule catches it.

| | z-score rule | IQR rule |
|---|---|---|
| Built from | mean, SD | quartiles |
| Pulled by outliers | yes, badly | barely |
| Assumes | roughly normal data | nothing about shape |
| Good for | large, near-normal data | skewed or small data, first pass in EDA |

> **Trap:** neither rule finds *errors*. They find *unusual* values. A rule is a prompt to look, not a verdict.

## What to do with an outlier

Decide what it is, then decide what to do, then write it down.

1. **Is it an error?** Impossible or nonsensical values: a temperature of 160°F, air pressure of 2 inches (sea level is about 29.9), a negative age, a unit mix-up. Jag found all of these in the accidents data. Fix it if you can recover the true value, otherwise set it to missing or drop the row.
2. **Is it rare but real?** A genuine billionaire in a salary survey, a real storm in wind-speed data. Keep it. It is information. Options: use robust statistics (median, IQR), [log-transform](#/t/logs) the column, cap it (winsorise at, say, the 1st and 99th percentiles), or use a model that isn't sensitive to it (tree-based models).
3. **Is it a placeholder?** Suspicious spikes at round numbers (999, 0, exactly 10) can be "not recorded" in disguise.
4. **Document it.** What you flagged, which rule, what you did and why. Someone (including future you) has to be able to reproduce it.

> **Key idea:** never drop outliers just because a rule flagged them. In fraud or [anomaly detection](#/t/anomaly), the outliers are the whole point.

Outliers in the **target label** (wrong or impossible classes, rare classes) are a separate problem: see [Label checks](#/t/labelchecks).

## Code

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x = np.array([3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 40])
s = pd.Series(x, name="value")

# Quartiles: both use the linear method by default
q1, q2, q3 = np.percentile(x, [25, 50, 75])     # 7.5, 10.0, 12.5
print(s.quantile([0.25, 0.5, 0.75]))             # same values, as a Series
print(np.percentile(x, 25, method="weibull"))    # 7.0: a different method

# IQR and Tukey's fences
iqr = q3 - q1                                     # 5.0
lower = q1 - 1.5 * iqr                            # 0.0
upper = q3 + 1.5 * iqr                            # 20.0

# Boolean mask: True where the value is outside the fences
is_outlier = (s < lower) | (s > upper)
print(s[is_outlier])                              # 40
print(is_outlier.mean())                          # share of rows flagged

# z-score rule for comparison
z = (s - s.mean()) / s.std(ddof=1)
print(s[z.abs() > 3])                             # empty: the 40 masks itself

# Box plot (seaborn uses whis=1.5, i.e. Tukey's fences, by default)
sns.boxplot(x=s)
plt.show()
```

To flag outliers per column on a whole DataFrame, compute `df.quantile([0.25, 0.75])` once and apply the same fences column by column.

## Quick reference

| Term | Formula / rule | Worked value |
|---|---|---|
| Percentile $p$ | value with $p\%$ of data at or below | — |
| Position (numpy linear) | $1 + (n-1)p$, then interpolate | Q1 at 3.5 |
| Q1, Q2, Q3 | 25th, 50th, 75th percentiles | 7.5, 10, 12.5 |
| IQR | $Q3 - Q1$ | 5 |
| Lower fence | $Q1 - 1.5 \times \text{IQR}$ | 0 |
| Upper fence | $Q3 + 1.5 \times \text{IQR}$ | 20 |
| Fences on normal data | about $\pm 2.7\sigma$, about 0.7% flagged | — |
| z-score | $(x - \bar{x}) / s$, flag $\lvert z \rvert > 3$ | 40 → 2.82 |
| Masking | outliers inflate mean and SD and hide themselves | — |
| Whiskers | furthest points inside the fences | 3 and 15 |
| Outlier decision | error → fix/drop; real → keep, robust stats; always document | — |
