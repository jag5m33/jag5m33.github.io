# Quartiles, IQR & outliers

How to cut sorted data into quarters, measure the spread of its middle half, and use that to flag values that look out of place.

## Definitions first

| Term | Plain meaning |
|---|---|
| **Percentile** | The 90th percentile is the value with 90% of the data at or below it. |
| **Quantile** | The same idea as a fraction: the 0.9 quantile = the 90th percentile. pandas calls it `quantile`. |
| **Q1** (lower quartile) | 25th percentile. A quarter of the data sits below it. |
| **Q2** | 50th percentile, which is the **median**. |
| **Q3** (upper quartile) | 75th percentile. Three quarters of the data sit below it. |
| **IQR** (interquartile range) | $Q3 - Q1$: the width of the middle 50% of the data. |
| **Outlier** | A value unusually far from the rest. "Unusually far" needs a rule, and the rules are below. |

> **Key idea:** quartiles depend only on the **order** of the data. Make the largest value a million and Q1, Q2, Q3 do not move. That is what makes them robust.

## Computing quartiles by hand

Let's say we have 11 values, already sorted:

$$
3,\ 5,\ 7,\ 8,\ 9,\ 10,\ 11,\ 12,\ 13,\ 15,\ 40
$$

numpy finds each quartile like this: work out its **position** in the sorted list, and if the position falls between two values, go that fraction of the way between them.

$$
\text{position} = 1 + (n - 1) \times p
$$

- $n$: number of values (11). $p$: the fraction you want (0.25 for Q1).

| Quartile | Position | Value |
|---|---|---|
| Q1 | $1 + 10 \times 0.25 = 3.5$ | halfway between 7 and 8 → **7.5** |
| Q2 | $1 + 10 \times 0.5 = 6$ | the 6th value → **10** |
| Q3 | $1 + 10 \times 0.75 = 8.5$ | halfway between 12 and 13 → **12.5** |

![The 11 sorted values with their position numbers. The arrows show where Q1, the median and Q3 land: two of them fall halfway between neighbours.](figures/quartiles-positions.svg)

In the figure, follow each arrow down: Q1 lands between the 3rd and 4th circles, so its value is between 7 and 8.

> **Trap:** a textbook "by hand" method (median of each half) gives Q1 = 7 and Q3 = 13 here. Both are fine: there are several conventions. On small data they differ slightly; on thousands of rows they agree almost exactly.

## IQR

$$
\text{IQR} = Q3 - Q1 = 12.5 - 7.5 = 5
$$

> **Say it like this:** "The middle half of the values span 5 units, from 7.5 to 12.5."

The IQR is the robust partner of the median, the way the SD is the partner of the mean. Compare the range: $40 - 3 = 37$, controlled entirely by the 40.

## Tukey's fences

John Tukey's rule: anything more than $1.5 \times \text{IQR}$ beyond the box is a potential outlier.

$$
\text{lower fence} = Q1 - 1.5 \times \text{IQR} \qquad \text{upper fence} = Q3 + 1.5 \times \text{IQR}
$$

**Worked:**

| Step | Calculation | Result |
|---|---|---|
| 1 | $1.5 \times 5$ | 7.5 |
| 2 | Lower fence $= 7.5 - 7.5$ | 0 |
| 3 | Upper fence $= 12.5 + 7.5$ | 20 |
| 4 | Any value below 0 or above 20? | 40 is flagged |

**Why 1.5?** It's a convention. On perfectly normal (bell-shaped) data the fences land at about 2.7 SDs from the mean, so only about 0.7% of points get flagged. On big data that is still a lot: 0.7% of 7.7 million rows is about 54,000 points.

![On a normal curve, the teal box is the middle 50%. The fences land at about 2.7 SDs either side, and the two wine slivers beyond them hold about 0.7% of the data.](figures/quartiles-normal-fences.svg)

In the figure, the wine tails are what Tukey's rule calls outliers even though nothing is wrong with them.

## Box plot anatomy

A box plot draws all of the above in one picture.

![Every part of a box plot for the 11-value dataset: box from Q1 to Q3, median line inside, whiskers reaching to the most extreme points still inside the fences, and the 40 drawn on its own as an outlier. The dotted lines are the fences, which are normally not drawn.](figures/quartiles-boxplot.svg)

In the figure, the whiskers do **not** go to the fences. They stop at the last real data point inside the fences (3 and 15). Anything beyond is drawn as a separate dot.

| Part | Shows |
|---|---|
| Box edges | Q1 and Q3 |
| Line in the box | Median |
| Whiskers | Furthest data points within the fences |
| Dots | Points beyond the fences |

How to read one, from Jag's [US-accidents EDA](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb):

- A box sitting hard against one whisker means skew.
- A box collapsed to a line means one value dominates.
- Box plots flag individual extreme points; histograms show shape but bury anything rare. Run both.

## The alternative: the z-score rule

A **z-score** says how many standard deviations a value is from the mean (more in [Distributions](#/t/distributions)). The rule: flag anything with $z$ above 3 or below −3.

**Worked on the same 11 values:** mean $\approx 12.09$, SD $\approx 9.89$. For the 40: $z = (40 - 12.09) / 9.89 \approx 2.82$. Under 3, so the z-score rule says the 40 is **not** an outlier.

### Why IQR is more robust

The mean and SD are pulled by the very outliers they're trying to find. Without the 40, the other ten values have mean 9.3 and SD about 3.7. Adding the 40 drags the mean to 12.1 and nearly triples the SD to 9.9, which stretches the "normal" band to cover the 40 itself. This is called **masking**.

![Top: mean ± 3 SD (gold band) is so inflated by the 40 that it contains it. Bottom: the IQR fences (teal band) are built from quartiles, which barely notice the 40, so it lands outside.](figures/quartiles-zscore-vs-iqr.svg)

In the figure, compare the widths of the two shaded bands. Same data, same 40; only the IQR rule catches it.

| | z-score rule | IQR rule |
|---|---|---|
| Built from | mean, SD | quartiles |
| Pulled by outliers | yes, badly | barely |
| Good for | large, roughly bell-shaped data | skewed or small data, first pass in EDA |

> **Trap:** neither rule finds *errors*. They find *unusual* values. A flag is a prompt to look, not a verdict.

## What to do with an outlier

1. **Is it an error?** Impossible values: a temperature of 160°F, air pressure of 2 inches, a negative age. Jag found all of these in the accidents data. Fix it if you can, otherwise set it to missing or drop the row.
2. **Is it rare but real?** A real storm in wind-speed data. Keep it: it is information. Use the median and IQR, [log-transform](#/t/logs) the column, cap it at, say, the 99th percentile, or use a tree-based model that doesn't mind.
3. **Is it a placeholder?** Spikes at round numbers (999, 0) can be "not recorded" in disguise.
4. **Write it down.** What you flagged, which rule, what you did and why.

> **Key idea:** never drop outliers just because a rule flagged them. In fraud or [anomaly detection](#/t/anomaly), the outliers are the whole point.

Odd values in the **target label** are a separate problem: see [Label checks](#/t/labelchecks).

## Code

```python
import numpy as np
import pandas as pd
import seaborn as sns

s = pd.Series([3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 40])

q1, q3 = s.quantile([0.25, 0.75])        # 7.5, 12.5
iqr = q3 - q1                             # 5.0
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr # 0.0, 20.0

is_outlier = (s < lower) | (s > upper)    # True outside the fences
print(s[is_outlier])                      # 40
print(is_outlier.mean())                  # share of rows flagged

z = (s - s.mean()) / s.std()              # z-score rule for comparison
print(s[z.abs() > 3])                     # empty: the 40 masks itself

sns.boxplot(x=s)                          # whiskers use 1.5 x IQR by default
```

## Quick reference

| Term | Rule | Worked value |
|---|---|---|
| Q1, Q2, Q3 | 25th, 50th, 75th percentiles | 7.5, 10, 12.5 |
| IQR | Q3 − Q1 | 5 |
| Fences | Q1 − 1.5 × IQR, Q3 + 1.5 × IQR | 0 and 20 |
| z-score rule | flag beyond ±3 SDs | 40 → 2.82, not flagged |
| Masking | outliers inflate mean and SD and hide themselves | — |
| Whiskers | furthest points inside the fences | 3 and 15 |
| Outlier decision | error → fix/drop; real → keep; always write it down | — |
