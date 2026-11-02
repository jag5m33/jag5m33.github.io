# Quartiles, IQR & outliers

Quartiles answer one question: **where does the middle half of my data sit, and which values sit suspiciously far outside it?**

**On this page**

- Quartiles: cutting sorted data into quarters
- IQR: the width of the middle half
- Tukey's fences: the outlier rule
- Reading a box plot
- The z-score rule, and why it can miss
- What to do with an outlier

## Quartiles

> **Definition:** **quartiles** cut sorted data into four equal parts. **Q1** has a quarter of the data below it, **Q2** is the median, **Q3** has three quarters below it.

- Q1 is the 25th **percentile**, Q3 the 75th. In pandas: `quantile(0.25)`, `quantile(0.75)`.
- Quartiles only care about **order**. Make the biggest value a million and they don't move. That's what makes them robust.

**Worked example.** 11 values, sorted: 3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 40.

- Q2 (median): the 6th value, **10**.
- Q1: lands halfway between the 3rd and 4th values (7 and 8), so **7.5**.
- Q3: lands halfway between the 8th and 9th values (12 and 13), so **12.5**.

![The 11 sorted values with their position numbers. Follow each arrow down: Q1 and Q3 land halfway between two neighbours, the median lands exactly on the 6th value.](figures/quartiles-positions.svg)

> **Where it goes wrong:** there are several ways to compute quartiles. The textbook "median of each half" method gives Q1 = 7 and Q3 = 13 here; pandas and numpy give 7.5 and 12.5. Neither is wrong. On thousands of rows they agree almost exactly.

## IQR

> **Definition:** the **IQR (interquartile range)** is the width of the middle 50% of the data.

$$
\text{IQR} = Q3 - Q1
$$

- Q3: the 75th percentile. Q1: the 25th percentile.

**Worked example.** IQR = 12.5 − 7.5 = **5**. Compare the range: 40 − 3 = 37, controlled entirely by the 40.

The IQR is the robust partner of the median, the way the SD is the partner of the mean.

> **Say it like this:** "The middle half of the values span 5 units, from 7.5 to 12.5."

## Tukey's fences

> **Definition:** an **outlier** is a value unusually far from the rest. **Tukey's fences** are the usual rule for "unusually far": more than 1.5 IQRs beyond the box.

$$
\text{lower fence} = Q1 - 1.5 \times \text{IQR} \qquad \text{upper fence} = Q3 + 1.5 \times \text{IQR}
$$

- Anything below the lower fence or above the upper fence gets flagged.

**Worked example.**

1. 1.5 × 5 = 7.5
2. Lower fence = 7.5 − 7.5 = **0**
3. Upper fence = 12.5 + 7.5 = **20**
4. 40 is above 20 → **flagged**.

Why 1.5? Convention. On bell-shaped data it flags about 0.7% of points, but on 7.7 million rows that's about 54,000 perfectly normal points.

## Reading a box plot

> **Definition:** a **box plot** draws all of the above in one picture: box from Q1 to Q3, a line at the median, whiskers, and dots for anything past the fences.

![Box plot of the 11 values. The box runs from 7.5 to 12.5 with the median at 10. The whiskers stop at 3 and 15, the last real points inside the fences (dotted). The 40 is drawn on its own as an outlier.](figures/quartiles-boxplot.svg)

- A box pushed hard against one whisker means **skew**.
- A box squashed to a line means one value dominates.
- Box plots show rare extremes; histograms show shape. Use both.

> **Where it goes wrong:** the whiskers do **not** reach the fences. They stop at the last real data point inside them.

## The z-score rule

> **Definition:** a **z-score** is how many standard deviations a value sits from the mean. The z-score rule flags anything beyond ±3.

$$
z = (\text{value} - \text{mean}) \div \text{SD}
$$

- More in [Distributions & the CLT](#/t/distributions).

**Worked example.** Same 11 values: mean ≈ 12.09, SD ≈ 9.89. For the 40: z = (40 − 12.09) ÷ 9.89 ≈ **2.82**. Under 3, so the z-score rule says the 40 is **not** an outlier.

Why? Without the 40, the SD is about 3.7. With it, the SD nearly triples to 9.9, stretching the "normal" band wide enough to cover the 40. This is **masking**: the outlier hides itself.

![Top: mean ± 3 SD (gold band) is so stretched by the 40 that it contains it. Bottom: the IQR fences (teal band) barely notice the 40, so it lands outside.](figures/quartiles-zscore-vs-iqr.svg)

Use the IQR rule for skewed or small data and as a first pass in EDA. The z-score rule suits large, roughly bell-shaped data.

## What to do with an outlier

> **Definition:** a flag is a **prompt to look**, not a verdict. Neither rule finds errors; they find unusual values.

1. **Error?** A temperature of 160°F, a negative age. Fix it, or set it to missing.
2. **Rare but real?** A genuine storm in wind-speed data. Keep it: use the median and IQR, [log-transform](#/t/logs), or cap it at the 99th percentile.
3. **Placeholder?** Spikes at 999 or 0 can be "not recorded" in disguise.
4. **Write it down:** what you flagged, which rule, what you did and why.

> **Where it goes wrong:** never drop outliers just because a rule flagged them. In fraud or [anomaly detection](#/t/isoforest), the outliers are the whole point.

```python
s = pd.Series([3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 40])
q1, q3 = s.quantile([0.25, 0.75])           # 7.5, 12.5
iqr = q3 - q1                               # 5.0
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr   # 0.0, 20.0
print(s[(s < lower) | (s > upper)])         # 40
```

## Quick reference

| Term | In one line |
|---|---|
| Q1 / Q2 / Q3 | 25th / 50th (median) / 75th percentile |
| IQR | Q3 − Q1: width of the middle half |
| Fences | Q1 − 1.5 × IQR and Q3 + 1.5 × IQR |
| Whiskers | Stop at the last real point inside the fences |
| z-score rule | Flag beyond ±3 SDs. Can be fooled by masking |
| Masking | An outlier stretches the SD and hides itself |

**Order:** sort → Q1 and Q3 → IQR → fences → flag → decide (error, real, or placeholder).
