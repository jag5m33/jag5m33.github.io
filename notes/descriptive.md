# Descriptive statistics

Descriptive statistics answer one question about a column: **where is the middle, how spread out is it, and which way does it lean?**

**On this page**

- Population vs sample: what you have and what you want
- The centre: mean, median, mode
- Why the median is often the honest one
- The spread: range and standard deviation
- Skew: which way the data lean
- Quick reference

## Population vs sample

> **Definition:** the **population** is everyone you care about. The **sample** is the part you actually measured.

- Population: Greek letters, mean μ, SD σ. Sample: ordinary letters, mean x̄, SD s, size n.
- You almost never have the whole population, so sample numbers stand in as estimates.

**Descriptive** statistics summarise the data you have. **Inferential** statistics ([Confidence intervals](#/t/ci), [Hypothesis testing](#/t/hypothesis)) use them to say something about the population.

Every example below uses these 8 values: **2, 4, 4, 4, 5, 5, 7, 9**.

## The centre: mean, median, mode

> **Definition:** the **mean** is the average: add everything up, divide by how many there are. The **median** is the middle value once the data are sorted. The **mode** is the most common value.

**Worked example.**

- Mean: 2 + 4 + 4 + 4 + 5 + 5 + 7 + 9 = 40, and 40 ÷ 8 = **5**.
- Median: 8 values, so average the 4th and 5th: (4 + 5) ÷ 2 = **4.5**.
- Mode: 4 appears three times, so **4**.

The mean is the **balance point**: the distances below it and above it cancel out exactly.

![The eight values as dots, with the mean (gold) and median (blue dashed). Each horizontal line is one value's distance from the mean. Add the signed numbers: −3 −1 −1 −1 +0 +0 +2 +4 = 0.](figures/descriptive-deviations.svg)

The mode is the only one that works for **categories** ("the most common accident severity is 2").

## When the median is the honest one

> **Definition:** a summary is **robust** if one extreme value can't drag it far. The median is robust; the mean is not.

Eight people in an office. Salaries in £ thousands: 24, 26, 27, 28, 30, 31, 32, 150.

- Mean = 348 ÷ 8 = **43.5**
- Median = (28 + 30) ÷ 2 = **29**

![One director's salary drags the mean (gold) to £43.5k, more than seven of the eight people earn. The median stays at £29k, where people actually are.](figures/descriptive-salaries.svg)

The median only cares about order, not how extreme the extremes are. Same logic for missing values: fill skewed columns with the median.

> **Say it like this:** "Salaries are right-skewed by a few high earners, so I report the median as the typical value. The mean is pulled up by the tail."

## The spread: range and standard deviation

> **Definition:** the **range** is largest minus smallest. The **standard deviation (SD)** is roughly how far a typical value sits from the mean, in the data's own units.

Range: 9 − 2 = **7**. Quick, but one outlier controls it completely. The [IQR](#/t/quartiles) is the robust alternative.

**SD in words:** find each value's distance from the mean, square them, average them, then square-root the answer.

$$
s = \sqrt{\text{sum of (value − mean)}^2 \div (n - 1)}
$$

- *value − mean*: each value's distance from the mean. *n*: how many values.

Why square? Plain distances cancel to zero (see the dots figure). Squaring makes them positive; the square root puts you back in the original units.

**Worked example** (mean = 5):

1. Distances: −3, −1, −1, −1, 0, 0, 2, 4
2. Squared: 9, 1, 1, 1, 0, 0, 4, 16. Sum = **32**
3. Divide by n − 1 = 7: about **4.57** (this is the **variance**)
4. Square root: **SD ≈ 2.14**

> **Say it like this:** "The values typically sit about 2.1 units from the mean of 5."

> **Where it goes wrong:** pandas and numpy disagree. `s.std()` divides by n − 1 (gives 2.14); `np.std(s)` divides by n (gives 2.0). Set `ddof=1` to be sure you get the sample SD.

> **Where it goes wrong:** the SD is the spread of **individual values**. It does not shrink with a bigger sample. The spread of your **estimate** is the standard error, a different thing: see [Distributions & the CLT](#/t/distributions).

## Skew

> **Definition:** **skew** is lopsidedness. It's named after the side with the **longer tail**.

- **Right skew:** tail to the right, towards big values. Mean > median. Salaries, house prices, durations.
- **Symmetric:** mean ≈ median.
- **Left skew:** tail to the left. Mean < median.

![Three shapes. In each skewed one the mode sits at the peak, the mean (gold) is dragged furthest into the tail, and the median sits between them.](figures/descriptive-skew.svg)

Our 8 values have skew about 0.82: mildly right, matching mean 5 > median 4.5. Rough guide: under 0.5 (either sign) fairly symmetric, 0.5 to 1 moderate, over 1 strong. For strong right skew, report the median and consider a [log transform](#/t/logs).

> **Where it goes wrong:** skew is named after the tail, **not** the peak. A right-skewed shape has its hump on the left. Humidity in the accidents data (skew −0.72) piles up at 90–100% with a thin tail down to 0, so it is **left**-skewed.

```python
s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])
s.describe()                       # count, mean, std, quartiles, min, max
s.mean(), s.median(), s.mode()[0]  # 5.0, 4.5, 4
s.std()                            # 2.14  sample SD (divides by n - 1)
s.skew()                           # 0.82  positive: mild right skew
```

## Quick reference

| Term | In one line |
|---|---|
| Mean | Add up, divide by how many. Pulled by extremes |
| Median | Middle of the sorted data. Robust |
| Mode | Most common value. Works for categories |
| Range | Max − min. One outlier controls it |
| Variance | Average squared distance from the mean |
| SD | Square root of the variance: typical distance from the mean |
| Skew | Named after the tail. Mean is dragged towards it |

**Order:** mean → distances from the mean → square → sum → ÷ (n − 1) → square root = SD.
