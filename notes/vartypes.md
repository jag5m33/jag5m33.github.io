# Variable types & their charts

The type of a variable decides which summary numbers make sense, which chart shows it honestly, and how pandas should store it.

## Why the type matters

- **Variable** = one column of a table; one thing measured or recorded about each row.
- The type tells you **what arithmetic is allowed**. You can average temperatures, but you cannot average US states. You can say severity 3 is worse than severity 2, but not that it is "1.5 times" worse.
- Get the type wrong and you get a chart or a statistic that looks fine but means nothing.

![A tree of variable types: categorical splits into nominal, ordinal and binary; numerical splits into discrete and continuous, each with an example from Jag's projects.](figures/vartypes-tree.svg)

In the figure, read top to bottom: first ask "label or quantity?", then ask the second question on that branch.

## Categorical variables (labels)

**Categorical** = the value is a label that puts the row into a group. Even if the labels are written as numbers, they are names, not amounts.

### Nominal

- **Definition:** categories with **no natural order**.
- **Examples:** `state` (CA, FL, TX), `weather_condition`, `source` (Source1, Source2, Source3), LEDGAR `label_name` (Governing Laws, Notices…).
- **Test:** can you shuffle the categories without losing meaning? Yes → nominal.

### Ordinal

- **Definition:** categories **with a natural order**, but the gaps between them are not equal or not known.
- **Examples:** `severity` 1–4 (4 is worse than 3, but not by a measured amount), Jag's `light_score` 0–4 (0 = full daylight … 4 = full darkness), t-shirt sizes S/M/L.
- **Test:** does "greater than" make sense but "minus" does not? → ordinal.

> **Trap:** An ordinal written as numbers tempts you to take the mean. A mean severity of 2.2 is hard to interpret because the step from 2 to 3 is not the same "size" as 3 to 4. Use the median, mode or the % in each level.

### Binary

- **Definition:** exactly **two** categories (yes/no, True/False, 0/1). A special case of nominal.
- **Examples:** the twelve road features like `traffic_signal`, Jag's `is_night` and `severe` (severity 3 or 4 = 1).
- **Useful fact:** if you code it as 0/1, its **mean is the proportion of 1s**. That is why `df["severe"].mean()` gives the severe rate directly.

## Numerical variables (quantities)

**Numerical** = the value is an amount, so arithmetic (differences, means) is meaningful.

### Discrete

- **Definition:** countable values, usually whole numbers, with **gaps** between possible values.
- **Examples:** number of slow-traffic features at a crash site (`constraint_count`, 0–5), number of words in a clause (`word_count`), number of rows per day (48).
- **Test:** can a value of 1.5 exist? No → discrete.

### Continuous

- **Definition:** any value in a range is possible, limited only by how precisely you measure.
- **Examples:** temperature, pressure, national demand (MW), duration in minutes.

### Interval vs ratio (briefly)

Two flavours of numerical scale, defined by what **zero** means:

| Scale | Zero means | "Twice as much" valid? | Example |
|---|---|---|---|
| Interval | an arbitrary point, not "none" | No | temperature in F or C: 60 F is not twice as hot as 30 F |
| Ratio | a true absence | Yes | demand in MW, distance, counts: 40,000 MW is twice 20,000 MW |

Differences and means work for both. Ratios ("twice", "% change") only make sense on a ratio scale.

## The decision table

| Variable type | Summary statistics | Best charts | Charts to avoid |
|---|---|---|---|
| Nominal | counts, %, mode, number of categories | bar / count plot, sorted by count | line chart (implies order), pie with many slices |
| Ordinal | counts, %, mode, median, percentiles | bar chart in the natural order | bars sorted by height (loses the order); mean as the headline |
| Binary | count and % of each value (mean of 0/1) | single bar pair, or one % per feature | histogram of 0/1 |
| Discrete | mean, median, SD, min/max, counts per value | bar chart with one bar per integer, or histogram with integer bins | KDE (smooths over impossible in-between values) |
| Continuous | mean, median, SD, IQR, percentiles, skew | histogram, KDE, box plot, violin | bar chart of every unique value; too few bins |

## The right chart for each type

![Four panels: nominal weather as bars sorted by count, ordinal severity as bars in 1-to-4 order, continuous temperature as a histogram with a KDE curve, and discrete constraint_count as one bar per integer.](figures/vartypes-grid.svg)

In the figure, notice the two bar charts differ in **how they are ordered**: nominal bars are sorted by height (nothing else gives an order), ordinal bars stay in 1 → 4 order even though 2 is the tallest. The discrete panel has gaps between bars because no value between two integers exists.

### Nominal: count plot sorted by frequency

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# categories, most common first
order = df["weather"].value_counts().index
# one bar per category, height = rows
sns.countplot(data=df, x="weather", order=order)
# long labels need rotating
plt.xticks(rotation=45, ha="right")
plt.ylabel("count")
plt.show()
```

With many categories, put them on the y-axis so labels are readable. Jag did this for states: `sns.countplot(data=df_temp, y="state", order=df_temp.state.value_counts().index)` ([eda.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb)).

### Ordinal: bar chart in the natural order

```python
# the order that carries meaning
sev_order = [1, 2, 3, 4]
sns.countplot(data=df, x="severity", order=sev_order)
plt.show()

# percentages are usually more useful than raw counts
print(df["severity"].value_counts(normalize=True).reindex(sev_order) * 100)
```

### Binary: one % per feature

```python
# True/False columns
bool_cols = df.select_dtypes(include="bool").columns
# mean of bools = share True
pct_true = (df[bool_cols].mean() * 100).sort_values()
# one bar per feature
pct_true.plot(kind="barh")
plt.xlabel("% of rows True")
plt.show()
```

This is how Jag summarised the twelve road features: `traffic_signal` 14.80% True down to `roundabout` 0.0032%.

### Continuous: histogram, KDE, box plot

- **Histogram** = the range is cut into equal-width **bins** and each bar counts the rows in that bin.
- **KDE** (kernel density estimate) = a smoothed outline of the histogram, made by placing a small bump on every point and adding them up.
- **Box plot** = the median (middle line), the IQR box (middle 50%), whiskers, and points beyond them. See [Quartiles, IQR & outliers](#/t/quartiles).

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# histogram with a KDE line on top; choose bins deliberately
sns.histplot(data=df, x="temperature", bins=30, kde=True, ax=axes[0])

# box plot of the same column
sns.boxplot(data=df, x="temperature", ax=axes[1])
plt.show()

# skewed column? use a log axis so the bulk is visible
sns.histplot(data=df, x="duration_min", bins=100, log_scale=True)
plt.show()
```

The log-axis line is what revealed Jag's duration spikes at about 30, 45, 60, 90, 120 and 360 minutes — default values, not measurements.

![Left: a box plot of demand that looks like one ordinary distribution. Right: the same data as a histogram, which shows two separate peaks labelled night and day.](figures/vartypes-box-hides.svg)

In the figure, the box plot on the left gives no hint that there are two groups. The histogram on the right shows them clearly. Jag found exactly this in GB demand: most months have two peaks, a low night one and a high day one ([elec eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)).

> **Key idea:** Always look at a histogram before trusting a box plot. A box plot summarises with five numbers, and two peaks can hide inside them.

### Discrete: one bar per integer

```python
# rows per integer value
counts = df["constraint_count"].value_counts().sort_index()
# gaps show discreteness
plt.bar(counts.index, counts.values, width=0.6)
# a tick on every integer
plt.xticks(counts.index)
plt.show()

# or a histogram whose bins are centred on integers
sns.histplot(data=df, x="constraint_count", discrete=True)
plt.show()
```

`discrete=True` makes seaborn put one bin on each integer, so bars do not straddle two values.

## Charts to avoid

![Left: a line drawn through nominal weather categories, which wrongly suggests an order. Right: a temperature histogram with only three bins, which hides the bell shape.](figures/vartypes-avoid.svg)

In the figure, the left line suggests "cloudy" sits between "fair" and "rain" and that there is a trend — neither is true. The right histogram has too few bins to show the shape at all.

## Pandas dtypes for each type

| Variable type | pandas dtype | How |
|---|---|---|
| Nominal | `category` | `df["state"] = df["state"].astype("category")` |
| Ordinal | ordered `category` | `pd.CategoricalDtype([...], ordered=True)` |
| Binary | `bool` or `int` 0/1 | `df["severe"] = df["severity"].isin([3, 4]).astype(int)` |
| Discrete | `int64` (or `Int64` if it has missing values) | `df["n"].astype("Int64")` |
| Continuous | `float64` | `pd.to_numeric(df["x"], errors="coerce")` |
| Date/time | `datetime64` | `pd.to_datetime(df["start_time"], format="ISO8601")` |

### Ordered categories in code

```python
# define the order once
light_dtype = pd.CategoricalDtype(
    ["day", "twilight", "night"], ordered=True)
df["light"] = df["light"].astype(light_dtype)

df["light"].min(), df["light"].max()      # works: uses the order
df[df["light"] > "day"]                   # comparisons follow the order
# the integer behind each label (0, 1, 2)
df["light"].cat.codes
# sorts day → twilight → night, not A–Z
df.sort_values("light")
```

A quick check on a toy series `["medium", "low", "high", "low"]` with order low < medium < high gives min `low`, max `high`, codes `[1, 0, 2, 0]`.

> **Trap:** An unordered `category` (or plain text) sorts alphabetically, so "high" comes before "low". Charts then show levels in the wrong order.

> **Trap:** `category` saves memory for text with few unique values (Jag converted `source`, `wind_direction` and `timezone`). It does not help for ID-like columns with hundreds of thousands of values.

## How to say it

> **Say it like this:** "I start by classifying each column: nominal, ordinal or binary if it's a label; discrete or continuous if it's a quantity. That decides the chart — count plots for nominal, ordered bars for ordinal, histograms and box plots for continuous — and the summary: mode and percentages for labels, median and IQR for skewed quantities."

## Quick reference

| Type | Defining question | Example | Chart | dtype |
|---|---|---|---|---|
| Nominal | labels, no order? | state, weather | count plot sorted by count | `category` |
| Ordinal | labels with an order? | severity, light_score | bars in natural order | ordered `category` |
| Binary | only two values? | traffic_signal, is_night | % True per feature | `bool` / 0-1 `int` |
| Discrete | countable, gaps between values? | constraint_count, word_count | bar per integer, `discrete=True` | `int64` / `Int64` |
| Continuous | any value in a range? | temperature, demand | histogram + KDE, box plot | `float64` |
| Interval | zero is arbitrary | temperature in F | — | — |
| Ratio | zero means none | MW, counts, distance | — | — |
