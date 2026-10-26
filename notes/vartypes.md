# Variable types & their charts

The type of a variable decides which summary numbers make sense, which chart shows it honestly, and how pandas should store it.

## Why the type matters

- **Variable** = one column of a table: one thing recorded about each row.
- The type tells you **what maths is allowed**. You can average temperatures, but not US states. Severity 3 is worse than severity 2, but not "1.5 times" worse.
- Get the type wrong and you get a chart or number that looks fine but means nothing.

![A tree of variable types: categorical splits into nominal, ordinal and binary; numerical splits into discrete and continuous, each with an example from Jag's projects.](figures/vartypes-tree.svg)

In the figure, read top to bottom: first ask "label or quantity?", then the second question on that branch.

## Categorical variables (labels)

**Categorical** = the value is a label that puts the row in a group. Even if written as numbers, they are names, not amounts.

- **Nominal** = labels with **no order**. Examples: `state`, `weather_condition`, LEDGAR `label_name`. Test: can you shuffle them without losing meaning? Yes → nominal.
- **Ordinal** = labels **with an order**, but the gaps are not equal. Examples: `severity` 1–4, Jag's `light_score` 0–4 (daylight to full darkness), sizes S/M/L.
- **Binary** = exactly **two** values (yes/no, 0/1). Examples: `traffic_signal`, `is_night`.

> **Trap:** Don't take the mean of an ordinal. A mean severity of 2.2 is hard to read, because the step from 2 to 3 is not the same size as 3 to 4. Use the median or the % in each level.

> **Key idea:** Code a binary as 0/1 and its **mean is the share of 1s**. So `df["severe"].mean()` gives the severe rate directly.

## Numerical variables (quantities)

**Numerical** = the value is an amount, so differences and means make sense.

- **Discrete** = countable values with **gaps** between them. Examples: `constraint_count` (0–5), `word_count`. Test: can 1.5 exist? No → discrete.
- **Continuous** = any value in a range is possible. Examples: temperature, demand (MW), duration.

## Which summary and chart for each type

| Type | Summary | Best chart | Avoid |
|---|---|---|---|
| Nominal | counts, %, mode | bars sorted by count | line chart (suggests an order) |
| Ordinal | counts, %, median | bars in the natural order | bars sorted by height |
| Binary | % of each value | one % bar per feature | histogram of 0/1 |
| Discrete | mean, median, counts per value | one bar per integer | smooth KDE curve |
| Continuous | mean, median, SD, IQR | histogram, box plot | too few bins |

![Four panels: nominal weather as bars sorted by count, ordinal severity as bars in 1-to-4 order, continuous temperature as a histogram with a KDE curve, and discrete constraint_count as one bar per integer.](figures/vartypes-grid.svg)

In the figure, the two bar charts differ in **order**: nominal bars are sorted by height, ordinal bars stay in 1 → 4 order even though 2 is tallest. The discrete panel has gaps because no value exists between two integers.

### Labels: count plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

# nominal: most common category first
sns.countplot(data=df, y="state", order=df["state"].value_counts().index)
plt.show()

# ordinal: keep the meaningful order
sns.countplot(data=df, x="severity", order=[1, 2, 3, 4])
plt.show()

# binary: % True for each True/False column
bool_cols = df.select_dtypes(include="bool").columns
(df[bool_cols].mean() * 100).sort_values().plot(kind="barh")
plt.show()
```

The last chart is how Jag summarised the twelve road features: `traffic_signal` 14.80% True down to `roundabout` 0.0032%.

### Quantities: histogram and box plot

- **Histogram** = the range is cut into equal **bins**; each bar counts the rows in a bin.
- **KDE** = a smooth outline of the histogram.
- **Box plot** = median, the middle-50% box, whiskers and outliers. See [Quartiles, IQR & outliers](#/t/quartiles).

```python
sns.histplot(data=df, x="temperature", bins=30, kde=True)
plt.show()
sns.boxplot(data=df, x="temperature")                       # same column as a box
plt.show()
sns.histplot(data=df, x="duration_min", log_scale=True)     # skewed: log axis
plt.show()
sns.histplot(data=df, x="constraint_count", discrete=True)  # one bar per integer
plt.show()
```

The log-axis plot revealed Jag's duration spikes at about 30, 45, 60, 90, 120 and 360 minutes — default values, not measurements.

![Left: a box plot of demand that looks like one ordinary distribution. Right: the same data as a histogram, which shows two separate peaks labelled night and day.](figures/vartypes-box-hides.svg)

In the figure, the box plot gives no hint of two groups; the histogram shows them clearly. Jag found exactly this in GB demand: a low night peak and a high day peak ([elec eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)).

> **Key idea:** Look at a histogram before trusting a box plot. Two peaks can hide inside five summary numbers.

![Left: a line drawn through nominal weather categories, which wrongly suggests an order. Right: a temperature histogram with only three bins, which hides the bell shape.](figures/vartypes-avoid.svg)

In this figure, the left line wrongly suggests "cloudy" sits between "fair" and "rain". The right histogram has too few bins to show the shape.

## Storing each type in pandas

| Type | pandas dtype | How |
|---|---|---|
| Nominal | `category` | `df["state"].astype("category")` |
| Ordinal | ordered `category` | see code below |
| Binary | `bool` or 0/1 `int` | `df["severity"].isin([3, 4]).astype(int)` |
| Discrete | `int64` | `df["n"].astype("Int64")` (capital I allows blanks) |
| Continuous | `float64` | `pd.to_numeric(df["x"], errors="coerce")` |
| Date/time | `datetime64` | `pd.to_datetime(df["start_time"])` |

```python
import pandas as pd

# tell pandas the order once
light_type = pd.CategoricalDtype(["day", "twilight", "night"], ordered=True)
df["light"] = df["light"].astype(light_type)

df.sort_values("light")       # sorts day -> twilight -> night, not A-Z
df[df["light"] > "day"]       # comparisons follow the order
```

> **Trap:** Plain text sorts alphabetically, so "high" comes before "low" and charts show levels in the wrong order.

> **Say it like this:** "I first classify each column: nominal, ordinal or binary if it's a label; discrete or continuous if it's a quantity. That decides the chart — count plots for labels, histograms and box plots for quantities — and the summary: percentages for labels, median and IQR for skewed numbers."

## Quick reference

| Type | Question | Example | Chart | dtype |
|---|---|---|---|---|
| Nominal | labels, no order? | state, weather | bars sorted by count | `category` |
| Ordinal | labels with an order? | severity | bars in natural order | ordered `category` |
| Binary | only two values? | traffic_signal | % True per feature | `bool` / 0-1 |
| Discrete | countable, with gaps? | constraint_count | one bar per integer | `int64` |
| Continuous | any value in a range? | temperature | histogram, box plot | `float64` |
