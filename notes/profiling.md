# Data profiling

Profiling means looking at every column on its own, before asking what it predicts, so you know what the data really contains.

## What profiling is

- **Profiling** = describing each column by itself: its type, gaps, repeats, spread and odd values.
- It is **not** about the target yet. The question is not "does this predict severity?" but "what is in this column, and can I trust it?".
- **Why first?** Every later step (charts, correlations, models) assumes the columns are what their names say. Profiling checks that.

> **Key idea:** Profile first, relate second. A pattern in a column you have not profiled might just be a recording error.

In his US Accidents project Jag called this **Stage 1: profile every column on its own** — "nothing here is about severity. It is about knowing the data" ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt)).

## The checklist

| Step | Question | pandas | What a problem looks like |
|---|---|---|---|
| 1. Shape | How many rows and columns? | `df.shape` | Fewer rows than expected |
| 2. Dtypes | Is each column stored as the right kind of thing? | `df.info()` | Dates or numbers stored as text (`object`) |
| 3. Missing | How many blanks per column? | `df.isna().sum()` | Several columns blank on the same rows |
| 4. Duplicates | Are any rows repeated? | `df.duplicated()` | Copies that would leak into the test set |
| 5. Cardinality | How many distinct values? | `df.nunique()` | 1 value (constant) or ~1 per row (an ID) |
| 6. Ranges | Are the numbers physically possible? | `df.describe()` | Max far beyond the 99th percentile |
| 7. Categories | What are the labels and how often? | `value_counts()` | Same thing spelt two ways ("N" and "North") |

## Shape, dtypes and memory

- **Shape** = (rows, columns).
- **Dtype** (data type) = how pandas stores a column: `int64` (whole numbers), `float64` (decimals), `bool`, `datetime64`, `category`, or `object` (text).
- **Deep memory** = memory including the text itself. Without `deep=True`, pandas only counts pointers to the text and badly under-reports.

```python
import pandas as pd

df = pd.read_csv("accidents.csv")   # load the raw file
print(df.shape)                     # (rows, columns)
df.info(show_counts=True)           # dtype + non-blank count per column
print(df.memory_usage(deep=True).sum())  # true memory, text included
```

**In his project:** 7,728,394 rows and 46 columns, with 79% of the memory hidden in text columns. So every step is slow, and plotting needs a sample.

## Missing values

- **Missing value** = a blank cell, shown by pandas as `NaN`, `None` or `NaT` (dates).
- Report the **count** and the **percentage**.

```python
missing = pd.DataFrame({
    "n_missing": df.isna().sum(),                     # count of blanks
    "pct_missing": (df.isna().mean() * 100).round(2)  # % of rows blank
}).sort_values("n_missing", ascending=False)
print(missing[missing.n_missing > 0])
```

![Missing values in the US Accidents categorical columns. The three weather columns (wine) each miss about 2.2–2.3% of rows; the other three (teal) miss far less.](figures/profiling-missing.svg)

In the figure, the three weather columns all sit around 2.2%. That made Jag suspect **one shared cause**. He traced it to `airport_code`: it is the weather station the readings came from, so no station meant no weather.

> **Trap:** Missing can hide as a real-looking value. In his GB demand data, `england_wales_demand` was **0** on three half-hours in 2018 while national demand was normal. `isna()` does not catch a 0 ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)).

## Duplicates and cardinality

- **Duplicate row** = a row identical to another. Leave out the ID column when checking, because two copies of one record get different IDs.
- **Cardinality** = the number of distinct values in a column.
  - 1 value = a **constant**. It tells the model nothing, so drop it.
  - About one value per row = an **identifier**. A model would memorise it instead of learning a pattern.

```python
cols = df.columns.drop("id")                       # ignore the unique ID
print(df.duplicated(subset=cols).sum())            # extra copies
df = df.drop_duplicates(subset=cols)               # remove them

card = df.nunique().sort_values()                  # distinct values per column
df = df.drop(columns=card[card == 1].index)        # drop constant columns
```

**In his project:** 140,811 rows were extra copies — almost all **pairs**, so double submission rather than a broken feed. Remove them **before** the train/test split, or the model gets tested on rows it has already seen.

![Cardinality of selected US Accidents columns on a log scale: country and turning_loop have one value, source has three, while street has 336,306 and zipcode 825,294.](figures/profiling-cardinality.svg)

In the figure, both ends are a problem. `country` (all USA) and `turning_loop` (all False) were dropped. `zipcode` had 825,294 values because it mixed 5-digit and ZIP+4 formats, so it was dropped too. The 336,306 street names were reduced to 8 road classes.

## Ranges and impossible values

- **Range** = min to max. If the max is far beyond the 99th percentile, look closer.
- **Impossible value** = something the real world cannot produce (a pressure of 2 inHg). An **outlier** is different: extreme, but real.

![A temperature histogram with a few planted values at -90 F and 160 F, inside shaded regions beyond the physical limits of -70 to 134 F.](figures/profiling-impossible.svg)

In the figure, the bulk is a normal hump and the few bars in the shaded areas are impossible. Jag found exactly these, -90 F and 160 F. They are impossible in Fahrenheit **and** Celsius, so they are errors, not a unit mix-up.

> **Key idea:** With 7.7 million rows the IQR rule flags millions of normal points, so Jag judged extremes against **physical limits** instead. See [Quartiles, IQR & outliers](#/t/quartiles).

> **Trap:** A spike can be real but **capped**. Visibility had a spike at exactly 10 miles because stations report "10 or more" as 10.

## Categories

```python
# share of each label in %, with blanks shown as their own row
print(df["wind_direction"].value_counts(normalize=True, dropna=False) * 100)
```

What to look for:
- **Same thing spelt two ways.** `wind_direction` had "North" and "N"; `timezone` had 7 values where there should be 4.
- **Long tails.** `weather_condition` had 144 values, but the top 13 covered 95% of rows.
- **Imbalance in the target.** Severity was about 80% class 2, so accuracy would be misleading (see [Label checks](#/t/labelchecks)).

> **Say it like this:** "Before modelling I profiled every column on its own: types, missing values, duplicates, cardinality, ranges against physical limits and value counts. In the accidents data that found two constant columns, impossible temperatures, 140,811 duplicate copies and labels spelt two ways."

## Traps

- **Sample for pictures, not numbers.** A 100,000-row sample is fine to see a shape, not for reported figures.
- **Re-profile after filtering.** After Jag kept only one data source, precipitation went from 28.5% to 47.0% missing.
- **Don't trust column names.** `airport_code` looked like location but was the weather station.

## Quick reference

| Term | Meaning | Code |
|---|---|---|
| Shape | rows, columns | `df.shape` |
| Dtype | storage type of a column | `df.dtypes` |
| Missing % | share of blanks | `df.isna().mean()*100` |
| Duplicates | repeated rows (ignore ID) | `df.duplicated(subset=cols)` |
| Cardinality | number of distinct values | `df.nunique()` |
| Constant column | one value only, no information | drop it |
| Impossible value | outside physical limits | compare with set limits |
| value_counts | label frequencies | `s.value_counts(normalize=True)` |
