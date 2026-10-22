# Data profiling

Profiling means looking at every column on its own, before asking what it predicts, so that you know what the data really contains.

## What profiling is (and is not)

- **Profiling** = describing each column by itself: its type, size, gaps, repeats, spread and odd values.
- It is **not** about the target yet. You are not asking "does this predict severity?". You are asking "what is in this column, and can I trust it?".
- **Why first?** Every later step (charts, correlations, models) quietly assumes the columns are what their names say. Profiling checks that assumption.

> **Key idea:** Profile first, relate second. A relationship found in a column you have not profiled might just be a recording error.

In his US Accidents project Jag called this **Stage 1: profile every column on its own**, with the rule "nothing here is about severity. It is about knowing the data" ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt)).

## The profiling checklist

Work down this table in order. Each row has a code section below.

| Step | Question | pandas | What a problem looks like |
|---|---|---|---|
| 1. Shape | How many rows and columns? | `df.shape` | Fewer rows than expected = a file did not load fully |
| 2. Dtypes | Is each column stored as the right kind of thing? | `df.dtypes`, `df.info()` | Dates or numbers stored as text (`object`) |
| 3. Memory | How heavy is it? | `df.memory_usage(deep=True)` | Most memory in text columns |
| 4. Missing | How many blanks per column? | `df.isna().sum()`, `.mean()*100` | Several columns missing on the same rows |
| 5. Duplicates | Are any rows repeated? | `df.duplicated(subset=...)` | Exact copies that would leak into test |
| 6. Cardinality | How many distinct values? | `df.nunique()` | 1 value (constant) or ~1 per row (an ID) |
| 7. Ranges | Are the numbers physically possible? | `df.describe()`, `.quantile()` | Max far beyond the 99th percentile |
| 8. Categories | What are the labels and how often? | `value_counts(dropna=False)` | Same thing spelt two ways ("N" and "North") |

## Steps 1–3: shape, dtypes and memory

- **Shape** = (rows, columns).
- **Dtype** (data type) = how pandas stores a column: `int64` (whole numbers), `float64` (decimals; the default numeric type that can hold `NaN` — whole numbers with gaps need the nullable `Int64`), `bool`, `datetime64`, `category`, and `object` / `str` (text).
- **Memory, shallow vs deep.** Shallow memory counts only the pointers to text; `deep=True` also counts the text itself. The gap is the "hidden" text weight.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("accidents.csv")   # load the raw file

print(df.shape)               # (rows, columns)
df.info(show_counts=True)     # dtype + non-null count
print(df.index)               # RangeIndex = a row counter

# split column names by kind so each group can be profiled separately
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

# shallow vs deep memory: how much is hidden inside text columns?
shallow = df.memory_usage(deep=False).sum()  # pointers only
deep = df.memory_usage(deep=True).sum()      # + the strings
print(f"{(deep - shallow) / deep * 100:.0f}% of memory is hidden in text")
```

**In his project:** 7,728,394 rows and 46 columns; 79% of the memory was hidden in text columns; the index was a plain RangeIndex. The meaning he drew: every operation is slow, most of the weight is in text columns that will be dropped or converted, and plotting will need a sample.

> **Trap:** A tiny toy table can show the same effect. An 8-row table with three text columns measured 516 bytes shallow but 1,547 bytes deep (pandas 3.0). Never judge memory without `deep=True`.

## Step 4: missing values (counts and %)

- **Missing value** = a blank cell, shown by pandas as `NaN` (numbers), `None` or `NaT` (dates).
- Report both the **count** (how many rows) and the **percentage** (how big a share). Counts matter for rare columns; percentages matter for deciding whether to drop.

```python
missing = pd.DataFrame({
    "n_missing": df.isna().sum(),                    # count of blanks
    "pct_missing": (df.isna().mean() * 100).round(2)  # % missing
}).sort_values("n_missing", ascending=False)
print(missing[missing.n_missing > 0])

# do the gaps fall on the SAME rows? (one cause, not many)
weather = ["weather_condition", "wind_direction", "weather_timestamp"]
print(df[weather].isna().all(axis=1).sum())  # rows missing all 3
```

![Missing values in the US Accidents categorical columns. The three weather columns (wine) each miss about 2.2–2.3% of rows; the other three (teal) miss far less.](figures/profiling-missing.svg)

In the figure, look at the colour split: the weather columns all sit around 2.2%, which is what made Jag suspect **one shared cause** rather than separate problems. He later traced it to `airport_code`: it is the weather station the readings were copied from, so if no station matched, no weather was copied in.

> **Trap:** Missing can hide as a real-looking value. In his GB demand notebook, `england_wales_demand` was **0** on three half-hours in 2018 while national demand was normal. `isna()` does not catch a 0 — he called this "missing values recorded as 0" ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)).

## Step 5: duplicates

- **Duplicate row** = a row identical to another on every column you care about. Usually you exclude the ID column, because two copies of one record get different IDs.
- `keep=False` marks **every** member of a duplicate group; `keep="first"` marks only the **extra copies**.

```python
check_cols = df.columns.drop("id")   # ignore the unique ID

# keep=False: every member of a group; keep="first": extra copies
in_groups = df.duplicated(subset=check_cols, keep=False).sum()
extras = df.duplicated(subset=check_cols, keep="first").sum()
print(in_groups, extras, in_groups - extras)  # last = distinct
```

**In his project:** 274,144 rows sat in duplicate groups and 140,811 were extra copies, so 274,144 − 140,811 = 133,333 distinct records lay behind them. Average group size 274,144 / 133,333 = 2.06, so almost all were **pairs** — double submission rather than a broken feed. They must be removed **before** the train/test split, or the model is tested on a row it has already memorised.

## Step 6: cardinality

- **Cardinality** = number of distinct values in a column (`nunique()`).
- **Low extreme:** 1 value = a **constant**. It carries zero information.
- **High extreme:** almost one value per row = an **identifier**. A model would memorise individual values instead of learning a pattern.

```python
card = df[cat_cols].nunique().sort_values()  # distinct values
print(card)

constant = card[card == 1].index.tolist()  # single-value columns
df = df.drop(columns=constant)             # constants carry no info
```

![Cardinality of selected US Accidents columns on a log scale: country and turning_loop have one value, source has three, while street has 336,306 and zipcode 825,294.](figures/profiling-cardinality.svg)

In the figure, both ends are a problem. `country` (100% USA) and `turning_loop` (100% False) were dropped. `zipcode` had 825,294 unique values against roughly 42,000 real US zipcodes, because it mixed 5-digit and ZIP+4 formats; it was dropped, and 336,306 street names were reduced to 8 road classes instead.

## Step 7: ranges and impossible values

- **Range** = min to max. Compare `max` with the 75th and 99th percentiles: if the max is far beyond them, look closer.
- **Impossible value** = a value the physical world cannot produce (a pressure of 2 inHg, a 820 mph wind). Different from an **outlier**, which is extreme but real.

```python
# extra percentiles show whether the max is a tail or a single broken value
print(df[num_cols].describe(percentiles=[0.5, 0.75, 0.99, 0.9999]).T)

# skew: how lopsided each column is (0 = symmetric)
print(df[num_cols].skew().round(2).sort_values())

# check against physical limits written down explicitly
limits = {"temperature(F)": (-70, 134), "pressure(in)": (20, 32)}
for col, (lo, hi) in limits.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()  # out of range
    print(col, bad)
```

![A temperature histogram with a few planted values at -90 F and 160 F, inside shaded regions beyond the physical limits of -70 to 134 F.](figures/profiling-impossible.svg)

In the figure, the bulk of the data is a normal hump; the few bars in the shaded areas are the impossible ones. Jag found exactly these, -90 F and 160 F, and noted that they are impossible in Fahrenheit **and** Celsius, so they are errors rather than a unit mix-up.

> **Key idea:** At 7.7 million rows the IQR rule flags millions of legitimate points, so Jag judged extremes against **physical limits**, not boxplot whiskers. See [Quartiles, IQR & outliers](#/t/quartiles).

> **Trap:** A spike can be real but **censored** (capped). Visibility had a spike at exactly 10 miles because airport stations report "10 or more" as 10.

## Step 8: value_counts for categories

```python
# share of each label, keeping NaN visible as its own row
print(df["wind_direction"].value_counts(normalize=True, dropna=False) * 100)

# cumulative coverage: how many labels cover 95% of rows?
cov = df["weather_condition"].str.lower().str.strip() \
        .value_counts(normalize=True).cumsum()
print((cov <= 0.95).sum() + 1)   # number of labels needed to reach 95%
```

What to look for:
- **Same thing spelt two ways.** `wind_direction` had "North" and "N"; `timezone` had 7 values where there should be 4. Jag noted it was the third time and that "a pattern is now visible": the dataset mixes vocabularies.
- **Long tails.** `weather_condition` had 144 distinct values after stripping; the top 13 covered 95% of rows, and the long tail of roughly 130 others shared the last 5%.
- **Imbalance in the target.** Severity was about 80% class 2, so accuracy would be a useless metric (more in [Label checks](#/t/labelchecks)).

## A one-function profile

Putting steps 2, 4 and 6 into one table you can print for any dataset:

```python
def profile(df):
    """One row per column: dtype, missing, unique values, an example."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        # first non-missing value, so you can see what the column holds
        "example": df.apply(lambda s: s.dropna().iloc[0]
                            if s.notna().any() else None),
    })

print(profile(df))
```

## How to say it

> **Say it like this:** "Before modelling I profiled every column on its own: shape and dtypes, deep memory, missing counts and percentages, duplicates, cardinality, ranges against physical limits, and value counts. In the accidents data that found two constant columns, ID-like location columns, impossible temperatures, 140,811 duplicate copies and three columns with the same thing spelt two ways."

## Traps

- **Sampling for statistics.** A 100,000-row random sample is fine for seeing a distribution's shape; it must not be used for reported numbers.
- **Profiling after splitting the target.** Do it on the whole raw table first, then again after any restriction (Jag re-profiled missingness after keeping only Source2, and precipitation went from 28.5% to 47.0% missing).
- **Trusting the column name.** `airport_code` looked like geography but was the weather station: provenance, not location.

## Quick reference

| Term | Meaning | Code |
|---|---|---|
| Shape | rows, columns | `df.shape` |
| Dtype | storage type of a column | `df.dtypes` |
| Deep memory | includes the text itself | `df.memory_usage(deep=True)` |
| Missing % | share of blanks | `df.isna().mean()*100` |
| Duplicates | repeated rows (ignore ID) | `df.duplicated(subset=cols, keep="first")` |
| Cardinality | number of distinct values | `df.nunique()` |
| Constant column | cardinality 1, no information | drop it |
| Impossible value | outside physical limits | compare with explicit limits |
| Censored value | real but capped (visibility = 10) | look for spikes at round values |
| value_counts | label frequencies | `s.value_counts(normalize=True, dropna=False)` |
