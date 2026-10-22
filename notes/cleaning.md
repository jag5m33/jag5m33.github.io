# Data cleaning

Data cleaning turns the raw table into one a model can learn from honestly, by fixing types, handling gaps and repeats, standardising text, removing leakage and writing down every decision.

## Order of work

Profiling ([Data profiling](#/t/profiling)) finds the problems; cleaning fixes them. Work in this order, because later steps depend on earlier ones:

| Step | Why here |
|---|---|
| 1. Merge / concatenate files | everything after works on one table |
| 2. Fix dtypes | dates and numbers must be real dates and numbers before any check |
| 3. Normalise strings | so "Clear" and "clear " count as one value |
| 4. Remove duplicates | **before** splitting, so no copy lands in both train and test |
| 5. Impossible values → missing | so they are treated as gaps, not as data |
| 6. Remove leakage columns | before any model sees them |
| 7. Split train / validation / test | |
| 8. Impute missing values | **fitted on train only** |
| 9. Save (parquet) | checkpoint the clean version |

Throughout: keep a **cleaning log**.

## Merging and concatenating files with different columns

- **Concatenate** (`pd.concat`) = stack tables **on top of each other** (more rows). Used when files are the same kind of data split by year or batch.
- **Merge** (`pd.merge`) = join tables **side by side** on a key column (more columns). Used when files hold different information about the same rows.

`pd.concat` matches columns **by name**. A column that one file lacks becomes `NaN` for that file's rows.

![Diagram of two yearly tables stacked with pd.concat. The 2018 rows have NaN in nsl_flow and viking_flow because those columns did not exist in the 2018 file.](figures/cleaning-concat.svg)

In the figure, the wine cells are blanks that mean "this column did not exist yet", not "the reading failed". You must remember the difference when you handle missing values later.

**From his GB demand notebook** ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)): six yearly NESO files, 2018–2023. A column-comparison grid showed 2019–2022 matched; 2018 and 2023 differed. He added the missing columns as `NaN`, tagged each row with its source file as a check, and stacked them.

```python
import numpy as np
import pandas as pd

# 1. load each year (parquet is much faster than CSV once saved)
dfs = {y: pd.read_csv(f"../data/raw/demanddata_{y}.csv")
       for y in range(2018, 2024)}

# 2. which files have the same columns? 1 = same columns, same order
frames = list(dfs.values())
grid = pd.DataFrame([[int(a.columns.equals(b.columns)) for b in frames]
                     for a in frames],
                    index=list(dfs), columns=list(dfs))
print(grid)

# 3. give every file every column, record where rows came from
all_cols = list(dfs[2023].columns)            # 2023 has every column
for y, d in dfs.items():
    d["df_source"] = y                        # a check, removed after
    dfs[y] = d.reindex(columns=all_cols + ["df_source"])  # adds NaN cols

# 4. stack, renumbering the index 0..n-1
df = pd.concat(dfs.values(), axis=0, ignore_index=True)
print(df.groupby("df_source").size())         # rows per file, as expected?
df = df.drop(columns="df_source")
```

**His check afterwards:** 17,520 rows per year (365 × 48) and 17,568 in 2020 (leap year), 105,168 in total. Blank cells appeared only in the columns that didn't exist in some files: `scottish_transfer` (only in 2023) and `nsl_flow`, `eleclink_flow`, `viking_flow`, `greenlink_flow` (missing in 2018).

```python
# merge: add columns from another table, matched on a key
weather = pd.read_csv("weather.csv")          # one row per station per hour
df = df.merge(weather, on=["station", "hour"], how="left",
              validate="many_to_one")         # error if keys are duplicated
# how many rows found no match?
print(df["temp"].isna().mean())
```

> **Trap:** `how="left"` keeps every row of the left table; unmatched rows get `NaN`. `validate="many_to_one"` stops a duplicated key on the right from silently multiplying your rows.

## Fixing dtypes

```python
# text dates -> real datetimes; anything unparseable becomes NaT
df["start_time"] = pd.to_datetime(df["start_time"], format="ISO8601",
                                  errors="coerce")

# text numbers -> floats; "N/A", "--" etc. become NaN instead of crashing
df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")

# few distinct text values -> category (saves memory, keeps the labels)
df["source"] = df["source"].astype("category")

# True/False -> 0/1 for models
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

print(df.dtypes)
```

- `errors="coerce"` = "if you can't convert it, make it missing". Check how many values it turned into `NaN` — a sudden jump means a format you didn't expect.
- Give `to_datetime` a `format`. Without one, Jag's notebook warned "Could not infer format, so each element will be parsed individually".
- On a toy column `["29.9", "30.1", "N/A"]`, `to_numeric(errors="coerce")` gives `[29.9, 30.1, NaN]` as `float64`.

> **Trap:** Time zones and clock changes. Jag's NESO data had 46 periods on the March clock-change day and 50 in October, so he built a continuous GMT timestamp instead: `pd.date_range(start="2018-01-01 00:00", periods=len(df), freq="30min", tz="Etc/GMT")` after sorting by date and period.

## Missing values

### Why is it missing? (MCAR, MAR, MNAR in plain words)

| Type | Plain words | Example | Safe to drop? |
|---|---|---|---|
| **MCAR** — missing completely at random | the gap has nothing to do with anything; like a coin flip | a random server hiccup loses some rows | yes, only costs sample size |
| **MAR** — missing at random | the gap depends on **other columns you have**, not on the missing value itself | weather missing when no airport station matched (remote sites) | dropping biases toward the other group; impute using those columns |
| **MNAR** — missing not at random | the gap depends on **the missing value itself** | a sensor that fails in extreme cold, so the coldest readings vanish | neither dropping nor simple imputation fixes it; flag it and say so |

![Three panels of temperature by site type, with recorded points in teal and missing points as wine circles. MCAR: missing points scattered everywhere. MAR: missing points almost all in the remote row. MNAR: missing points are exactly the coldest temperatures.](figures/cleaning-mcar.svg)

In the figure, look at **where** the wine circles fall. Scattered everywhere → MCAR. Concentrated in one group of another column → MAR. Concentrated at one end of the variable itself → MNAR.

How to tell MCAR from MAR in practice: compare rows with and without the gap.

```python
# does missingness line up with another column?
is_missing = df["weather_condition"].isna()
# share of rows with no airport_code, split by weather missing / present
print(df.groupby(is_missing)["airport_code"].apply(
    lambda s: s.isna().mean()))

# does missingness line up with the TARGET? (from his dcdm notebook)
for col in df.columns[df.isna().any()]:
    pct = df.groupby(df[col].isna())["severity"].apply(
        lambda s: s.isin([3, 4]).mean() * 100)
    print(col, pct.round(1).to_dict())        # {False: %, True: %}
```

Jag's version of the second check found that accidents with no weather reading were 2.5–11.4 points more likely to be severe, so he kept a `weather_missing` flag rather than throwing the information away ([py_acc README](https://github.com/jag5m33/py_acc/blob/main/README.md)).

### The options

**1. Drop rows or columns**

```python
# drop rows missing a key column
df = df.dropna(subset=["light_score"])
# drop a column that is mostly empty
df = df.drop(columns=["precipitation(in)"])
```

Jag dropped `precipitation` (47% missing in Source2) and the rows missing `light_score`; the 23,246 rows missing all light values (0.3%) were also missing several other columns — "one problem affecting 23,246 rows".

**2. Impute a single value: mean, median or mode**

- **Impute** = fill a gap with an estimate.
- Mean for roughly symmetric numbers, **median** for skewed numbers, **mode** (most common value) for categories.

```python
df["humidity"] = df["humidity"].fillna(df["humidity"].median())
df["wind_direction"] = df["wind_direction"].fillna(
    df["wind_direction"].mode()[0])           # mode() returns a Series
# or keep "unknown" as its own category instead of guessing
df["wind_direction"] = df["wind_direction"].fillna("NA")
```

**3. Impute by group**

Fill with the median of **similar rows**, not of everyone.

```python
# a missing temperature in Minnesota should not get California's median
df["temp_filled"] = df["temp"].fillna(
    df.groupby("state")["temp"].transform("median"))
```

On a toy table with CA temps 70, 72, NaN and MN temps 20, NaN, 24: the CA gap becomes 71 and the MN gap 22. A global median would have put 47 in both — wrong for each state.

**4. Flag with an indicator column**

```python
df["temp_missing"] = df["temp"].isna().astype(int)   # 1 = was missing
df["temp"] = df["temp"].fillna(df["temp"].median())   # then fill
```

The model can then learn whether "missing" itself carries signal (as `weather_missing` did).

### SimpleImputer: fit on train only

**Fit on train only** = compute the fill value (e.g. the median) from training rows, then apply that **same** value to validation and test. If you compute it on all rows, information from the test set leaks into training.

```python
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X = df.drop(columns="severe"); y = df["severe"]
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42)   # 0.25 x 0.8 = 0.2

cols = ["temperature(f)", "humidity(%)", "pressure(in)",
        "visibility(mi)", "wind_speed(mph)"]
imputer = SimpleImputer(strategy="median").set_output(transform="pandas")

# learn medians HERE only
X_train[cols] = imputer.fit_transform(X_train[cols])
X_val[cols] = imputer.transform(X_val[cols])          # reuse train medians
X_test[cols] = imputer.transform(X_test[cols])
# the medians it learned
print(imputer.statistics_)
```

This is exactly the 60/20/20 split and median imputation in his [dcdm.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/dcdm.ipynb). `SimpleImputer(add_indicator=True)` also adds a `missingindicator_<col>` 0/1 column for you.

## Duplicates

```python
check_cols = df.columns.drop("id")                    # ignore the unique ID
print(df.duplicated(subset=check_cols).sum())         # extra copies

# where do they come from? (source, year)
dups = df[df.duplicated(subset=check_cols, keep="first")]
print(dups[["source", "start_year"]].value_counts())

df = df.drop_duplicates(subset=check_cols, keep="first")
```

In Source2 he removed 3,581 exact duplicates (3,305,373 → 3,301,792 rows) before the split. For time series, duplicates are "two rows for the same timestamp": `df.duplicated(subset=["settlement_date", "settlement_period"])` found none in the NESO data. For text, also check conflicting labels — see [Label checks](#/t/labelchecks).

## Impossible values → missing

```python
limits = {"temperature(f)": (-70, 134), "humidity(%)": (0, 100),
          "pressure(in)": (20, 32), "wind_speed(mph)": (0, 150)}

for col, (lo, hi) in limits.items():
    outside = (df[col] < lo) | (df[col] > hi)
    print(col, outside.sum())                 # log how many you change
    df[col] = df[col].mask(outside)           # mask() puts NaN where True
```

These are the physical limits from his dcdm notebook. They then get imputed like any other gap. Hidden missing values need the same treatment: `england_wales_demand == 0` on three half-hours was a recording error, not real demand.

## String normalisation

```python
s = df["weather_condition"]
s = s.str.strip()                             # " Clear" -> "Clear"
s = s.str.lower()                             # "Clear" -> "clear"
s = s.str.replace(r"\s+", " ", regex=True)    # collapse double spaces
df["weather_condition"] = s

# map different spellings of the same thing to one label
wind_map = {"north": "N", "n": "N", "west": "W", "w": "W",
            "calm": "CALM", "variable": "VAR", "var": "VAR"}
d = df["wind_direction"].str.strip().str.lower()
df["wind_direction"] = d.map(wind_map).fillna(d.str.upper())

print(df["wind_direction"].value_counts(dropna=False))   # check the result
```

Jag's raw `wind_direction` had "North" and "N", "West" and "W"; after merging to 16 compass points, CALM was the most common value (17.21%). `timezone` had 7 values for 4 zones.

> **Trap:** Prefer `map` with a dictionary over `replace(..., regex=True)` for whole labels. A regex replace of "North" → "N" also hits "Northeast", and a pattern like "var" can match inside other words. His notes record VAR and VARIABLE still split after the first pass.

### Merging labels only when justified

"clear" and "fair" look like two words for the same sky, so merging them seems harmless. But in the accidents data they differed in % severe by 21.8 points because they marked **different years** (the weather supplier switched wording at the end of 2019), not different weather. Merging would have hidden the clue that exposed the drifting label.

**Rule:** merge two labels only when (a) they mean the same thing **and** (b) you have checked that they don't differ on anything else — year, source, target rate. Write the check in the cleaning log.

## Removing leakage columns

- **Leakage** = a column that would not be available at prediction time, or that is really a measurement of the target. It makes validation scores look great and real use fail.
- **Test** (Jag's question): "would a dispatcher have this number at the moment the crash happens?"

```python
leaky = ["end_time", "time_delta", "end_lat", "end_lng",
         "distance(mi)", "description"]
df = df.drop(columns=leaky)
```

His five leakage groups all "measure the SIZE OF THE DISRUPTION, in time or in space… Severity IS the size of the disruption." He kept them only to **validate** the label, never as inputs. Other kinds: ID-like columns, columns derived from the target (`severity` once `severe` exists), and near-copies of the target (TSD and England & Wales demand are versions of ND, "so using them would show the model the answer").

## Saving to parquet

```python
df.to_parquet("../data/df_cleaned.parquet", engine="pyarrow", index=False)
# dtypes come back intact
df = pd.read_parquet("../data/df_cleaned.parquet")
```

- **Parquet** = a compressed, column-based file format. Smaller and much faster than CSV, and it **keeps dtypes** (dates stay dates, categories stay categories). A CSV turns everything back into text.
- Save a checkpoint after each stage so a notebook can restart from there, as Jag does ("Checkpoint: the merged table is saved as `df_merged.parquet`").

## The cleaning log habit

Record **every** decision as you make it: what, how many rows, why, and the evidence.

```python
log = []                                       # one dict per decision

def record(step, detail, rows_before, rows_after, reason):
    log.append(dict(step=step, detail=detail, rows_before=rows_before,
                    rows_after=rows_after, reason=reason))

n0 = len(df)
df = df.drop_duplicates(subset=check_cols)
record("duplicates", "drop exact copies (all cols except id)", n0, len(df),
       "pairs = double submission; must go before split")

pd.DataFrame(log).to_csv("../data/cleaning_log.csv", index=False)
```

His notes.txt is a long-form version of this: every step has TESTED, RESULT, MEANING, NEXT, and a DECISION where something changed ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt)). It is what lets a reviewer (or you, three months later) trust the final table.

## Caveats

| Action | Hidden cost |
|---|---|
| Mean / median imputation | **shrinks variance** and weakens correlations; every filled row sits at one value |
| Dropping rows with gaps | can **bias** the sample if the gaps are MAR / MNAR (e.g. losing remote sites) |
| Dropping columns | loses information; check whether missingness itself predicts the target first |
| Imputing before the split | **leaks** test information into training |
| Group imputation | needs enough rows per group; tiny groups give noisy fill values |
| Indicator columns | add features; useful only if missingness carries signal |
| Merging labels | can hide a real difference (clear / fair) |
| Restricting to a subset (Source2) | changes **coverage**: Source2 had 9 states almost absent and no 2023 |
| IQR rule for outliers on huge data | flags millions of real values; use physical limits instead |
| `errors="coerce"` | silently turns bad formats into NaN; count them |

![Two histograms of the same temperature column. Left: observed values only, SD about 14.8. Right: after filling 323 missing rows with the median, a tall spike at the median and SD about 12.1.](figures/cleaning-impute.svg)

In the figure, the right-hand histogram has a tall spike at the median: 323 of 1,000 rows were filled with one value, so the spread drops from about 14.8 to 12.1. On a toy series of 8 values plus 4 gaps, the SD fell from 6.41 to 5.12 after median filling — same effect.

## How to say it

> **Say it like this:** "I merged the yearly files by column name and checked row counts per year; fixed dtypes with coerce and counted what it changed; normalised strings; dropped duplicates before splitting; turned physically impossible values into missing; removed leakage columns that are only known after the event; split 60/20/20; and imputed medians fitted on the training set only, keeping a flag where missingness predicted the target. Every step is in a cleaning log."

## Quick reference

| Task | Code |
|---|---|
| Stack files | `pd.concat(list_of_dfs, ignore_index=True)` |
| Join on a key | `df.merge(other, on="key", how="left", validate="many_to_one")` |
| Text → date | `pd.to_datetime(s, format="ISO8601", errors="coerce")` |
| Text → number | `pd.to_numeric(s, errors="coerce")` |
| Category | `s.astype("category")` |
| MCAR / MAR / MNAR | random / depends on other columns / depends on itself |
| Drop | `dropna(subset=[...])`, `drop(columns=[...])` |
| Fill | `fillna(s.median())`, `fillna(s.mode()[0])` |
| Fill by group | `fillna(df.groupby(g)[c].transform("median"))` |
| Flag | `df[c + "_missing"] = df[c].isna().astype(int)` |
| Imputer | `SimpleImputer(strategy="median")`: `fit_transform` train, `transform` val/test |
| Duplicates | `drop_duplicates(subset=cols)` before the split |
| Out of range | `s.mask((s < lo) \| (s > hi))` |
| Strings | `.str.strip().str.lower()`, then `.map(dict)` |
| Leakage | drop anything not known at prediction time |
| Save | `to_parquet(path, index=False)` |
