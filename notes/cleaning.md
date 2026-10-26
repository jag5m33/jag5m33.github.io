# Data cleaning

Data cleaning turns the raw table into one a model can learn from honestly: fix types, handle gaps and repeats, tidy text, remove leakage and write down every decision.

## Order of work

[Data profiling](#/t/profiling) finds the problems; cleaning fixes them. Order matters:

| Step | Why here |
|---|---|
| 1. Stack / join files | everything after works on one table |
| 2. Fix dtypes | dates and numbers must be real dates and numbers first |
| 3. Tidy strings | so "Clear" and "clear " count as one value |
| 4. Remove duplicates | **before** splitting, so no copy lands in both train and test |
| 5. Impossible values → missing | treat them as gaps, not data |
| 6. Remove leakage columns | before any model sees them |
| 7. Split train / validation / test | |
| 8. Fill missing values | **learned from train only** |
| 9. Save (parquet) | checkpoint the clean version |

Throughout: keep a **cleaning log**.

## Stacking and joining files

- **Concatenate** (`pd.concat`) = stack tables **on top of each other** (more rows). For the same data split by year.
- **Merge** (`pd.merge`) = join tables **side by side** on a shared key column (more columns).

`pd.concat` matches columns **by name**. A column one file lacks becomes `NaN` for that file's rows.

![Diagram of two yearly tables stacked with pd.concat. The 2018 rows have NaN in nsl_flow and viking_flow because those columns did not exist in the 2018 file.](figures/cleaning-concat.svg)

In the figure, the wine cells mean "this column did not exist yet", not "the reading failed". Remember that when you handle missing values.

```python
import pandas as pd

# load six yearly NESO files and stack them (from his GB demand notebook)
dfs = [pd.read_csv(f"../data/raw/demanddata_{y}.csv") for y in range(2018, 2024)]
df = pd.concat(dfs, ignore_index=True)    # missing columns become NaN
print(len(df))                            # expect 105,168 rows

# join: add weather columns matched on station and hour
df = df.merge(weather, on=["station", "hour"], how="left")
```

**His check** ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)): 17,520 rows per year (365 × 48), 17,568 in leap-year 2020, 105,168 in total. Blanks appeared only in columns missing from some files, such as `viking_flow` (not in 2018).

## Fixing dtypes and tidying text

- `errors="coerce"` = "if you can't convert it, make it missing". Count how many it changed — a jump means a format you didn't expect. Example: `["29.9", "30.1", "N/A"]` becomes `[29.9, 30.1, NaN]`.
- To merge spellings of the same thing, use `map` with a dictionary.

```python
df["start_time"] = pd.to_datetime(df["start_time"], format="ISO8601",
                                  errors="coerce")
df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")  # bad -> NaN
df["source"] = df["source"].astype("category")  # few labels -> category

d = df["wind_direction"].str.strip().str.lower()   # " North" -> "north"
wind_map = {"north": "N", "n": "N", "west": "W", "w": "W", "calm": "CALM"}
df["wind_direction"] = d.map(wind_map).fillna(d.str.upper())
```

Jag's raw `wind_direction` had "North" and "N", "West" and "W"; after merging to 16 compass points, CALM was the most common value (17.21%).

> **Trap:** Clock changes. Jag's NESO data had 46 periods on the March clock-change day and 50 in October, so he built a continuous GMT timestamp instead.

> **Trap:** Only merge labels when nothing else differs. "clear" and "fair" look like the same sky, but they differed in % severe by 21.8 points because they marked **different years**. Merging would have hidden that clue (see [Label checks](#/t/labelchecks)).

## Duplicates, impossible values and leakage

- **Duplicates:** `df.drop_duplicates(subset=cols)`, leaving out the ID column. In Source2 Jag removed 3,581 copies before the split. In time series, a duplicate is two rows for the same timestamp.
- **Impossible values → missing:** use `df[col].mask((df[col] < lo) | (df[col] > hi))` with physical limits (temperature −70 to 134 F). They are then filled like any other gap. Hidden gaps too: `england_wales_demand == 0` was a recording error.
- **Leakage** = a column you would not have at prediction time, or one that is really the answer. It makes scores look great and real use fail. Jag's test: "would a dispatcher have this number at the moment the crash happens?" `end_time`, `distance(mi)` and `description` all "measure the SIZE OF THE DISRUPTION… Severity IS the size of the disruption", so he dropped them as inputs.

## Missing values

### Why is it missing?

Ask whether the gaps are random or follow a pattern. Compare the target for rows with and without the gap. Jag found accidents with no weather reading were 2.5–11.4 points more likely to be severe, so "missing" itself carried information and he kept a `weather_missing` flag ([py_acc README](https://github.com/jag5m33/py_acc/blob/main/README.md)).

### The options

1. **Drop** rows or columns. Jag dropped `precipitation` (47% missing).
2. **Fill with one value** (**impute** = fill a gap with an estimate): median for skewed numbers, mode (most common value) for categories, or a label like "unknown".
3. **Fill by group**: use the median of similar rows, not everyone.
4. **Flag**: add a 0/1 column so the model can learn if "missing" matters.

```python
# 3. group fill: a missing Minnesota temperature gets Minnesota's median
state_median = df.groupby("state")["temp"].transform("median")
df["temp_filled"] = df["temp"].fillna(state_median)

# 4. flag first, then fill
df["temp_missing"] = df["temp"].isna().astype(int)   # 1 = was missing
```

**Tiny example.** CA temps 70, 72, NaN and MN temps 20, NaN, 24. Group fill gives CA 71 and MN 22. One overall median would put 47 in both — wrong for each state.

### Fit on train only

Learn the fill value (e.g. the median) from **training rows only**, then reuse it for validation and test. Otherwise test-set information leaks into training.

```python
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)        # 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42)  # 20% val, 60% train

imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
X_train[cols] = imputer.fit_transform(X_train[cols])   # learn medians HERE only
X_val[cols] = imputer.transform(X_val[cols])           # reuse train medians
X_test[cols] = imputer.transform(X_test[cols])
```

That is the 60/20/20 split (0.25 × 0.8 = 0.2) and median fill in his [dcdm.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/dcdm.ipynb).

![Two histograms of the same temperature column. Left: observed values only, SD about 14.8. Right: after filling 323 missing rows with the median, a tall spike at the median and SD about 12.1.](figures/cleaning-impute.svg)

In the figure, filling 323 of 1,000 rows with one value makes a tall spike at the median, and the spread drops from about 14.8 to 12.1. Median filling makes data look less varied than it really is.

## Save and log

- **Parquet** = a compressed, column-based file format. Much faster than CSV, and it **keeps dtypes** (dates stay dates). Save with `df.to_parquet("df_cleaned.parquet", index=False)` after each stage.
- **Cleaning log** = a record of every decision: what, how many rows, why. Jag's notes.txt does this with TESTED, RESULT, MEANING, NEXT and DECISION for each step ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt)).

## Caveats

- **Median filling** shrinks the spread (figure above).
- **Dropping rows with gaps** can bias the sample if the gaps follow a pattern (e.g. losing remote sites).
- **Filling before the split** leaks test information.
- **Keeping only a subset** (Source2) changes coverage: 9 states almost absent and no 2023.
- **`errors="coerce"`** silently makes NaN — always count them.

> **Say it like this:** "I stacked the yearly files and checked row counts; fixed dtypes and tidied strings; dropped duplicates before splitting; turned impossible values into missing; removed leakage columns only known after the event; split 60/20/20; and filled medians learned from the training set only, with a flag where missingness predicted the target. Every step is logged."

## Quick reference

| Task | Code |
|---|---|
| Stack files | `pd.concat(list_of_dfs, ignore_index=True)` |
| Join on a key | `df.merge(other, on="key", how="left")` |
| Text → date / number | `pd.to_datetime(...)`, `pd.to_numeric(..., errors="coerce")` |
| Tidy strings | `.str.strip().str.lower()`, then `.map(dict)` |
| Duplicates | `drop_duplicates(subset=cols)` before the split |
| Fill | `fillna(s.median())`, by group with `transform("median")` |
| Flag | `df[c + "_missing"] = df[c].isna().astype(int)` |
| Imputer | `fit_transform` on train, `transform` on val/test |
| Leakage | drop anything not known at prediction time |
| Save | `to_parquet(path, index=False)` |
