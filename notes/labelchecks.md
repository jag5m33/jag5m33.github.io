# Label checks

A label check asks whether the target column itself can be trusted, because a model can only be as good as the answers it learns from.

## Why check the label?

- **Label** (or **target**) = the column the model learns to predict: `severity` in the accidents data, `label_name` in LEDGAR.
- Most EDA checks the **inputs**. But if the label is inconsistent, the model learns the inconsistency, and your scores measure agreement with a broken answer key.
- **Label noise** = labels that are wrong, or mean different things in different rows.

> **Key idea:** A label is usually a human decision, not a measurement. In Jag's words: "Severity is not measured by an instrument. Someone at the reporting provider decides where the line falls between minor and serious delay."

## Four checks, in order

| Check | Question | Red flag |
|---|---|---|
| 1. Conflicting duplicates | Does the same input appear with two labels? | Any conflict at all |
| 2. Label rate over time / by source | Is the rate stable across years and providers? | Jumps with no real-world reason |
| 3. Synonym test | Do two inputs that mean the same thing get the same label rate? | A big gap between synonyms |
| 4. Read by hand | Do the labels look right when you read rows? | Near-identical rows, different labels |

## Check 1: conflicting duplicates

Find rows with identical **input**, then count how many different labels each group has. More than one → the true label can't be known.

```python
train["text"] = train["text"].str.lower()             # so case can't hide copies
dups = train[train["text"].duplicated(keep=False)]    # every member of each group
n_labels = dups.groupby("text")["label_name"].nunique()
conflict = n_labels[n_labels > 1].index               # texts with 2+ labels

train = train[~train["text"].isin(conflict)]          # drop conflicting rows
train = train.drop_duplicates(subset="text")          # drop extra copies
```

**Tiny example.** 7 clauses: text A twice ("No Waivers" and "Waivers"), text B twice (both "Notices"), text C twice (both "Governing Laws"), text D once. That is 3 extra copies and 1 conflict. Drop both A rows and the extra B and C → 3 rows left.

**In LEDGAR** ([eda.ipynb](https://github.com/jag5m33/py_legal_data/blob/main/notebooks/eda.ipynb)): 64 duplicate clauses, 3 with conflicting labels — No Waivers / Waivers, Applicable Laws / Governing Laws, Qualifications / Existence. "In each case the labels overlap in meaning." After cleaning, 59,933 training clauses remained.

## Check 2: label rate over time and by source

A real-world rate should change slowly and for a reason. Group it by **when** and **where** the row came from.

```python
df["severe"] = df["severity"].isin([3, 4]).astype(int)      # 1 = severe
# % severe per year (rows) and source (columns)
rate = df.pivot_table(index="start_year", columns="source",
                      values="severe", aggfunc="mean") * 100
rate.plot(marker="o")
```

![Left: % severe by year across all sources, falling from about 35% in 2016-2018 to 2.9% in 2023. Right: the same split by provider: Source1 collapses, Source3 drifts down, Source2 stays between 30 and 37%.](figures/labelchecks-year.svg)

In the figure, the left panel (all data) is flat for three years, then falls every year. The right panel shows **who** changed: Source2 (teal) stays flat; Source1 (wine) falls below 3%.

**In the accidents project** ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt)):
- % severe fell from 34.2 (2016) to 2.9 (2023). "Accidents did not become twelve times less disruptive. The LABEL changed."
- Even in 2016 the three providers disagreed (28.4 vs 36.8 vs 47.5%).
- Decision: keep **Source2 only**, the one stable provider (3,305,373 rows). New baseline 33.92%.

> **Trap:** A drifting label spoils **every** feature. Recent years had far more rows, so anything common recently looked "safe". A model would learn "older record = severe" — a fact about the filing system, not about roads.

## Check 3: the synonym test

- **Synonyms** = two words for the same thing ("clear" and "fair" sky).
- Different conditions **should** differ in label rate. Synonyms **should not** — the only difference is wording.

Jag's analogy: two thermometers in one room reading 20 C and 68 F is fine; 20 C and 35 C means one is broken. "This is a check on the instrument."

```python
for w in ["clear", "fair", "overcast", "cloudy"]:
    has = df["weather_condition"].str.contains(w, case=False, na=False)
    pct = df.loc[has, "severe"].mean() * 100   # % severe for this word
    print(w, round(pct, 1), has.sum())          # word, %, rows
```

**What it found:**
- "clear" vs "fair" differed by **21.8 points**; "overcast" vs "cloudy" by **15.1 points**.
- The wording switched at the end of 2019 in all three sources (no "clear" after 2019), so "the 21.8-point gap was measuring WHEN, not WHAT". That led to Check 2.

![Bar chart of the gap in % severe between synonym pairs: clear vs fair 21.8 points on all sources but 4.6 on Source2 only; overcast vs cloudy 15.1 points vs 0.4.](figures/labelchecks-synonyms.svg)

In the figure, wine bars are the gaps in all data; teal bars are the gaps in Source2 only. The synonyms now nearly agree — "the proof that the Source2 decision was correct".

## Check 4: read samples by hand

Numbers find suspicious groups; **reading** tells you why. Read the conflicting duplicates, a few random rows per label (`.sample(3, random_state=0)`), and later the rows the model gets wrong.

**In LEDGAR**, labels "come from whatever heading a contract's author chose, so near-identical wording can carry different labels" ([README](https://github.com/jag5m33/py_legal_data/blob/main/README.md)).

![Horizontal bars of LEDGAR's most-confused validation pairs: Governing Laws predicted as Applicable Laws 73 times, the reverse 33, Amendments as Modifications 25, down to Waivers as No Waivers 15. Pairs that were also found as conflicting duplicates are wine.](figures/labelchecks-ledgar.svg)

In the figure, wine bars are pairs Check 1 had already flagged. "The conflicting-label pairs found in the EDA are the same pairs the model confuses most" — so much of the remaining error is label noise, not the model. His next step is to try merging near-identical labels.

## What to do when a label fails

| Finding | What to do |
|---|---|
| Conflicting duplicates | Drop them (true label unknown) |
| Rate drifts by source / year | Keep a consistent subset (Jag: Source2) |
| Near-synonym labels | Merge them, or expect and report the confusion |
| Rare labels | Combine levels (Jag: severe = 3 or 4) |

Odd **values** in an input (a 160 F temperature) are a different problem — see [Data profiling](#/t/profiling).

> **Say it like this:** "Before trusting the target I checked conflicting duplicates, its rate over time and by provider, whether synonyms got the same rate, and read examples. Two words for the same weather differed by 21.8 points; that traced to providers grading severity differently, so I kept the one stable provider."

## Quick reference

| Term | Meaning | Code |
|---|---|---|
| Label noise | wrong or inconsistent labels | — |
| Conflicting duplicates | same input, 2+ labels | `groupby(input)[label].nunique() > 1` |
| Label drift | rate changes for non-real reasons | `groupby(year)[target].mean()` |
| Synonym test | equal inputs should share a label rate | `str.contains(word)` then `mean()` |
| Baseline | label rate over all rows (Source2: 33.92%) | `df[target].mean()` |
