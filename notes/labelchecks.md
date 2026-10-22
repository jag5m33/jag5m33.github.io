# Label checks

A label check asks whether the target column itself can be trusted, because a model can only be as good as the answers it learns from.

## Why check the label?

- **Label** (or **target**) = the column the model learns to predict: `severity` in the accidents data, `label_name` in LEDGAR.
- Every other check in EDA is about the **inputs**. But if the label is inconsistent, a model learns the inconsistency, and your metrics measure agreement with a broken answer key.
- **Label noise** = labels that are wrong, or that mean different things in different rows.

> **Key idea:** A label is usually a human decision, not a measurement. Different people, providers or years can draw the line in different places. Jag's phrasing: "Severity is not measured by an instrument. Someone at the reporting provider decides where the line falls between minor and serious delay."

## Four checks, in order

| Check | Question | Red flag |
|---|---|---|
| 1. Duplicates with conflicting labels | Does the same input appear with two different labels? | Any conflict at all |
| 2. Label distribution over time (or by source) | Is the label rate stable across years, providers, batches? | Steps or trends that have no real-world explanation |
| 3. Label vs proxy features | Do inputs that **should** give the same label actually do so? | Two synonyms with very different label rates |
| 4. Read samples by hand | Do the labels look right when you read the rows? | Near-identical rows with different labels; labels that overlap in meaning |

## Check 1: duplicates with conflicting labels

- Find rows whose **input** is identical, then count how many distinct labels each group has.
- One label per group → harmless duplicates (still remove extra copies before splitting).
- More than one → the true label cannot be known from the data.

```python
# standardise first so trivial differences don't hide duplicates
train["text"] = train["text"].str.lower()

print(train["text"].duplicated().sum())              # extra copies

# every member of a group
dups = train[train["text"].duplicated(keep=False)]
n_labels = dups.groupby("text")["label_name"].nunique()
conflict = n_labels[n_labels > 1].index              # inputs with 2+ labels
print("texts with conflicting labels:", len(conflict))

# read them side by side
print(train[train["text"].isin(conflict)]
      .sort_values("text")[["label_name", "text"]])

# drop the conflicting rows entirely, then the extra copies of the rest
train = train[~train["text"].isin(conflict)]
train = train.drop_duplicates(subset="text").reset_index(drop=True)
```

**Tiny example.** 7 clauses: one text appears twice as "No Waivers" and "Waivers", one appears twice as "Governing Laws", one twice as "Notices", plus one unique clause. `duplicated().sum()` = 3 extra copies; 1 text has conflicting labels; after cleaning, 3 rows remain.

**In LEDGAR** ([eda.ipynb](https://github.com/jag5m33/py_legal_data/blob/main/notebooks/eda.ipynb)): 64 duplicate clauses in train, 3 of them with conflicting labels:
- No Waivers / Waivers
- Applicable Laws / Governing Laws
- Qualifications / Existence

"In each case the labels overlap in meaning." Jag removed the conflicting rows and the extra copies of the rest, leaving 59,933 training clauses.

## Check 2: label distribution over time and by source

Group the label rate by **when** and **where** the row came from. A real-world rate should change slowly and for a reason.

```python
df["severe"] = df["severity"].isin([3, 4]).astype(int)

# % severe per year
print(df.groupby("start_year")["severe"].mean().mul(100).round(1))

# % severe per provider and year, as a table
rate = (df.pivot_table(index="start_year", columns="source",
                       values="severe", aggfunc="mean") * 100).round(1)
print(rate)
rate.plot(marker="o"); plt.ylabel("% severe"); plt.show()
```

![Left: % severe by year across all sources, falling from about 35% in 2016-2018 to 2.9% in 2023. Right: the same split by provider: Source1 collapses, Source3 drifts down, Source2 stays between 30 and 37%.](figures/labelchecks-year.svg)

In the figure, the left panel is the whole dataset: flat for three years, then falling every year. The right panel shows **who** changed. Source2 (teal) stays flat; Source1 (wine) falls to under 3%.

**In the accidents project** ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt), 2.4–2.6):
- % severe by year: 34.2 (2016), 35.6 (2017), 35.6 (2018), 27.9 (2019), 18.4 (2020), 11.4 (2021), 6.9 (2022), 2.9 (2023). "Accidents did not become twelve times less disruptive. The LABEL changed."
- By source, the three providers disagreed even in 2016 (28.4 vs 36.8 vs 47.5), so they were never using one standard.
- Decision: keep **Source2 only**, the one provider whose rate holds steady (3,305,373 rows, 43% of the data). New baseline 33.92%.

> **Trap:** A drifting label contaminates **every** feature. Recent years held far more rows, so anything more common recently looked artificially safe. A model would learn "older record = severe" — a fact about the filing system.

## Check 3: label vs proxy features (the synonym test)

- **Proxy feature** = an input that should, on its own, imply nearly the same label as another input. The cleanest proxies are **synonyms**: two words for the same thing.
- **Logic:** different conditions **should** differ in label rate. Synonyms **should not**, because there is no real difference between them — only a difference in wording.

Jag's analogy: two thermometers in one room reading 20 C and 68 F is fine; one reading 20 C and the other 35 C means an instrument is broken. "This is a check on the instrument."

```python
# % severe for each keyword in the weather text
words = ["clear", "fair", "overcast", "cloud", "rain", "snow", "fog"]
rows = []
for w in words:
    has = df["weather_condition"].str.contains(w, case=False, na=False)
    rows.append({"word": w,
                 "pct_severe": df.loc[has, "severe"].mean() * 100,
                 "n": has.sum()})
print(pd.DataFrame(rows).round(2))

# what is attached to the choice of word? cross it with source and year
w = df["weather_condition"].str.lower()
sub = df[w.isin(["clear", "fair"])]            # just the two synonyms
pd.crosstab([sub["source"], sub["start_year"]],
            sub["weather_condition"].str.lower())
```

**What it found:**
- "clear" and "fair" describe the same sky but differed in % severe by **21.8 points**; "overcast" vs "cloudy" by **15.1 points**.
- Crossing with source and year: Source1 "clear" went 58,195 (2018) → 13,451 (2019) → 0 from 2020, while "fair" went 862 → 99,455 → 749,567 (2022). There were zero rows of clear or overcast after 2019 in all three sources.
- So the weather wording changed **at the end of 2019**, simultaneously across all three sources, pointing at the shared weather-data supplier. "The 21.8-point gap was measuring WHEN, not WHAT." That led straight to Check 2.

![Bar chart of the gap in % severe between synonym pairs: clear vs fair 21.8 points on all sources but 4.6 on Source2 only; overcast vs cloudy 15.1 points vs 0.4.](figures/labelchecks-synonyms.svg)

In the figure, wine bars are the gaps in the blended data; teal bars are the same gaps after restricting to Source2. The synonyms now agree, which Jag called "the proof that the Source2 decision was correct". The leftover 4.6 points fits Source2's own dip from 36.8 to 30.3 around 2019.

> **Key idea:** The synonym test is a cheap, powerful way to validate a label. Find two inputs that must mean the same thing; if the label rate differs, something other than the input is driving the label.

## Check 4: read samples by hand

Numbers find suspicious groups; **reading** tells you why. Read:
- the conflicting duplicates from Check 1;
- a random sample from each label;
- the rows the model gets wrong (after modelling), grouped by true and predicted label.

```python
# 3 random examples from each of a few labels, full text visible
pd.set_option("display.max_colwidth", 300)
for lab in ["Governing Laws", "Applicable Laws", "Venues"]:
    print("==", lab)
    print(train.loc[train["label_name"] == lab, "text"]
               .sample(3, random_state=0).to_string())

# the longest items are often several records stuck together
train.nlargest(5, "word_count")[["label_name", "word_count", "text"]]
```

**In LEDGAR**, labels "come from whatever heading a contract's author chose, so near-identical wording can carry different labels" ([README](https://github.com/jag5m33/py_legal_data/blob/main/README.md)). After modelling, the most-confused pairs were the same near-synonyms:

![Horizontal bars of LEDGAR's most-confused validation pairs: Governing Laws predicted as Applicable Laws 73 times, the reverse 33, Amendments as Modifications 25, down to Waivers as No Waivers 15. Pairs that were also found as conflicting duplicates are wine.](figures/labelchecks-ledgar.svg)

In the figure, the wine bars are pairs that Check 1 had already flagged as conflicting duplicates. "The conflicting-label pairs found in the EDA are the same pairs the model confuses most" — so a large share of the remaining error is label noise, not model error. His planned next step is to try merging near-identical labels (for example Governing Laws and Applicable Laws) to measure how much.

## Label checks vs value outliers

These checks are about the **label's meaning**. Odd **values** in a numeric column (a 160 F temperature) are a different problem, handled with physical limits or the IQR rule — see [Quartiles, IQR & outliers](#/t/quartiles) and [Data profiling](#/t/profiling).

## What to do when a label fails

| Finding | Options |
|---|---|
| Conflicting duplicates | Drop them (true label unknown), or pick one rule and document it |
| Rate drifts by source / year | Restrict to a consistent subset (Jag: Source2), or keep source and year as explicit controls |
| Near-synonym labels | Merge them if they truly mean the same; otherwise expect those pairs to be confused and report it |
| Rare labels | Collapse (Jag collapsed severity 1–4 into severe = 3 or 4) or use macro F1 to watch them |
| Leakage in a validator | Use leaky columns only to **validate** the label, never as inputs (severe accidents should last longer) |

## How to say it

> **Say it like this:** "Before trusting the target I checked it four ways: conflicting duplicates, its rate over time and by provider, whether synonyms got the same label rate, and by reading examples. In the accidents data, two words for the same weather differed by 21.8 points; that traced to a wording change at the end of 2019 and then to providers grading severity differently, so I kept the one provider whose rate was stable."

## Traps

- **Treating a falling label rate as a real trend** without asking who recorded it.
- **Checking the label only once.** After any restriction, re-run the checks (the synonym gap was the validation).
- **Assuming conflicting duplicates are rare enough to ignore.** Three out of 60,000 is tiny, but they pointed at the exact pairs the model later confused.

## Quick reference

| Term | Meaning | Code |
|---|---|---|
| Label noise | labels that are wrong or inconsistent | — |
| Conflicting duplicates | same input, 2+ labels | `groupby(input)[label].nunique() > 1` |
| Label drift | label rate changes over time for non-real reasons | `groupby(year)[target].mean()` |
| Provider effect | label rate differs by source | `pivot_table(index=year, columns=source)` |
| Synonym test | two equivalent inputs should share a label rate | `str.contains(word)` then `mean()` |
| Hand reading | read samples per label and the confused pairs | `.sample(3, random_state=0)` |
| Baseline | label rate over all rows (Source2: 33.92%) | `df[target].mean()` |
