# Univariate, bivariate & multivariate analysis

EDA moves from one column at a time, to pairs of columns, to several columns together — and the variable types decide the right chart at each step.

## The three levels

| Level | Variables | Question | Typical tools |
|---|---|---|---|
| **Univariate** | 1 | What does this column look like on its own? | `describe`, `value_counts`, histogram |
| **Bivariate** | 2 | Do these two columns move together? | scatter, box per group, crosstab |
| **Multivariate** | 3+ | Does the pair's link change when a third column changes? | hue, facets, pivot table |

- **Uni** = one, **bi** = two, **multi** = many. Work in this order.
- Jag's GB demand notebook uses exactly these headings: "Univariate analysis: look at each column on its own", then "Bivariate analysis: look at how other columns relate to ND" ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)).

## Univariate: one column

This is [Data profiling](#/t/profiling) plus the charts in [Variable types & their charts](#/t/vartypes). For a number, look at the **centre** (mean vs median), **spread** (SD, IQR) and **shape** (one peak or two, long tail?). For a label, look at counts and percentages.

**From his projects:**
- GB demand (ND): median 27,202 MW, mean 27,963 MW. Mean above median means a long right tail (**right-skewed**).
- LEDGAR `word_count`: median 86 words, some clauses over 1,000.
- Accidents `severity`: about 80% class 2 — a heavily **imbalanced** target.

## Bivariate: the pair of types decides the tool

| Pair | Chart | Summary number |
|---|---|---|
| numeric – numeric | scatter plot | correlation $r$ |
| category – numeric | box or violin per group | difference in group medians |
| category – category | crosstab, stacked bar | difference in row % |

### Numeric – numeric

- **Scatter plot** = one dot per row, one column on each axis.
- **Pearson $r$** = a number from −1 to +1 saying how closely the dots follow a **straight line**. More in [Correlation](#/t/correlation).

![Three scatter plots: a tight upward cloud with r near 0.9, a shapeless cloud with r near 0, and a looser downward cloud with r near -0.6.](figures/unibimulti-scatter.svg)

In the figure, the number in each title is the $r$ value. Notice how loose the right-hand cloud still is at $r = -0.6$.

> **Trap:** $r$ near 0 does not mean "no relationship". In GB demand, solar vs ND was −0.07 over all rows but −0.54 at summer midday, because night-time zeros hid the effect.

### Category – numeric

- One box per group: category on the x-axis, number on the y-axis.
- **Violin plot** = a box plot with the shape drawn around it, so two peaks become visible.

![Left: box plots of demand for summer, spring/autumn and winter, rising left to right. Right: violins of the same data, where each violin bulges twice.](figures/unibimulti-catnum.svg)

In the figure, the boxes show winter is highest. The violins show what boxes hide: each season bulges twice (day and night).

```python
import seaborn as sns, matplotlib.pyplot as plt

# numeric-numeric: scatter with a straight trend line
sns.regplot(data=df, x="embedded_solar_generation", y="nd", scatter_kws={"s": 8})
plt.show()
print(df["embedded_solar_generation"].corr(df["nd"]))   # Pearson r

# category-numeric: one box per weekday, plus the numbers behind it
sns.boxplot(data=df, x="weekday", y="nd")
plt.show()
print(df.groupby("month")["nd"].median())  # median: safe when skewed
```

**His results:** the gap between the highest and lowest group medians was about 13,000 MW for time of day, 10,000 MW for month and 4,400 MW across six years. He used that gap as the **effect size** — how big each pattern is.

### Category – category

- **Crosstab** = a table of counts, one category as rows and the other as columns.
- **Row-normalised crosstab** = the same as percentages, each row adding to 100%. It answers "within this group, what share is…?".

**Tiny worked example.** 10 accidents, road type vs severe (1 = severe):

| road | severe = 0 | severe = 1 | total |
|---|---|---|---|
| Interstate | 1 | 3 | 4 |
| Local | 5 | 1 | 6 |

- Interstate: 3 / 4 = 75% severe.
- Local: 1 / 6 = 17% severe.

A **58-point gap** — which raw counts hide, because the groups are different sizes.

```python
import pandas as pd
# row %: each row sums to 1
ct = pd.crosstab(df["road"], df["severe"], normalize="index")
sns.heatmap(ct * 100, annot=True, fmt=".1f", cmap="Reds")  # % per cell
plt.show()
```

![Left: a heatmap of a row-normalised crosstab of road type by severity, with percentages in each cell. Right: the same percentages as 100% stacked bars.](figures/unibimulti-catcat.svg)

In the figure, each heatmap row adds to 100 and each stacked bar fills the full width. Compare the wine segments to compare severe rates.

## Feature vs target

When there is a **target** (the column you will predict), ask: **does this feature change the target?** Jag's method ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt)):

1. **Baseline** = % severe across all rows.
2. For each value of a feature, compute % severe and compare with the baseline.
3. **Judge by the size of the gap**, not p-values — "at 3–7 million rows everything is statistically significant".

```python
df["severe"] = df["severity"].isin([3, 4]).astype(int)   # 1 = severe
baseline = df["severe"].mean() * 100                      # % severe overall
by_road = df.groupby("street_type")["severe"].agg(["mean", "size"])
by_road["gap"] = by_road["mean"] * 100 - baseline         # + raises, - lowers
print(by_road.sort_values("gap"))
```

![Horizontal bars of % severe by road type in Source2, from Local at 14.7% to Interstate at 80.4%, with a dashed baseline line at 33.92%. Bars above the baseline are wine, below are teal.](figures/unibimulti-target.svg)

In the figure, the dashed line is the baseline (33.92%). Road type spans 14.67% (Local) to 80.40% (Interstate) — "the strongest single column in the dataset by a wide margin". Keep the **row count** beside each %: a rare category can show a big gap from very few rows.

## Multivariate: add a third variable

The key question: **does a two-column pattern still hold when a third column is held fixed?** A third column that drives both is called a **confounder**.

- **Hue** = colour points by a third column.
- **Facets** = one small panel per value of a third column, same axes.
- **Pairplot** = a grid of scatter plots for every pair of numeric columns (`sns.pairplot`).
- **Correlation heatmap** = $r$ for every pair in one coloured grid. Jag's showed ND, TSD and England & Wales demand at 0.98–1.00 — near-copies that add nothing new.

![Three scatter panels of demand against solar, one per season, with points coloured by day or night.](figures/unibimulti-facets.svg)

In the figure, each panel holds season fixed and colour splits day from night. Night points sit at solar = 0 and flatten the overall pattern: $r$ = 0.00 over all rows but −0.64 for daytime only (synthetic data, same effect as Jag's real data).

![A lower-triangle correlation heatmap of demand, solar, temperature and hour, with r values written in each cell.](figures/unibimulti-corr.svg)

In this heatmap, red is positive and blue negative. The top half is hidden because it would just repeat the bottom half.

### The split test (Jag's method)

To check if a feature's effect is real, repeat the comparison **inside each value** of the likely confounder:
- gap survives → real effect;
- gap disappears → the link belonged to the other column.

![Dumbbell chart: for each road type, % severe with and without a traffic signal. The gap is large on Local, State Highway, US Highway and Freeway, and vanishes on Interstate where both are about 81%.](figures/unibimulti-split.svg)

In the figure, each line is one road type; wine = no signal, teal = signal. The gap survives on every road type except Interstate (both about 81%), so the traffic-signal effect is real.

The same test exposed a fake effect: `give_way` had a raw gap of 16 points, but on Local roads it was 14.55% vs 14.68%. "It was local roads against motorways wearing a different label." (A two-key table: `df.pivot_table(index="street_type", columns="give_way", values="severe")`.)

> **Key idea:** A strong link between two columns can be a third column in disguise. For wind vs demand, $r$ was +0.10 overall but −0.23 at winter 18:00 — the sign flipped once season was held fixed (Simpson's paradox).

> **Say it like this:** "I look at each column alone, then each feature against the target as % severe vs a baseline, judged by the size of the gap. Then I hold a likely confounder fixed — with facets or a pivot table — to check the effect survives."

## Quick reference

| What | Chart | Summary | Code |
|---|---|---|---|
| one numeric | histogram, box | median, IQR | `s.describe()` |
| one category | count plot | counts, % | `value_counts()` |
| num – num | scatter | Pearson $r$ | `s1.corr(s2)` |
| cat – num | box, violin | group medians | `groupby().median()` |
| cat – cat | heatmap, stacked bar | row % | `pd.crosstab(..., normalize="index")` |
| feature – target | bars vs baseline | gap from baseline | `groupby(f)["target"].mean()` |
| 3+ variables | hue, facets | two-key table | `pivot_table(index, columns)` |
| Confounder | — | third column driving both | split test |
