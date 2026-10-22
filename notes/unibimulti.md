# Univariate, bivariate & multivariate analysis

EDA moves from one column at a time, to pairs of columns, to several columns together, and the right chart and summary at each step depend on the variable types involved.

## The three levels

| Level | Number of variables | Question | Typical tools |
|---|---|---|---|
| **Univariate** | 1 | What does this column look like on its own? | `describe`, `value_counts`, histogram, count plot |
| **Bivariate** | 2 | Do these two columns move together? | scatter, correlation, box per group, crosstab |
| **Multivariate** | 3 or more | Does the pair's relationship change when a third column changes? | hue, facets, pairplot, `pivot_table` with two keys |

- **Uni-** = one, **bi-** = two, **multi-** = many.
- Work in this order. Univariate is [Data profiling](#/t/profiling) plus the charts in [Variable types & their charts](#/t/vartypes). Jag's GB demand notebook has exactly these headings: "Univariate analysis: look at each column on its own", then "Bivariate analysis: look at how other columns relate to ND" ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)).

## Univariate: what to look for per type

| Type | Look for | Code |
|---|---|---|
| Continuous | centre (mean vs median), spread (SD, IQR), **shape** (skew, one peak or two), tails, impossible values | `s.describe()`, `s.skew()`, `sns.histplot(x=s, kde=True)` |
| Discrete | which integers occur, spikes at round numbers, the max | `s.value_counts().sort_index()` |
| Nominal | number of categories, the most common, a long tail, duplicate spellings | `s.value_counts(normalize=True, dropna=False)` |
| Ordinal | counts in natural order; is any level nearly empty? | `s.value_counts().reindex(order)` |
| Binary | % True; rare flags (under 1%) need row counts, not just % | `s.mean() * 100` |

```python
# one continuous column, top to bottom
s = df["nd"]
print(s.describe(percentiles=[0.5, 0.75, 0.99]))   # centre, spread, tail
print("skew:", round(s.skew(), 2))                 # > 0 = long right tail
print("mean > median?", s.mean() > s.median())     # quick skew check
# shape: one peak or several?
sns.histplot(x=s, bins=100)
plt.show()
```

**From his projects:**
- GB demand (ND): median 27,202 MW, mean 27,963 MW, so the mean is above the median — right-skewed. The histogram was lumpy, and splitting by month showed two peaks (night and day).
- LEDGAR `word_count`: median 86 words, 95th percentile 305, some clauses over 1,000 words.
- Accidents `severity`: about 80% class 2 — a heavily imbalanced target.

## Bivariate analysis by pair type

The pair of **types** decides the tool. There are three combinations.

| Pair | Chart | Number that summarises it |
|---|---|---|
| numeric – numeric | scatter plot (+ trend line) | correlation $r$ |
| categorical – numeric | box / violin per group; bar of group means | difference in group medians or means |
| categorical – categorical | crosstab heatmap; stacked bar | difference in row percentages |

### Numeric – numeric: scatter and correlation

- **Scatter plot** = one dot per row, x = one column, y = the other.
- **Pearson correlation** $r$ = a number from −1 to +1 measuring how closely the dots follow a **straight line**. Full detail in [Correlation](#/t/correlation).

![Three scatter plots: a tight upward cloud with r near 0.9, a shapeless cloud with r near 0, and a looser downward cloud with r near -0.6.](figures/unibimulti-scatter.svg)

In the figure, the number in each title is what `df.corr()` would give. Notice how loose the right-hand cloud is even at $r = -0.6$.

```python
# scatter with a fitted straight line
sns.regplot(data=df, x="embedded_solar_generation", y="nd",
            scatter_kws={"s": 8, "alpha": 0.4})
plt.show()

# r for one pair
# Pearson by default
r = df["embedded_solar_generation"].corr(df["nd"])
# rank-based
rho = df["embedded_solar_generation"].corr(df["nd"], method="spearman")
print(round(r, 2), round(rho, 2))
```

- **Spearman** correlation uses ranks instead of raw values, so it captures any steadily rising or falling relationship, not only straight lines, and is less pulled by outliers.
- On big data, sample before scattering: Jag used `df_merged.sample(5000, random_state=42)` for his correlation heatmap.

> **Trap:** $r$ near 0 does not mean "no relationship". In GB demand, solar vs ND was −0.07 over all rows but −0.54 at summer midday, because night-time zeros and season hid the effect.

### Categorical – numeric: box / violin and grouped means

- Put the category on the x-axis and the number on the y-axis: one box per group.
- **Violin plot** = a box plot with the KDE shape drawn around it, so two peaks become visible.
- **Grouped means / medians** = `groupby(category)[number].agg(...)` — the table behind the chart.

![Left: box plots of demand for summer, spring/autumn and winter, rising left to right. Right: violins of the same data, where each violin bulges twice.](figures/unibimulti-catnum.svg)

In the figure, the boxes show winter sits highest. The violins add something the boxes hide: each season has two bulges, the day and night groups.

```python
# one box per weekday, in calendar order (from his elec notebook)
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
sns.boxplot(data=df, x="weekday", y="nd", order=weekday_order)
plt.show()

# violin: shape as well as quartiles
sns.violinplot(data=df, x="season", y="nd", inner="quartile", cut=0)
plt.show()

# the numbers behind the chart
print(df.groupby("month")["nd"].agg(["median", "mean", "count"]))
```

**His results** with this pair type, read as "gap between the highest and lowest group medians": time of day about 13,000 MW, month about 10,000 MW, year about 4,400 MW over six years. He called this the **effect size** of each pattern.

> **Trap:** Use `median` when the numeric column is skewed. Jag used medians for ND and for LEDGAR word counts per class for this reason.

### Categorical – categorical: crosstab, normalised crosstab, heatmap, stacked bar

- **Crosstab** (contingency table) = a table of counts with one category's values as rows and the other's as columns.
- **Normalised crosstab** = the same table as percentages. `normalize="index"` makes **each row add to 100%**, which answers "within this group, what share is…?".

**Tiny worked example.** 10 accidents, road type and severe (1 = severe):

| road | severe = 0 | severe = 1 | row total |
|---|---|---|---|
| Interstate | 1 | 3 | 4 |
| Local | 5 | 1 | 6 |
| All | 6 | 4 | 10 |

Row-normalise by dividing each cell by its row total:
- Interstate: 3 / 4 = 0.75 severe.
- Local: 1 / 6 = 0.17 severe.
- Overall: 4 / 10 = 0.40.

So the **% severe differs by 58 points** between road types, which a raw count table hides because the groups are different sizes.

```python
pd.crosstab(df["road"], df["severe"])                            # counts
pd.crosstab(df["road"], df["severe"], margins=True)              # + totals
ct = pd.crosstab(df["road"], df["severe"], normalize="index")    # row %
print((ct * 100).round(1))

# heatmap of the % table
sns.heatmap(ct * 100, annot=True, fmt=".1f", cmap="Reds")
plt.show()

# 100% stacked bar: one bar per road type, split by severe / not severe
(ct * 100).plot(kind="barh", stacked=True)
plt.xlabel("% of rows")
plt.show()
```

![Left: a heatmap of a row-normalised crosstab of road type by severity, with percentages in each cell. Right: the same percentages as 100% stacked bars.](figures/unibimulti-catcat.svg)

In the figure, each row of the heatmap adds to 100, and each stacked bar fills the full width; compare the wine segments across bars to compare severe rates.

Jag used crosstabs to show the four light columns are **nested**: `pd.crosstab(df.sunrise_sunset, df.astronomical_twilight, normalize=True)` gave Day/Night = 0.00, so if `sunrise_sunset` says Day, all four do ([eda.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb)).

> **Trap:** `normalize=True` divides by the grand total (whole table sums to 1); `normalize="index"` divides by row totals; `normalize="columns"` by column totals. Pick the one that answers your question.

## Target-focused bivariate analysis (feature vs target)

When there is a **target** (the column you will predict), the most useful bivariate question is: **does this feature change the target?**

Jag's method in Stage 2 of the accidents project ([notes.txt](https://github.com/jag5m33/py_acc/blob/main/notes.txt)):

1. **Baseline** = % severe across all rows.
2. **Single-column test:** for each value of a feature, recompute % severe and compare with the baseline.
3. **Judge by size of the gap**, not p-values. "At 3–7 million rows everything is statistically significant."

```python
df["severe"] = df["severity"].isin([3, 4]).astype(int)   # binary target
baseline = df["severe"].mean() * 100                      # % severe overall

by_road = (df.groupby("street_type")["severe"]
             .agg(pct_severe=lambda s: s.mean() * 100, n="size")
             .sort_values("pct_severe"))
# + raises, - lowers
by_road["gap"] = by_road["pct_severe"] - baseline
print(round(baseline, 2)); print(by_road.round(2))
```

![Horizontal bars of % severe by road type in Source2, from Local at 14.7% to Interstate at 80.4%, with a dashed baseline line at 33.92%. Bars above the baseline are wine, below are teal.](figures/unibimulti-target.svg)

In the figure, the dashed line is the baseline (33.92% in Source2). Bars far from it are strong features. Road type spans 66 points, from 14.67% (Local) to 80.40% (Interstate) — "the strongest single column in the dataset by a wide margin".

Always keep the **row count** next to the percentage: a rare category (like `roundabout`, True in 0.0032% of rows) can show a big gap from very few rows.

## Multivariate analysis

**Multivariate** = three or more variables at once. The main question: **does a two-variable pattern hold when a third variable is held fixed?** A third variable that drives both is a **confounder**.

### Hue and facets

- **Hue** = colour the points or bars by a third (categorical) column.
- **Facets** = draw one small panel per value of a third column, with shared axes, so panels can be compared.

![Three scatter panels of demand against solar, one per season, with points coloured by day or night.](figures/unibimulti-facets.svg)

In the figure, each panel holds season fixed; within it, colour separates day and night. Night points sit at solar = 0 and pull the overall pattern flat: in this synthetic data $r$ = 0.00 over all rows but −0.64 for daytime rows only. Comparing like with like is what makes a hidden relationship visible — the same thing Jag found for real solar data.

```python
# hue: a third variable as colour
sns.scatterplot(data=df, x="embedded_solar_generation", y="nd",
                hue="period")

# facets: one panel per month (from his elec notebook)
g = sns.FacetGrid(summer_df, col="month", col_wrap=4, height=4,
                  sharex=True, sharey=True)
g.map_dataframe(sns.scatterplot, x="embedded_solar_generation", y="nd",
                marker="x")
g.set_axis_labels("Embedded Solar Generation", "ND")
plt.show()

# displot does histogram facets in one call
sns.displot(data=df, x="nd", col="month", col_wrap=4, kind="hist",
            height=3, kde=True)
```

### Pairplot

- **Pairplot** = a grid of every pair of numeric columns as scatter plots, with each column's histogram on the diagonal. A fast multivariate overview.

![A corner pairplot of demand, solar, temperature and hour: histograms on the diagonal, scatter plots below.](figures/unibimulti-pairplot.svg)

In the figure, read each lower cell as "row variable vs column variable". The demand–hour cell shows a step (night low, day high) that a single $r$ would summarise badly.

```python
cols = ["nd", "embedded_solar_generation", "embedded_wind_generation"]
# pairplots are slow on big data
sample = df[cols].sample(2000, random_state=42)
sns.pairplot(sample, corner=True, plot_kws={"s": 8, "alpha": 0.5})
plt.show()
```

### Correlation heatmap

```python
corr = sample.corr(numeric_only=True)                    # r for every pair
# hide repeated top half
mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()
```

![A lower-triangle correlation heatmap of demand, solar, temperature and hour, with r values written in each cell.](figures/unibimulti-corr.svg)

In the figure, red is positive and blue negative; the top half is hidden because $r(x, y) = r(y, x)$. This is the same masking Jag used. His heatmap showed ND, TSD and England & Wales demand at 0.98–1.00 — near-copies, so they carry no new information.

### groupby / pivot_table with two keys

Using the 10-row example above with a `signal` column added:

```python
# mean of a 0/1 target = % severe, for every road x signal combination
pt = df.pivot_table(index="road", columns="signal",
                    values="severe", aggfunc="mean")
print(pt)

# the same with groupby + unstack, plus the row count
df.groupby(["road", "signal"])["severe"].agg(["mean", "size"]).unstack()
```

On the toy data, overall `signal=False` → 0.50 severe and `signal=True` → 0.25. Inside each road type the picture differs, which is the point of adding the second key.

### The split test (Jag's method)

To check whether a feature's effect is real or borrowed from a confounder, repeat the comparison **inside each value** of the confounder and compare cells **in the same row**:
- gap survives → real effect;
- gap disappears → the link belonged to the other column.

![Dumbbell chart: for each road type, % severe with and without a traffic signal. The gap is large on Local, State Highway, US Highway and Freeway, and vanishes on Interstate where both are about 81%.](figures/unibimulti-split.svg)

In the figure, each horizontal line is one road type; the wine dot is "no signal", the teal dot "signal". The gap survives in every road type except Interstate, so road type is not causing it — the traffic-signal effect is real. On interstates both sides sit near 81%: "saturation".

```python
pt = df.pivot_table(index="street_type", columns="traffic_signal",
                    values="severe", aggfunc="mean") * 100
pt["gap"] = pt[False] - pt[True]    # compare within the same row
print(pt.round(2))
```

The same test exposed features with **no** real effect: `give_way` had a raw gap of 16 points (18.29 vs 34.03), but within Local roads it was 14.55 vs 14.68. "It was local roads against motorways wearing a different label."

> **Key idea:** A pair of columns can show a strong relationship that is really a third column in disguise. Jag's wind result is the sharpest case: $r = +0.10$ over all data but $-0.23$ at winter 18:00 — the sign flipped once season was held fixed (Simpson's paradox).

## How to say it

> **Say it like this:** "I look at each column alone first, then each feature against the target using % severe against a baseline, judged by the size of the gap. Then I add a third variable — facets, hue or a two-key pivot table — to check whether the effect survives when a likely confounder, like road type or season, is held fixed."

## Quick reference

| Pair | Chart | Summary | Code |
|---|---|---|---|
| one numeric | histogram, box | median, IQR, skew | `sns.histplot`, `s.describe()` |
| one categorical | count plot | counts, % | `sns.countplot`, `value_counts` |
| num – num | scatter, regplot | Pearson / Spearman $r$ | `s1.corr(s2)` |
| cat – num | box, violin | group medians | `sns.boxplot`, `groupby().median()` |
| cat – cat | heatmap, stacked bar | row % | `pd.crosstab(..., normalize="index")` |
| feature – target | bars vs baseline | % target per value, gap | `groupby(f)["target"].mean()` |
| 3+ variables | hue, facets, pairplot | two-key table | `pivot_table(index, columns)` |
| Confounder | — | a third variable driving both | split test within its values |
