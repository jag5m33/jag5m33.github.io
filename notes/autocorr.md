# Autocorrelation

Autocorrelation measures how similar a time series is to a shifted copy of itself, and it is how you find the cycles in data like half-hourly electricity demand.

## Time series basics

- **Time series** = measurements of one thing taken in time order at a regular step. GB national demand (ND) is one value every half-hour.
- $x_t$ = the value at time step $t$. $x_{t-1}$ is the step before it.
- **Lag** = how many steps back you look. Lag $k$ compares $x_t$ with $x_{t-k}$. For half-hourly data, lag 1 = 30 minutes, lag 48 = one day, lag 336 = one week ($48 \times 7$).
- **Trend** = a slow, long-run drift up or down. ND fell from a median of about 29,700 MW (2018) to 25,300 MW (2023).
- **Seasonality** = a pattern that repeats with a fixed period: a **daily** cycle (low at 04:00, peak at 17:30–18:30), a **weekly** cycle (weekends lower) and a **yearly** cycle (winter high, summer low).
- **Noise** = what is left: small, irregular wiggles.

![Two weeks of synthetic half-hourly demand. Each day has a low night and a high evening peak; the shaded weekend days are lower. An arrow marks 48 steps as one day.](figures/autocorr-series.svg)

In the figure, the same shape repeats every 48 steps (one day), and the shaded weekend days sit lower — a pattern that repeats every 336 steps. This is synthetic data built to look like ND, so the numbers below can be computed exactly.

## The idea in one sentence

Correlation (see [Correlation](#/t/correlation)) compares **two different columns**. Autocorrelation compares **one column with itself, shifted by $k$ steps**. "Auto" = self.

## Lag scatter plots: seeing it before computing it

Take the series, shift it by $k$, and scatter $x_{t-k}$ (x-axis) against $x_t$ (y-axis):
- points along the rising diagonal → the value $k$ steps ago predicts today's value well → high positive autocorrelation;
- a cloud → no link;
- a falling cloud → negative autocorrelation (high then means low now).

![Three lag scatter plots of the synthetic demand. Lag 1 hugs the diagonal (r about 0.97); lag 24, half a day, slopes downward (r about -0.60); lag 48, one day, slopes upward again (r about 0.84).](figures/autocorr-lagscatter.svg)

In the figure:
- **Lag 1** (30 min): demand barely changes in half an hour, so points hug the line.
- **Lag 24** (12 hours): you are comparing daytime with night-time, so high goes with low: negative.
- **Lag 48** (one day): same time yesterday, so strongly positive again.

```python
import pandas as pd
import matplotlib.pyplot as plt

# one value per half-hour, in time order
s = df.set_index("timestamp")["nd"]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
for ax, k in zip(axes, [1, 24, 48]):
    ax.scatter(s.shift(k), s, s=3, alpha=0.3)  # x = k steps ago, y = now
    ax.set_title(f"lag {k}")
plt.show()

# pandas also has a one-liner for a single lag
pd.plotting.lag_plot(s, lag=48)
```

## The ACF formula, piece by piece

The **autocorrelation function** (ACF) gives one number $r_k$ for each lag $k$:

$$
r_k = \frac{\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{n} (x_t - \bar{x})^2}
$$

Read it in calculation order:

1. $\bar{x}$ = the **mean** of the whole series.
2. $x_t - \bar{x}$ = each point's **deviation**: positive if above average, negative if below.
3. $(x_t - \bar{x})(x_{t+k} - \bar{x})$ = multiply each deviation by the deviation $k$ steps later.
   - both above average → positive product;
   - both below → negative × negative = positive;
   - one above, one below → negative product.
4. **Numerator**: add those products. This is the **autocovariance** at lag $k$ (times $n$) — "do the point and the point $k$ steps later tend to sit on the same side of the mean?". There are only $n-k$ pairs, because the last $k$ points have no partner.
5. **Denominator**: the sum of squared deviations — the same quantity at lag 0, i.e. the series compared with itself. Dividing by it scales the answer to lie between −1 and +1.

**Where it comes from.** It is Pearson's correlation $r = \frac{\sum (x-\bar{x})(y-\bar{y})}{\sqrt{\sum (x-\bar{x})^2 \sum (y-\bar{y})^2}}$ with $y$ replaced by the shifted series. Because both are the **same** series, we use one mean and one total variance for both, so the square root of the product becomes just $\sum (x_t - \bar{x})^2$.

So $r_0 = 1$ always (a series is perfectly correlated with itself), and $r_k$ is the fraction of that "self-similarity" left after shifting by $k$.

## Worked example: 8 numbers, lag 1 by hand

Let's say we have 8 readings: 2, 4, 6, 5, 3, 4, 6, 6.

![The eight values plotted in order with a dashed line at the mean of 4.5 and vertical lines showing each deviation.](figures/autocorr-worked.svg)

In the figure, the vertical lines are the deviations $x_t - \bar{x}$; below the dashed mean line they are negative.

**Step 1 — mean.** $\bar{x} = (2+4+6+5+3+4+6+6)/8 = 36/8 = 4.5$.

**Step 2 — deviations, squares and lag-1 products.**

| $t$ | $x_t$ | $d_t = x_t - 4.5$ | $d_t^2$ | $d_{t+1}$ | $d_t \times d_{t+1}$ |
|---|---|---|---|---|---|
| 1 | 2 | −2.5 | 6.25 | −0.5 | 1.25 |
| 2 | 4 | −0.5 | 0.25 | 1.5 | −0.75 |
| 3 | 6 | 1.5 | 2.25 | 0.5 | 0.75 |
| 4 | 5 | 0.5 | 0.25 | −1.5 | −0.75 |
| 5 | 3 | −1.5 | 2.25 | −0.5 | 0.75 |
| 6 | 4 | −0.5 | 0.25 | 1.5 | −0.75 |
| 7 | 6 | 1.5 | 2.25 | 1.5 | 2.25 |
| 8 | 6 | 1.5 | 2.25 | — | — |
| **Sum** | | 0 | **16.00** | | **2.75** |

**Step 3 — divide.** $r_1 = 2.75 / 16 = 0.172$.

**In words:** a reading tells you only a little about the next one; the series zig-zags too much for neighbours to be strongly alike.

Doing the same with lag 2 gives $r_2 = -0.594$: values two steps apart tend to sit on **opposite** sides of the mean (the series goes up-down roughly every two steps).

**Check in Python:**

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

x = np.array([2, 4, 6, 5, 3, 4, 6, 6], dtype=float)

# deviations from the mean (4.5)
d = x - x.mean()
# lag-1 products / total sum of squares
r1 = (d[:-1] * d[1:]).sum() / (d ** 2).sum()
print(round(r1, 3))                          # 0.172

print(acf(x, nlags=2).round(3))              # [1.    0.172 -0.594]

# 0.301 -- different! see the trap below
print(round(pd.Series(x).autocorr(lag=1), 3))
```

> **Trap:** `pd.Series.autocorr(lag)` gives **0.301**, not 0.172. Pandas computes an ordinary Pearson correlation between the two overlapping pieces (`x[:-1]` and `x[1:]`), each with **its own** mean and SD. The ACF formula uses the whole-series mean and variance. On 8 points the difference is large; on 100,000 half-hours it is negligible. Use `statsmodels.acf` when you want the textbook ACF.

## Reading an ACF plot

An **ACF plot** (correlogram) draws $r_k$ as a bar for $k = 0, 1, 2, \dots$. The shaded band around 0 starts at about $\pm 1.96/\sqrt{n}$ at lag 1 and widens with the lag (statsmodels uses Bartlett's formula by default; pass `bartlett_confint=False` for a constant $\pm 1.96/\sqrt{n}$ band). Bars inside it are what pure noise would give.

![ACF of the synthetic demand up to lag 700: a large positive spike every 48 lags, negative dips at half-day lags like 24, and a taller spike at lag 336 than its neighbouring daily peaks.](figures/autocorr-acf.svg)

In the figure, follow the arrows:
- **Lag 48** ($r \approx 0.82$): same half-hour yesterday. A spike repeats every 48 lags, so the daily cycle repeats reliably.
- **Lag 24** ($r \approx -0.60$): half a day back compares day with night.
- **Lag 336** ($r \approx 0.83$): one week back — same half-hour, **same weekday**. It stands taller than the daily peaks either side (lags 288 and 384, both about 0.71–0.73), because comparing Saturday with Saturday removes the weekday/weekend difference.
- The peaks slowly shrink with lag because of the trend and noise.
- These ACF values come from `statsmodels`. The lag-scatter titles above used `pandas.autocorr`, so they differ slightly (0.84 vs 0.82 at lag 48), for the reason given in the trap above.

```python
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df_merged.nd, lags=700)     # from his elec notebook
plt.xlabel("lags")
plt.ylabel("correlation (r)")
plt.show()

# or numbers for chosen lags
for k in [1, 24, 48, 336]:
    print(k, round(df_merged.nd.autocorr(lag=k), 3))
```

**On the real data**, Jag's notes on his ACF ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)):
- at each peak (whole days back), $r$ is about 0.8–0.9: "same time on a different day, so very similar";
- between peaks (half a day back), $r$ drops to about 0.05–0.1: "daytime compared with night-time, so very different";
- 14 bumps = 14 days: "the daily cycle repeats reliably".

He used lags 48 and 336 as the two periods for the decomposition step (MSTL) later.

The synthetic series dips to about −0.60 at half-day lags rather than near 0, because its daily cycle was built as a symmetric wave; real demand's day–night shape is less symmetric, so its half-day value need not go as negative.

## PACF, briefly

- **Partial autocorrelation** (PACF) at lag $k$ = the correlation between $x_t$ and $x_{t-k}$ **after removing** what the lags in between (1 to $k-1$) already explain.
- Why it helps: if lag 1 is strong, lag 2 looks strong too, simply because $x_t$ is like $x_{t-1}$ which is like $x_{t-2}$. PACF strips that chain out and shows the **direct** link.

![PACF of the synthetic demand up to lag 60: a large bar at lag 1 (about 0.97), small negative bars at lags 2 to 5, and further spikes near lag 48.](figures/autocorr-pacf.svg)

In the figure, lag 1 dominates (0.97). Most of the large ACF values at lags 2, 3, 4… were lag 1 "passed along". Spikes reappear around lag 48 because the daily cycle is a direct link of its own.

```python
from statsmodels.graphics.tsaplots import plot_pacf
# Yule-Walker method, fast on long series
plot_pacf(s, lags=60, method="ywm")
plt.show()
```

PACF is used mainly to choose the order of autoregressive (AR) models; for EDA the ACF is the main tool.

## Why autocorrelation matters

### 1. Observations are not independent

- Many methods assume rows are **i.i.d.** — independent and identically distributed. Independent means one row tells you nothing about the next.
- With $r_1 \approx 0.97$, each half-hour almost tells you the next. You have far less independent information than the row count suggests, so standard errors, confidence intervals and p-values computed as if rows were independent are too optimistic.

### 2. Random train/test splits leak

- A random split puts 10:00 in train and 10:30 in test. Because they are almost identical, the model is tested on near-copies of training rows and the score comes out falsely high — **leakage**.
- **Fix:** split by time. Train on the past, validate and test on later periods.

```python
# make sure rows are in time order
s = s.sort_index()
# everything up to the end of 2021
train = s[:"2021-12-31"]
val   = s["2022-01-01":"2022-12-31"]       # the next year
# the final year, untouched until the end
test  = s["2023-01-01":]

# for cross-validation, sklearn has a time-aware splitter
from sklearn.model_selection import TimeSeriesSplit
for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(s):
    # each test fold is later than its train fold
    pass
```

> **Trap:** The same issue applies whenever rows carry time. Jag's accidents README lists it as a limitation: rows were split at random rather than by date, "so the validation score doesn't show performance on future years".

### 3. For anomaly detection, remove the expected pattern first

- A value of 30,000 MW is normal on a winter night but high on a summer night. Judging it against the overall average would flag every winter evening.
- So: model the **expected** pattern (trend + daily + weekly + yearly), subtract it, and look for anomalies in the **residual** (what's left).

![ACF of the raw synthetic series (light) with daily spikes, overlaid with the ACF of the residual after subtracting a typical weekday/weekend half-hour profile (dark), where the spikes at 48 and 336 vanish.](figures/autocorr-residual.svg)

In the figure, the light bars are the raw ACF with its daily spikes. The dark bars are after subtracting a typical profile for each half-hour (weekday and weekend separately): the spikes at 48 and 336 are gone ($r_{48} \approx 0.02$). What remains is short-range: neighbouring half-hours still move together ($r_1 \approx 0.69$), which is why anomalies usually show up as a **run** of half-hours rather than one.

```python
# crude version of "remove the expected pattern": subtract a typical profile
key = [s.index.dayofweek >= 5,                    # weekend or not
       s.index.hour * 2 + s.index.minute // 30]   # half-hour slot 0-47
# typical value for that slot
expected = s.groupby(key).transform("median")
resid = s - expected                              # the unexplained part

# daily/weekly spikes should vanish
plot_acf(resid, lags=700); plt.show()
```

Jag's plan uses MSTL instead of this simple profile: "ACF measures how often the patterns repeat: every 48 half-hours (daily) and every 336 (weekly). MSTL uses those two numbers to separate the patterns from the data. What's left is the leftover (residual)… When the residual is much bigger than usual… it gets flagged as an anomaly."

## How to say it

> **Say it like this:** "Autocorrelation is the correlation of a series with itself shifted by k steps. For half-hourly demand the ACF peaks at lag 48 and higher at 336, confirming daily and weekly cycles. It matters because rows aren't independent, so I split by time, and for anomaly detection I remove those cycles first and look for unusual residuals."

## Assumptions & caveats

- **Regular spacing.** Lags only mean "one day" if every step is exactly 30 minutes. Jag rebuilt the timestamp in GMT because clock-change days have 46 or 50 periods.
- **Stationarity.** The ACF assumes the mean and variance don't drift. A strong trend makes the ACF decay slowly and can hide cycles; remove the trend first if needed.
- **Linear only.** Like Pearson's $r$, it measures straight-line similarity.
- **Long lags are noisy.** At lag $k$ only $n-k$ pairs remain.

## Quick reference

| Term | Meaning | Code |
|---|---|---|
| Lag $k$ | compare $x_t$ with $x_{t-k}$ | `s.shift(k)` |
| Trend | slow long-run drift | — |
| Seasonality | repeating pattern with fixed period (48, 336) | — |
| Autocovariance | $\sum (x_t-\bar{x})(x_{t+k}-\bar{x})$ | — |
| ACF $r_k$ | autocovariance / total sum of squares | `statsmodels.tsa.stattools.acf` |
| pandas autocorr | Pearson on the two overlapping pieces | `s.autocorr(lag=k)` |
| ACF plot | $r_k$ against $k$, with noise band | `plot_acf(s, lags=700)` |
| PACF | direct link at lag $k$, middle lags removed | `plot_pacf(s, lags=60)` |
| Lag scatter | $x_{t-k}$ vs $x_t$ | `pd.plotting.lag_plot(s, lag=k)` |
| Time split | train on past, test on future | `TimeSeriesSplit` |
| Residual | data minus expected pattern | `s - expected` |
