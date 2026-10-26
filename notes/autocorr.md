# Autocorrelation

Autocorrelation measures how similar a time series is to a shifted copy of itself, and it is how you find the cycles in data like half-hourly electricity demand.

## Time series basics

- **Time series** = measurements of one thing taken in time order at a regular step. GB national demand (ND) is one value every half-hour.
- $x_t$ = the value at time step $t$.
- **Lag** = how many steps back you look. Lag $k$ compares $x_t$ with $x_{t-k}$. For half-hourly data:
  - lag 1 = 30 minutes;
  - lag 48 = one day;
  - lag 336 = one week ($48 \times 7$).
- **Trend** = a slow drift up or down. ND fell from a median of about 29,700 MW (2018) to 25,300 MW (2023).
- **Seasonality** = a pattern that repeats with a fixed period: daily (low at 04:00, peak around 18:00), weekly (weekends lower), yearly (winter high).
- **Noise** = the small, irregular wiggles left over.

![Two weeks of synthetic half-hourly demand. Each day has a low night and a high evening peak; the shaded weekend days are lower. An arrow marks 48 steps as one day.](figures/autocorr-series.svg)

In the figure, the same shape repeats every 48 steps (one day), and the shaded weekend days sit lower — a pattern that repeats every 336 steps. The data is synthetic, built to look like ND.

## The idea

[Correlation](#/t/correlation) compares **two different columns**. Autocorrelation compares **one column with itself, shifted by $k$ steps**. "Auto" = self.

To see it, scatter the value $k$ steps ago (x-axis) against the value now (y-axis):
- points along the rising diagonal → strong positive autocorrelation (the past predicts now);
- a shapeless cloud → no link;
- a falling cloud → negative (high then means low now).

![Three lag scatter plots of the synthetic demand. Lag 1 hugs the diagonal (r about 0.97); lag 24, half a day, slopes downward (r about -0.60); lag 48, one day, slopes upward again (r about 0.84).](figures/autocorr-lagscatter.svg)

In the figure:
- **Lag 1** (30 min): demand barely changes in half an hour, so points hug the line.
- **Lag 24** (12 hours): day compared with night, so high goes with low — negative.
- **Lag 48** (one day): same time yesterday, so strongly positive again.

```python
import pandas as pd
s = df.set_index("timestamp")["nd"]     # one value per half-hour, in time order
pd.plotting.lag_plot(s, lag=48)         # x = 48 steps ago, y = now
```

## The formula

$$
r_k = \frac{\sum d_t \, d_{t+k}}{\sum d_t^2}
$$

- $d_t = x_t - \bar{x}$ = how far each point sits above (+) or below (−) the series mean $\bar{x}$.
- **Top:** multiply each deviation by the one $k$ steps later, then add. Both above the mean, or both below, gives a positive product. One above and one below gives a negative product.
- **Bottom:** the same sum with no shift (the series against itself). Dividing by it keeps $r_k$ between −1 and +1.

It is just Pearson's $r$ with the series playing both columns. So $r_0 = 1$ always.

### Worked example: 8 numbers, lag 1

Let's say we have 8 readings: 2, 4, 6, 5, 3, 4, 6, 6.

![The eight values plotted in order with a dashed line at the mean of 4.5 and vertical lines showing each deviation.](figures/autocorr-worked.svg)

In the figure, the vertical lines are the deviations $d_t$; below the dashed mean line they are negative.

**Step 1 — mean.** $36 / 8 = 4.5$.

**Step 2 — deviations and products.**

| $t$ | $x_t$ | $d_t$ | $d_t^2$ | $d_t \times d_{t+1}$ |
|---|---|---|---|---|
| 1 | 2 | −2.5 | 6.25 | (−2.5)(−0.5) = 1.25 |
| 2 | 4 | −0.5 | 0.25 | (−0.5)(1.5) = −0.75 |
| 3 | 6 | 1.5 | 2.25 | (1.5)(0.5) = 0.75 |
| 4 | 5 | 0.5 | 0.25 | (0.5)(−1.5) = −0.75 |
| 5 | 3 | −1.5 | 2.25 | (−1.5)(−0.5) = 0.75 |
| 6 | 4 | −0.5 | 0.25 | (−0.5)(1.5) = −0.75 |
| 7 | 6 | 1.5 | 2.25 | (1.5)(1.5) = 2.25 |
| 8 | 6 | 1.5 | 2.25 | — |
| **Sum** | | | **16.00** | **2.75** |

**Step 3 — divide.** $r_1 = 2.75 / 16 = 0.172$.

**In words:** one reading tells you only a little about the next; the series zig-zags too much.

```python
import numpy as np
from statsmodels.tsa.stattools import acf

x = np.array([2, 4, 6, 5, 3, 4, 6, 6], dtype=float)
print(acf(x, nlags=1).round(3))      # [1.    0.172]
```

> **Trap:** pandas' `s.autocorr(lag=1)` gives 0.301 here, because it uses a slightly different recipe. On tiny data that matters; on thousands of half-hours the two agree closely.

## Reading an ACF plot

The **ACF plot** (autocorrelation function) draws $r_k$ as a bar for every lag $k$. The shaded band near 0 is what pure random noise would give, so bars inside it mean "no real link".

![ACF of the synthetic demand up to lag 700: a large positive spike every 48 lags, negative dips at half-day lags like 24, and a taller spike at lag 336 than its neighbouring daily peaks.](figures/autocorr-acf.svg)

In the figure, follow the arrows:
- **Peak at 48** (about 0.8): same half-hour yesterday. A peak every 48 lags = a **daily cycle**.
- **Dip at 24**: half a day back compares day with night.
- **Taller peak at 336**: one week back — same half-hour **and** same weekday. It beats the daily peaks either side because Saturday is compared with Saturday. That is the **weekly cycle**.

```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_merged.nd, lags=700)     # from his elec notebook
```

**On the real data**, Jag noted ([eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)):
- at each peak (whole days back) $r$ is about 0.8–0.9: "same time on a different day, so very similar";
- between peaks (half a day back) it drops to about 0.05–0.1: "daytime compared with night-time, so very different";
- 14 bumps = 14 days: "the daily cycle repeats reliably".

He uses 48 and 336 as the two cycle lengths for the next step (MSTL).

## Why it matters

### 1. Rows are not independent

Neighbouring half-hours are almost the same ($r_1 \approx 0.97$), so 100,000 rows hold much less independent information than it seems.

### 2. Random train/test splits leak

A random split puts 10:00 in train and 10:30 in test. They are near-copies, so the test score comes out falsely high — **leakage**. **Fix:** split by time — train on the past, test on later periods.

```python
s = s.sort_index()                    # rows in time order
train = s[:"2021-12-31"]              # the past
val   = s["2022-01-01":"2022-12-31"]  # the next year
test  = s["2023-01-01":]              # final year, untouched until the end
```

> **Trap:** This applies whenever rows carry time. Jag's accidents README lists it as a limitation: rows were split at random, "so the validation score doesn't show performance on future years".

### 3. For anomaly detection, remove the cycles first

30,000 MW is normal on a winter evening but high on a summer night. So: work out the **expected** pattern, subtract it, and look for anomalies in the **residual** (what's left).

![ACF of the raw synthetic series (light) with daily spikes, overlaid with the ACF of the residual after subtracting a typical weekday/weekend half-hour profile (dark), where the spikes at 48 and 336 vanish.](figures/autocorr-residual.svg)

In the figure, the light bars are the raw ACF with its daily peaks. The dark bars are after subtracting a typical value for each half-hour: the peaks at 48 and 336 are gone. Neighbouring half-hours still move together, which is why anomalies usually show up as a **run** of half-hours.

Jag's plan: "ACF measures how often the patterns repeat: every 48 half-hours (daily) and every 336 (weekly). MSTL uses those two numbers to separate the patterns from the data. What's left is the leftover (residual)… When the residual is much bigger than usual… it gets flagged as an anomaly."

## Caveats

- **Regular spacing.** Lag 48 only means "one day" if every step is exactly 30 minutes. Jag rebuilt the timestamp in GMT because clock-change days have 46 or 50 periods.
- **Trend can hide cycles.** A strong drift makes the ACF fade slowly; remove the trend first if needed.
- **Straight-line similarity only**, like Pearson's $r$.

> **Say it like this:** "Autocorrelation is the correlation of a series with itself shifted by k steps. For half-hourly demand the ACF peaks every 48 lags and higher at 336, so there are daily and weekly cycles. Because rows aren't independent I split by time, and for anomaly detection I remove those cycles and look at the residual."

## Quick reference

| Term | Meaning | Code |
|---|---|---|
| Lag $k$ | compare now with $k$ steps ago | `s.shift(k)` |
| Seasonality | repeating pattern (48 = day, 336 = week) | — |
| $r_k$ | correlation of the series with itself at lag $k$ | `acf(x, nlags=k)` |
| ACF plot | $r_k$ for every lag, with a noise band | `plot_acf(s, lags=700)` |
| Lag scatter | value $k$ steps ago vs now | `pd.plotting.lag_plot(s, lag=k)` |
| Time split | train on past, test on future | `s[:"2021-12-31"]` |
| Residual | data minus expected pattern | `s - expected` |
