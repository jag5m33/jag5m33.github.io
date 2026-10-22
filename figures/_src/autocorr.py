import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf, pacf

rng = np.random.default_rng(0)
n = 48 * 7 * 8                                   # 8 weeks of half-hours
t = np.arange(n)
hh = t % 48                                       # half-hour of the day
dow = (t // 48) % 7                               # 0 = Monday
daily = -np.cos(2 * np.pi * hh / 48) * 4000 + np.where((hh > 32) & (hh < 40), 2500, 0)
weekend = np.where(dow >= 5, -3000, 0)
trend = -2 * t / 48                               # slow drift down
noise = np.zeros(n); e = rng.normal(0, 500, n)
for i in range(1, n): noise[i] = 0.7 * noise[i - 1] + e[i]
x = 27000 + daily + weekend + trend + noise
idx = pd.date_range("2019-03-04", periods=n, freq="30min")
s = pd.Series(x, index=idx)

# (a) the series: two weeks
fig, ax = plt.subplots(figsize=(10, 3.6))
w = s.iloc[:48 * 14]
ax.plot(w.index, w.values, color=TEAL, lw=1.3)
for d in range(14):
    if (d % 7) >= 5:
        ax.axvspan(idx[d * 48], idx[d * 48 + 47], color=LIGHT_GOLD, alpha=0.6, lw=0)
ax.annotate("", xy=(idx[48 + 36], w.iloc[48 + 36] + 1500), xytext=(idx[36], w.iloc[36] + 1500),
            arrowprops=dict(arrowstyle="<->", color=WINE))
ax.text(idx[42], w.iloc[36] + 2100, "48 steps = 1 day", color=WINE, fontsize=10, ha="center")
ax.text(idx[5 * 48 + 10], 20800, "weekend\n(lower)", color=MUTED, fontsize=9.5)
ax.set_ylabel("demand (MW)"); ax.set_title("Synthetic half-hourly demand: daily and weekly cycles")
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
ax.tick_params(axis="x", rotation=45)
save(fig, "autocorr-series")

# (b) lag scatter plots
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
sub = s.iloc[:48 * 21]
for ax, k, lab in zip(axes, [1, 24, 48], ["30 min", "12 hours", "1 day"]):
    a = sub.values[k:]; b = sub.values[:-k]
    pick = rng.choice(len(a), 600, replace=False)
    ax.scatter(b[pick], a[pick], s=7, color=TEAL, alpha=0.6)
    r = s.autocorr(k)                              # full-series value
    ax.set_title(f"lag {k} ({lab}): r = {r:.2f}", fontsize=11.5)
    ax.set_xlabel(f"$x_{{t-{k}}}$")
    lo, hi = sub.min(), sub.max(); ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
axes[0].set_ylabel("$x_t$")
fig.tight_layout()
save(fig, "autocorr-lagscatter")
for k in [1, 24, 48, 336]:
    print("lag", k, round(s.autocorr(k), 3))

# (c) ACF up to 700
r = acf(s.values, nlags=700)
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.vlines(np.arange(701), 0, r, color=TEAL, lw=0.8)
ax.axhline(0, color=MUTED, lw=0.8)
band = 1.96 / np.sqrt(n)
ax.axhspan(-band, band, color=LIGHT_TEAL, alpha=0.6, lw=0)
ax.annotate(f"lag 48 (1 day)\nr = {r[48]:.2f}", xy=(48, r[48]), xytext=(110, 1.18), fontsize=10, color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.annotate(f"lag 24 (half a day)\nr = {r[24]:.2f}", xy=(24, r[24]), xytext=(-5, -1.0), ha="left", va="center", bbox=dict(fc=PAPER, ec="none", pad=1), fontsize=10, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE))
ax.annotate(f"lag 336 (1 week)\nr = {r[336]:.2f}: higher than\nneighbouring daily peaks", xy=(336, r[336]), xytext=(390, 1.12),
            fontsize=10, color=WINE, arrowprops=dict(arrowstyle="->", color=WINE))
ax.set_ylim(-1.2, 1.55); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("lag k (half-hours)"); ax.set_ylabel("$r_k$")
ax.set_title("ACF: a peak every 48 lags, and a taller one at 336", pad=28)
save(fig, "autocorr-acf")
print("r48", r[48], "r24", r[24], "r336", r[336], "r288", r[288], "r384", r[384])

# (d) PACF
p = pacf(s.values, nlags=60, method="ywm")
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.vlines(np.arange(61), 0, p, color=GOLD, lw=1.6)
ax.axhline(0, color=MUTED, lw=0.8); ax.axhspan(-band, band, color=LIGHT_GOLD, alpha=0.6, lw=0)
ax.annotate(f"lag 1: {p[1]:.2f}", xy=(1, p[1]), xytext=(6, 0.85), fontsize=10, color=WINE, arrowprops=dict(arrowstyle="->", color=WINE))
ax.annotate(f"lag 2: {p[2]:.2f}", xy=(2, p[2]), xytext=(8, -0.32), fontsize=10, color=WINE, arrowprops=dict(arrowstyle="->", color=WINE))
ax.text(14, 0.45, "after lag 1 most bars are small: the big ACF values at\nlags 2, 3, 4 ... were mostly lag 1 passed along.\nSpikes return near lag 48 (the daily cycle).", fontsize=10, color=MUTED)
ax.set_xlabel("lag k (half-hours)"); ax.set_ylabel("partial $r_k$")
ax.set_title("PACF: the direct link at each lag, middle lags removed")
save(fig, "autocorr-pacf")
print("pacf", np.round(p[:6], 3), np.round(p[46:50], 3))

# (e) worked example
x8 = np.array([2, 4, 6, 5, 3, 4, 6, 6], float)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(1, 9), x8, marker="o", color=TEAL)
ax.axhline(x8.mean(), color=GOLD, ls="--"); ax.text(8.3, x8.mean() + 0.15, "mean 4.5", color=GOLD, va="center", fontsize=10)
for i, v in enumerate(x8):
    ax.vlines(i + 1, x8.mean(), v, color=WINE if v < x8.mean() else BLUE, lw=1.4, alpha=0.7)
ax.set_xlabel("t"); ax.set_ylabel("$x_t$"); ax.set_xticks(range(1, 9)); ax.set_xlim(0.5, 9.4)
ax.set_title("The 8-point example: deviations from the mean")
save(fig, "autocorr-worked")

# (f) residual after removing daily profile
prof = s.groupby([s.index.dayofweek >= 5, s.index.hour * 2 + s.index.minute // 30]).transform("median")
res = s - prof
rr = acf(res.values, nlags=700)
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.vlines(np.arange(701), 0, r, color=LIGHT_TEAL, lw=0.8, label="raw series")
ax.vlines(np.arange(701), 0, rr, color=WINE, lw=0.8, label="after removing the daily/weekly profile")
ax.axhline(0, color=MUTED, lw=0.8); ax.legend(loc="upper right")
ax.set_xlabel("lag k (half-hours)"); ax.set_ylabel("$r_k$")
ax.set_title("Remove the expected pattern: the cycles disappear from the ACF")
save(fig, "autocorr-residual")
print("res r48", rr[48], "r1", rr[1], "r336", rr[336])
