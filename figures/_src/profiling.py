import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np

# (1) Missing values in the accidents data (real counts from notes.txt 1.7)
N = 7_728_394
miss = {"wind_direction": 175_206, "weather_condition": 173_459, "weather_timestamp": 120_228,
        "airport_code": 22_635, "street": 10_869, "timezone": 7_808}
names = list(miss)[::-1]; vals = [miss[k] / N * 100 for k in names]
fig, ax = plt.subplots(figsize=(8, 3.6))
cols = [TEAL if k in ("airport_code", "street", "timezone") else WINE for k in names]
ax.barh(names, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(v + 0.03, i, f"{v:.2f}%  ({miss[names[i]]:,})", va="center", fontsize=10, color=INK)
ax.set_xlabel("% of 7,728,394 rows missing")
ax.set_xlim(0, 3.4)
ax.set_title("Missing values cluster in the weather columns")
ax.text(1.9, 1.0, "weather columns\n(shared cause?)", color=WINE, fontsize=10)
save(fig, "profiling-missing")

# (2) Cardinality on a log scale (real nunique from notes.txt)
card = {"country": 1, "turning_loop": 1, "source": 3, "timezone (raw)": 7,
        "weather_condition": 144, "street": 336_306, "zipcode": 825_294}
fig, ax = plt.subplots(figsize=(8, 3.8))
k = list(card); v = [card[x] for x in k]
colours = [MUTED, MUTED, TEAL, TEAL, GOLD, WINE, WINE]
ax.barh(k, v, color=colours)
ax.set_xscale("log")
for i, x in enumerate(v):
    ax.text(x * 1.3, i, f"{x:,}", va="center", fontsize=10)
ax.set_xlim(0.8, 2e7)
ax.set_xlabel("number of unique values, nunique() (log scale)")
ax.set_title("Cardinality: from useless constants to useless IDs")
ax.text(2e3, 0.6, "1 value = no information\n(dropped)", color=MUTED, fontsize=10, va="center")
ax.annotate("too many to learn from\n(dropped / reduced)", xy=(3e5, 5), xytext=(2e3, 3.4), color=WINE, fontsize=10, va="center",
            arrowprops=dict(arrowstyle="->", color=WINE))
save(fig, "profiling-cardinality")

# (3) Impossible values: a temperature column with planted errors
rng = np.random.default_rng(0)
t = rng.normal(62, 18, 3000)
t = np.concatenate([t, [-90, -90, 160, 160, 158]])
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(t, bins=np.arange(-100, 175, 5), color=LIGHT_TEAL, edgecolor=TEAL)
ax.axvspan(-110, -70, color=LIGHT_WINE, alpha=0.7)
ax.axvspan(134, 180, color=LIGHT_WINE, alpha=0.7)
ax.set_xlim(-105, 175)
# mark the bad values so they are visible despite tiny counts
ax.scatter([-90, 160], [8, 8], marker="v", s=90, color=WINE, zorder=5)
ax.annotate("-90 F: impossible", xy=(-90, 14), xytext=(-95, 120), color=WINE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.annotate("160 F: impossible", xy=(160, 14), xytext=(120, 150), color=WINE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.text(-67, 200, "physical limits\n-70 to 134 F", color=MUTED, fontsize=10)
ax.set_xlabel("temperature (F)"); ax.set_ylabel("count")
ax.set_title("Judge extremes against physical limits, not whiskers")
save(fig, "profiling-impossible")
