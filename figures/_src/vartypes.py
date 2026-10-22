import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np, pandas as pd, seaborn as sns

rng = np.random.default_rng(0)

# (1) the type tree
fig, ax = plt.subplots(figsize=(9, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
def box(x, y, t, c, w=1.9, fc=None):
    ax.add_patch(plt.Rectangle((x - w/2, y - 0.32), w, 0.64, fc=fc or "none", ec=c, lw=1.8))
    ax.text(x, y, t, ha="center", va="center", fontsize=11, color=INK)
def line(x1, y1, x2, y2): ax.plot([x1, x2], [y1 + 0.32 if y1 < y2 else y1 - 0.32, y2 - 0.32 if y1 < y2 else y2 + 0.32], color=MUTED, lw=1.2)
box(5, 4.4, "Variable", INK, w=1.6)
box(2.6, 3.1, "Categorical\n(labels)", WINE, w=2.2, fc=LIGHT_WINE)
box(7.4, 3.1, "Numerical\n(quantities)", TEAL, w=2.2, fc=LIGHT_TEAL)
line(5, 4.4, 2.6, 3.1); line(5, 4.4, 7.4, 3.1)
for x, t in [(1.0, "Nominal"), (2.6, "Ordinal"), (4.2, "Binary")]:
    box(x, 1.7, t, WINE, w=1.4); line(2.6, 3.1, x, 1.7)
for x, t in [(6.4, "Discrete"), (8.6, "Continuous")]:
    box(x, 1.7, t, TEAL, w=1.6); line(7.4, 3.1, x, 1.7)
ex = {1.0: "state,\nweather", 2.6: "severity 1-4,\nlight_score", 4.2: "traffic_signal\nTrue/False",
      6.4: "count of\nconstraints", 8.6: "temperature,\nnational demand"}
for x, t in ex.items():
    ax.text(x, 0.75, t, ha="center", va="center", fontsize=9.5, color=MUTED, style="italic")
ax.text(5, 0.05, "examples from Jag's projects", ha="center", fontsize=9, color=MUTED)
save(fig, "vartypes-tree")

# (2) right chart per type grid
weather = rng.choice(["fair", "cloudy", "rain", "fog", "snow"], 600, p=[.45, .25, .17, .08, .05])
sev = rng.choice([1, 2, 3, 4], 600, p=[.01, .8, .165, .025])
temp = rng.normal(62, 17, 600)
cc = rng.choice([0, 1, 2, 3, 4], 600, p=[.62, .22, .1, .045, .015])

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax = axes[0, 0]
order = pd.Series(weather).value_counts().index
sns.countplot(x=weather, order=order, color=WINE, ax=ax)
ax.set_title("Nominal: bars sorted by count"); ax.set_xlabel("weather"); ax.set_ylabel("count")

ax = axes[0, 1]
s = pd.Series(sev).value_counts().reindex([1, 2, 3, 4], fill_value=0)
ax.bar(["1", "2", "3", "4"], s.values, color=[LIGHT_WINE, WINE, WINE, WINE])
ax.set_title("Ordinal: bars in natural order"); ax.set_xlabel("severity (1 = least)"); ax.set_ylabel("count")
ax.annotate("keep 1 → 4 order,\nnot sorted by height", xy=(2.9, s[3] + 10), xytext=(1.7, 330), fontsize=9.5, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED))

ax = axes[1, 0]
sns.histplot(temp, bins=30, kde=True, color=TEAL, ax=ax, edgecolor="white")
ax.set_title("Continuous: histogram + KDE"); ax.set_xlabel("temperature (F)"); ax.set_ylabel("count")
ax.text(0.98, 0.92, "choose bins;\nKDE = smoothed\noutline", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=MUTED)

ax = axes[1, 1]
vc = pd.Series(cc).value_counts().sort_index()
ax.bar(vc.index, vc.values, width=0.6, color=GOLD)
ax.set_xticks(range(5))
ax.set_title("Discrete: one bar per integer"); ax.set_xlabel("constraint_count"); ax.set_ylabel("count")
ax.text(0.98, 0.9, "gaps between bars:\nno value 1.5 exists", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=MUTED)
fig.tight_layout()
save(fig, "vartypes-grid")

# (3) box plot for continuous + what it hides
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
bimodal = np.concatenate([rng.normal(21000, 2000, 300), rng.normal(31000, 2500, 300)])
ax = axes[0]
sns.boxplot(x=bimodal, color=LIGHT_TEAL, ax=ax, linecolor=TEAL, width=0.4)
ax.set_title("Box plot: median, IQR, whiskers"); ax.set_xlabel("demand (MW)")
ax = axes[1]
sns.histplot(bimodal, bins=40, color=TEAL, ax=ax, edgecolor="white")
ax.set_title("Same data as a histogram: two peaks"); ax.set_xlabel("demand (MW)"); ax.set_ylabel("count")
ax.annotate("night", xy=(21000, 30), xytext=(13500, 36), color=WINE, fontsize=10, arrowprops=dict(arrowstyle="->", color=WINE))
ax.annotate("day", xy=(31000, 28), xytext=(36000, 36), color=WINE, fontsize=10, arrowprops=dict(arrowstyle="->", color=WINE))
fig.tight_layout()
save(fig, "vartypes-box-hides")

# (4) charts to avoid
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
ax = axes[0]
labs = ["fair", "cloudy", "rain", "fog", "snow"]
cnt = pd.Series(weather).value_counts().reindex(labs)
ax.plot(labs, cnt.values, marker="o", color=WINE)
ax.set_title("Avoid: line through nominal labels")
ax.text(0.5, 0.75, "the line implies an order\nand in-between values\nthat do not exist", transform=ax.transAxes, fontsize=9.5, color=MUTED)
ax.set_ylabel("count")
ax = axes[1]
ax.hist(temp, bins=3, color=LIGHT_TEAL, edgecolor=TEAL)
ax.set_title("Avoid: too few bins"); ax.set_xlabel("temperature (F)")
ax.text(0.98, 0.6, "3 bins hide\nthe shape", transform=ax.transAxes, ha="right", fontsize=9.5, color=MUTED)
fig.tight_layout()
save(fig, "vartypes-avoid")
