import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np, pandas as pd

rng = np.random.default_rng(0)

# (1) median imputation squashes the distribution
x = rng.normal(60, 15, 1000)
miss = rng.random(1000) < 0.3
xo = x[~miss]; med = np.median(xo)
xi = np.where(miss, med, x)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
bins = np.arange(10, 112, 3)
axes[0].hist(xo, bins=bins, color=LIGHT_TEAL, edgecolor=TEAL)
axes[0].set_title(f"Observed values: SD = {xo.std(ddof=1):.1f}")
axes[1].hist(xi, bins=bins, color=LIGHT_TEAL, edgecolor=TEAL)
axes[1].set_title(f"After median imputation: SD = {xi.std(ddof=1):.1f}")
axes[1].annotate(f"{miss.sum()} rows all set\nto the median ({med:.1f})", xy=(med, 300), xytext=(78, 250), fontsize=10, color=WINE,
                 arrowprops=dict(arrowstyle="->", color=WINE))
for ax in axes: ax.set_xlabel("temperature (F)")
axes[0].set_ylabel("count")
fig.tight_layout()
save(fig, "cleaning-impute")
print("missing", miss.sum(), "sd", xo.std(ddof=1), xi.std(ddof=1), "med", med)

# (2) MCAR / MAR / MNAR
n = 300
temp = rng.normal(60, 15, n)
remote = rng.random(n) < 0.3
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
cases = [
    ("MCAR: missing at random\n(coin flip)", rng.random(n) < 0.3),
    ("MAR: depends on another column\n(remote site → no station)", remote & (rng.random(n) < 0.9)),
    ("MNAR: depends on the value itself\n(sensor fails when very cold)", temp < 45),
]
for ax, (title, m) in zip(axes, cases):
    jitter = rng.uniform(-0.25, 0.25, n)
    ypos = np.where(remote, 1, 0) + jitter
    ax.scatter(temp[~m], ypos[~m], s=12, color=TEAL, alpha=0.7, label="recorded")
    ax.scatter(temp[m], ypos[m], s=14, facecolor="none", edgecolor=WINE, label="missing")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("true temperature (F)")
axes[0].set_yticks([0, 1], ["city", "remote"])
axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
fig.tight_layout()
save(fig, "cleaning-mcar")

# (3) concat with different columns
fig, ax = plt.subplots(figsize=(9, 2.9)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(1.2, 4.8)
cols = ["date", "period", "nd", "nsl_flow", "viking_flow"]
def table(x0, y0, header, rows, fill_missing):
    w = 1.25
    for j, c in enumerate(header):
        ax.add_patch(plt.Rectangle((x0 + j * w, y0), w, 0.45, fc=LIGHT_GOLD, ec=MUTED, lw=0.8))
        ax.text(x0 + j * w + w / 2, y0 + 0.22, c, ha="center", va="center", fontsize=9)
    for i, (lab, missing) in enumerate(rows):
        for j, c in enumerate(header):
            miss = c in missing
            ax.add_patch(plt.Rectangle((x0 + j * w, y0 - 0.45 * (i + 1)), w, 0.45,
                                       fc=LIGHT_WINE if miss else "none", ec=MUTED, lw=0.8))
            val = {"date": lab, "period": str(i % 2 + 1), "nd": "value", "nsl_flow": "value", "viking_flow": "value"}[c]
            ax.text(x0 + j * w + w / 2, y0 - 0.45 * (i + 1) + 0.22, "NaN" if miss else val, ha="center", va="center",
                    fontsize=9, color=WINE if miss else INK)
table(0.6, 4.2, cols, [("2018-01-01", ["nsl_flow", "viking_flow"]), ("2018-01-01", ["nsl_flow", "viking_flow"]),
                       ("2023-01-01", []), ("2023-01-01", [])], None)
ax.text(0.2, 3.75, "2018\nrows", fontsize=9.5, color=MUTED, ha="center", va="center")
ax.text(0.2, 2.85, "2023\nrows", fontsize=9.5, color=MUTED, ha="center", va="center")
ax.text(7.2, 3.2, "pd.concat matches columns\nby NAME. A column that a\nfile does not have is\nfilled with NaN.", fontsize=10, color=INK, va="center")
ax.text(3.7, 2.2, "Blanks here are 'did not exist yet', not 'failed to record'.\nJag's elec data: nsl, eleclink, viking, greenlink blank in 2018;\nscottish_transfer only in 2023.",
        fontsize=9.5, color=MUTED, ha="center", va="top")
save(fig, "cleaning-concat")
