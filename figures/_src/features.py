import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from math import comb
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 1. degree 1 vs 2 vs 3 vs 12
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 10, 14))
y = 2 + 0.3*(x - 5)**2 + rng.normal(0, 0.9, x.size)
xx = np.linspace(0, 10, 300)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
for ax, d, col, verdict in zip(axes, [1, 2, 3, 12], [WINE, TEAL, BLUE, ORANGE],
                               ["underfits (high bias)", "just right", "also fine", "overfits (high variance)"]):
    m = make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression()).fit(x[:, None], y)
    ax.scatter(x, y, color=INK, s=22, zorder=3)
    ax.plot(xx, m.predict(xx[:, None]), color=col)
    ax.set_ylim(-1, 12); ax.set_title(f"degree {d}", color=col)
    ax.text(5, 11.2, verdict, ha="center", fontsize=10, color=MUTED)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.tight_layout(); save(fig, "features-poly")

# 2. feature explosion
fig, ax = plt.subplots(figsize=(8, 3.8))
ds = np.arange(1, 6)
for n, col in zip([2, 5, 10, 20], [TEAL, GOLD, WINE, BLUE]):
    k = [comb(n + d, d) - 1 for d in ds]
    ax.plot(ds, k, marker="o", color=col, label=f"{n} original features")
    ax.text(5.08, k[-1], f"{k[-1]:,}", va="center", fontsize=10, color=col)
ax.set_yscale("log"); ax.set_xticks(ds); ax.set_xlim(0.8, 5.7)
ax.set_xlabel("polynomial degree"); ax.set_ylabel("number of features (log scale)")
ax.set_title("PolynomialFeatures: columns grow combinatorially")
ax.legend(loc="upper left", fontsize=10)
save(fig, "features-explosion")

# 3. leakage trap in feature selection
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
def box(ax, x, y, w, h, txt, fc, ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=10)
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
a = axes[0]; a.set_title("Wrong: select on ALL rows, then cross-validate", color=WINE)
box(a, 0.3, 4.3, 9.4, 1.0, "all data (train + validation folds)", LIGHT_WINE)
arrow(a, 5, 4.3, 5, 3.6)
box(a, 1.5, 2.6, 7, 1.0, "pick the 'best' features using every y", LIGHT_WINE)
arrow(a, 5, 2.6, 5, 1.9)
box(a, 1.5, 0.8, 7, 1.0, "CV score looks great (validation already seen)", "none", WINE)
b = axes[1]; b.set_title("Right: selection is a step INSIDE each fold", color=TEAL)
box(b, 0.3, 4.3, 5.6, 1.0, "training folds", LIGHT_TEAL)
box(b, 6.3, 4.3, 3.4, 1.0, "validation fold", LIGHT_GOLD)
arrow(b, 3.1, 4.3, 3.1, 3.6)
box(b, 0.3, 2.6, 5.6, 1.0, "select features + fit model", LIGHT_TEAL)
arrow(b, 5.9, 3.1, 6.9, 2.25)
arrow(b, 8.0, 4.3, 8.0, 2.25)
box(b, 6.3, 1.2, 3.4, 1.0, "honest score", "none", TEAL)
b.text(3.1, 1.5, "Pipeline([select, model])\nrefits this per fold", ha="center", fontsize=10, color=MUTED)
save(fig, "features-leakage")

# 4. datetime: hour as a circle
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
h = np.arange(24)
axes[0].plot(h, h, marker="o", ms=4, color=WINE)
axes[0].annotate("23:00 and 00:00 look\n23 units apart", (23, 23), (8, 19), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=MUTED))
axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("feature value"); axes[0].set_title("hour as a plain number")
s, c = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
axes[1].scatter(s, c, color=TEAL, s=30, zorder=3)
for hh, dx in [(0, 0.12), (6, 0), (12, 0), (18, 0), (23, -0.22)]:
    axes[1].text(s[hh]*1.2 + dx, c[hh]*1.2, f"{hh:02d}:00", ha="center", va="center", fontsize=9,
                 color=WINE if hh in (0, 23) else INK)
axes[1].set_aspect("equal"); axes[1].set_xlim(-1.45, 1.45); axes[1].set_ylim(-1.35, 1.35)
axes[1].set_xlabel("hour_sin"); axes[1].set_ylabel("hour_cos"); axes[1].set_title("hour as sin/cos: 23:00 sits next to 00:00")
fig.tight_layout(); save(fig, "features-cyclic")
