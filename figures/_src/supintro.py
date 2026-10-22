import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from matplotlib.patches import FancyBboxPatch
rng = np.random.default_rng(0)

# 1. regression vs classification
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
x = rng.uniform(40, 150, 30); y = 2.2 * x + 60 + rng.normal(0, 25, 30)
ax.scatter(x, y, color=TEAL, s=35, zorder=3)
xs = np.linspace(35, 155, 50); b, a = np.polyfit(x, y, 1)
ax.plot(xs, b * xs + a, color=GOLD)
print("pred100", b*100+a); ax.annotate(f"predict a number:\n~£{b*100+a:.0f}k for 100 m²", xy=(100, b*100+a), xytext=(42, 330),
            fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_xlabel("size (m²)  = feature x"); ax.set_ylabel("price (£k)  = target y")
ax.set_title("Regression: y is a number")
ax = axes[1]
n = 25
a0 = rng.normal([2.5, 2.5], 0.8, (n, 2)); a1 = rng.normal([5.5, 5.0], 0.8, (n, 2))
ax.scatter(*a0.T, color=TEAL, marker="o", s=40, label="y = 0 (benign)")
ax.scatter(*a1.T, color=WINE, marker="x", s=45, label="y = 1 (malignant)")
xs = np.linspace(0, 8, 10); ax.plot(xs, 8 - xs, color=GOLD, ls="--")
ax.text(6.6, 1.9, "decision\nboundary", color=GOLD, fontsize=10)
ax.set_xlim(0, 8); ax.set_ylim(0, 8)
ax.set_xlabel("feature x₁ (tumour size)"); ax.set_ylabel("feature x₂ (age)")
ax.set_title("Classification: y is a category")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); save(fig, "supintro-regcls")

# 2. train / val / test split + workflow
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 3.2)
parts = [(0, 6, LIGHT_TEAL, TEAL, "Train 60%", "fit the model\n(learn w, b)"),
         (6, 2, LIGHT_GOLD, GOLD, "Validation 20%", "tune settings\n(C, depth, threshold)"),
         (8, 2, LIGHT_WINE, WINE, "Test 20%", "score ONCE\nat the very end")]
for x0, w, fc, ec, t, s in parts:
    ax.add_patch(FancyBboxPatch((x0 + 0.05, 1.7), w - 0.1, 1.0, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.8))
    ax.text(x0 + w / 2, 2.2, t, ha="center", va="center", fontweight="bold", color=INK)
    ax.text(x0 + w / 2, 0.95, s, ha="center", va="center", fontsize=10.5, color=INK)
ax.annotate("", xy=(7, 1.45), xytext=(3, 1.45), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
ax.annotate("", xy=(9, 1.45), xytext=(7, 1.45), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
ax.text(5, 0.15, "the order: split  →  fit on train  →  compare/tune on validation  →  test once", ha="center", color=MUTED, fontsize=10.5)
save(fig, "supintro-split")
