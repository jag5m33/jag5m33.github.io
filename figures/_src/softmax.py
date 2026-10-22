import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
z = np.array([2.0, 1.0, 0.1]); e = np.exp(z); p = e / e.sum()
names = ["class A", "class B", "class C"]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, vals, t, col, fmt in [(axes[0], z, "1. scores z (any number)", BLUE, "{:.1f}"),
                              (axes[1], e, "2. e^z (all positive)", GOLD, "{:.3f}"),
                              (axes[2], p, "3. divide by sum 11.213", TEAL, "{:.3f}")]:
    b = ax.bar(names, vals, color=col, width=0.6)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + max(vals) * 0.03, fmt.format(v), ha="center", fontsize=10.5)
    ax.set_title(t, fontsize=12); ax.set_ylim(0, max(vals) * 1.2)
axes[2].text(1, 0.6, "sums to 1", color=TEAL, ha="center", fontsize=10.5)
fig.tight_layout(); save(fig, "softmax-steps")

# LEDGAR train vs validation
fig, ax = plt.subplots(figsize=(8, 3.4))
metrics = ["Accuracy", "Macro F1", "Micro F1"]
tr = [0.899, 0.887, 0.899]; va = [0.839, 0.780, 0.839]
xx = np.arange(3); w = 0.36
b1 = ax.bar(xx - w/2, tr, w, color=TEAL, label="train"); b2 = ax.bar(xx + w/2, va, w, color=WINE, label="validation")
for bars in (b1, b2):
    for r in bars: ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.01, f"{r.get_height():.3f}", ha="center", fontsize=10)
ax.set_xticks(xx); ax.set_xticklabels(metrics); ax.set_ylim(0.5, 1.0)
ax.text(1 + w/2, 0.55, "gap\n0.107", ha="center", fontsize=10.5, color="white", fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); ax.set_title("LEDGAR softmax logistic regression (100 clause types)")
save(fig, "softmax-ledgar")
