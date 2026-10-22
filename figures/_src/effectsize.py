import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats

# 1. two worlds: same 8-point gap, different spreads
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, sd, ttl in [(axes[0], 25, "people vary by ~25 points"), (axes[1], 2, "people vary by ~2 points")]:
    x = np.linspace(136 - 4 * sd - 4, 136 + 4 * sd + 4, 800)
    a, b = stats.norm.pdf(x, 140, sd), stats.norm.pdf(x, 132, sd)
    ax.fill_between(x, a, color=LIGHT_GOLD, alpha=0.8); ax.plot(x, a, color=GOLD, label="placebo")
    ax.fill_between(x, b, color=LIGHT_TEAL, alpha=0.8); ax.plot(x, b, color=TEAL, label="drug")
    ax.set_yticks([]); ax.set_xlabel("blood pressure")
    ax.set_title(f"{ttl}: d = {8/sd:.2f}", fontsize=11.5)
    ax.set_ylim(0, a.max() * 1.25)
axes[1].legend(fontsize=10, loc="upper right")
fig.suptitle("Same 8-point drop, two worlds", y=1.04, fontsize=13, weight="bold")
save(fig, "effectsize-two-worlds")

# 2. d = 0.2, 0.5, 0.8
x = np.linspace(-4, 5, 600)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
for ax, d, band in zip(axes, [0.2, 0.5, 0.8], ["small", "medium", "large"]):
    a, b = stats.norm.pdf(x), stats.norm.pdf(x, d)
    ax.fill_between(x, np.minimum(a, b), color=RULE)
    ax.plot(x, a, color=GOLD, label="control"); ax.plot(x, b, color=TEAL, label="treated")
    ov = 2 * stats.norm.cdf(-d / 2)
    ax.set_title(f"d = {d} ({band}): {ov:.0%} overlap", fontsize=11.5)
    ax.set_yticks([]); ax.set_xlabel("outcome (in SDs)")
axes[2].legend(fontsize=9.5, loc="upper right", bbox_to_anchor=(1.25, 1.0))
fig.suptitle("Cohen's d is how many SDs apart the two bells sit (grey = shared area)", y=1.05, fontsize=13, weight="bold")
save(fig, "effectsize-d-bands")

# 3. p-value vs effect size: tiny effect, huge n
rng = np.random.default_rng(0)
ns = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
d = 0.05
p = [stats.ttest_ind(rng.normal(d, 1, n), rng.normal(0, 1, n)).pvalue for n in ns]
pe = [2 * stats.norm.sf(d / np.sqrt(2 / n)) for n in ns]  # expected p at true d
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(ns, pe, color=WINE, marker="o", ms=4)
ax.axhline(0.05, color=MUTED, ls="--", lw=1.2); ax.text(25, 0.06, "α = 0.05", color=MUTED, fontsize=10)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("n per group (log scale)"); ax.set_ylabel("typical p-value (log)")
ax.set_title("A trivial effect (d = 0.05) becomes 'significant' once n is big")
save(fig, "effectsize-p-vs-n")
for n, q in zip(ns, pe): print(n, round(q, 4))
