import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats

# ---------- 1. worked dataset: dots, mean line, deviations ----------
x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
mean = x.mean()
fig, ax = plt.subplots(figsize=(8, 3.6))
# stack repeated values so every point is visible
counts = {}
ys = []
for v in x:
    counts[v] = counts.get(v, 0) + 1
    ys.append(counts[v])
ys = np.array(ys)
for v, y in zip(x, ys):
    d = v - mean
    col = TEAL if d > 0 else (WINE if d < 0 else MUTED)
    ax.annotate("", xy=(v, y), xytext=(mean, y),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.6, alpha=0.8))
    ax.text(v + (0.12 if d == 0 else 0), y + 0.22, f"{d:+.0f}" if d else "0", ha="left" if d == 0 else "center",
            fontsize=11, color=col, fontweight="bold")
ax.scatter(x, ys, s=110, color=INK, zorder=3)
ax.axvline(mean, color=GOLD, lw=2.4, zorder=1)
ax.text(mean, 3.85, "mean = 5", color=GOLD, ha="center", fontweight="bold")
ax.axvline(np.median(x), color=BLUE, lw=1.6, ls="--")
ax.text(np.median(x) - 0.08, 3.45, "median = 4.5", color=BLUE, ha="right", fontsize=11)
ax.set_ylim(0.4, 4.2); ax.set_xlim(0.8, 10)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("value")
ax.set_title("Each point's deviation from the mean (they always sum to 0)")
save(fig, "descriptive-deviations")

# ---------- 2. salaries: one big value drags the mean ----------
sal = np.array([24, 26, 27, 28, 30, 31, 32, 150])
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.scatter(sal, np.zeros_like(sal), s=120, color=INK, zorder=3)
ax.axvline(sal.mean(), color=GOLD, lw=2.4)
ax.axvline(np.median(sal), color=BLUE, lw=2.4, ls="--")
ax.text(sal.mean() + 2, 0.6, f"mean = £{sal.mean():.1f}k\n(above 7 of 8 people)", color=GOLD, fontsize=11, va="center")
ax.text(np.median(sal) + 2, -0.6, f"median = £{np.median(sal):.0f}k\n(a typical person)", color=BLUE, fontsize=11, va="center")
ax.annotate("the director", xy=(150, 0), xytext=(128, 0.55), fontsize=11, color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.set_ylim(-1, 1); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("salary (£ thousands)")
ax.set_title("Eight salaries: one outlier pulls the mean, not the median")
save(fig, "descriptive-salaries")

# ---------- 3. skew: three distributions with mean / median / mode ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
specs = [
    ("Left (negative) skew", stats.beta(5, 2), (0, 1)),
    ("Symmetric", stats.norm(0, 1), (-3.5, 3.5)),
    ("Right (positive) skew", stats.gamma(a=2, scale=1), (0, 9)),
]
for ax, (title, dist, (lo, hi)) in zip(axes, specs):
    xs = np.linspace(lo, hi, 400)
    pdf = dist.pdf(xs)
    ax.fill_between(xs, pdf, color=LIGHT_TEAL, alpha=0.7)
    ax.plot(xs, pdf, color=TEAL)
    mode = xs[np.argmax(pdf)]
    med = dist.median(); mn = dist.mean()
    top = pdf.max()
    if title == "Symmetric":
        ax.axvline(mn, color=INK, lw=1.8)
        ax.text(mn, top * 1.07, "mean = median = mode", ha="center", fontsize=10)
    else:
        for val, col, lab, ls, h in [(mode, WINE, "mode", "-", 1.22),
                                     (med, BLUE, "median", "--", 1.12),
                                     (mn, GOLD, "mean", "-", 1.02)]:
            ax.axvline(val, color=col, lw=1.8, ls=ls)
        # labels stacked to avoid overlap
        order = [("mode", mode, WINE), ("median", med, BLUE), ("mean", mn, GOLD)]
        for k, (lab, val, col) in enumerate(order):
            ax.text(val, top * (1.36 - 0.12 * k), lab, color=col, ha="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc=PAPER, ec="none"))
    ax.set_ylim(0, top * 1.45)
    ax.set_title(title, fontsize=12)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([])
axes[0].text(0.05, 0.05, "tail ←", transform=axes[0].transAxes, color=MUTED)
axes[2].text(0.75, 0.05, "→ tail", transform=axes[2].transAxes, color=MUTED)
fig.suptitle("The mean is dragged furthest towards the tail; the median sits between", y=1.03, fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "descriptive-skew")

# ---------- 4. Bessel: dividing by n underestimates ----------
rng = np.random.default_rng(0)
X = rng.normal(0, 2, size=(20000, 5))   # true variance = 4, samples of n = 5
v0 = X.var(axis=1, ddof=0); v1 = X.var(axis=1, ddof=1)
fig, ax = plt.subplots(figsize=(8, 3.8))
bins = np.linspace(0, 16, 50)
ax.hist(v0, bins=bins, color=WINE, alpha=0.45, label="divide by n")
ax.hist(v1, bins=bins, color=TEAL, alpha=0.45, label="divide by n − 1")
ax.axvline(4, color=INK, lw=2); ax.text(4.2, ax.get_ylim()[1] * 0.92, "true σ² = 4", color=INK)
ax.axvline(v0.mean(), color=WINE, ls="--"); ax.axvline(v1.mean(), color=TEAL, ls="--")
bb = dict(boxstyle="round,pad=0.2", fc=PAPER, ec="none")
ax.text(v0.mean() - 0.2, ax.get_ylim()[1] * 0.72, f"average\n{v0.mean():.2f}", color=WINE, ha="right", fontsize=10, bbox=bb)
ax.text(v1.mean() + 0.3, ax.get_ylim()[1] * 0.55, f"average\n{v1.mean():.2f}", color=TEAL, fontsize=10, bbox=bb)
ax.set_xlabel("sample variance from one sample of n = 5")
ax.set_ylabel("how many samples")
ax.set_title("20,000 samples of size 5: only n − 1 is right on average")
ax.legend(loc="upper right")
save(fig, "descriptive-bessel")
print("bessel means", v0.mean(), v1.mean())
