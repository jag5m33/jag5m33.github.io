import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats

x = np.array([3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 40])
q1, q2, q3 = np.percentile(x, [25, 50, 75])
iqr = q3 - q1
lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr

# ---------- 1. positions on the sorted data ----------
fig, ax = plt.subplots(figsize=(10, 3.2))
pos = np.arange(1, len(x) + 1)
ax.scatter(pos, np.zeros_like(pos), s=500, color=LIGHT_TEAL, edgecolor=TEAL, zorder=2)
for p, v in zip(pos, x):
    ax.text(p, 0, str(v), ha="center", va="center", fontsize=12, zorder=3)
    ax.text(p, -0.55, f"#{p}", ha="center", color=MUTED, fontsize=10)
for qpos, qv, lab, col in [(3.5, q1, "Q1", WINE), (6, q2, "Q2 (median)", BLUE), (8.5, q3, "Q3", GOLD)]:
    ax.annotate(f"{lab}\nposition {qpos:g}\nvalue {qv:g}", xy=(qpos, 0.22), xytext=(qpos, 1.05),
                ha="center", color=col, fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
ax.set_ylim(-0.9, 1.9); ax.set_xlim(0.3, 11.7)
ax.axis("off")
ax.set_title("Sorted data (n = 11) with the quartile positions numpy uses: 1 + (n − 1)p", fontsize=12)
save(fig, "quartiles-positions")

# ---------- 2. box plot anatomy ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
y = 0
inside = x[(x >= lo_f) & (x <= hi_f)]
wlo, whi = inside.min(), inside.max()
ax.add_patch(plt.Rectangle((q1, y - 0.25), iqr, 0.5, facecolor=LIGHT_TEAL, edgecolor=TEAL, lw=2))
ax.plot([q2, q2], [y - 0.25, y + 0.25], color=INK, lw=2.6)
ax.plot([wlo, q1], [y, y], color=TEAL, lw=2); ax.plot([q3, whi], [y, y], color=TEAL, lw=2)
ax.plot([wlo, wlo], [y - 0.12, y + 0.12], color=TEAL, lw=2); ax.plot([whi, whi], [y - 0.12, y + 0.12], color=TEAL, lw=2)
ax.scatter([40], [y], s=70, facecolor="none", edgecolor=WINE, lw=2, zorder=3)
for f in (lo_f, hi_f):
    ax.axvline(f, color=MUTED, ls=":", lw=1.5)
# labels
def lab(xv, yv, xt, yt, t, c=INK):
    ax.annotate(t, xy=(xv, yv), xytext=(xt, yt), ha="center", va="center", fontsize=10.5, color=c, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc=PAPER, ec="none"),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
lab(q1, y + 0.25, q1 - 5, 1.0, f"Q1 = {q1:g}\n(25th percentile)", WINE)
lab(q2, y + 0.25, q2, 1.45, f"median Q2 = {q2:g}", BLUE)
lab(q3, y + 0.25, q3 + 5, 1.0, f"Q3 = {q3:g}\n(75th percentile)", GOLD)
lab(wlo, y - 0.12, wlo - 3, -1.05, f"lower whisker = {wlo:g}\n(smallest point\ninside the fence)", TEAL)
lab(whi, y - 0.12, whi + 7, -1.05, f"upper whisker = {whi:g}\n(largest point\ninside the fence)", TEAL)
lab(40, y - 0.05, 37, -1.05, "outlier = 40\n(beyond the fence)", WINE)
bb = dict(boxstyle="round,pad=0.15", fc=PAPER, ec="none")
ax.text(lo_f, 2.05, f"lower fence\nQ1 − 1.5·IQR = {lo_f:g}", ha="center", color=MUTED, fontsize=10, bbox=bb)
ax.text(hi_f, 2.05, f"upper fence\nQ3 + 1.5·IQR = {hi_f:g}", ha="center", color=MUTED, fontsize=10, bbox=bb)
ax.annotate("", xy=(q1, -0.38), xytext=(q3, -0.38), arrowprops=dict(arrowstyle="<->", color=INK))
ax.text((q1 + q3) / 2, -0.62, f"IQR = {iqr:g}\n(box = middle 50%)", ha="center", va="top", fontsize=10, bbox=bb)
ax.set_ylim(-1.6, 2.5); ax.set_xlim(-5, 44)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("value")
ax.set_title("Anatomy of a box plot (the 11-value dataset)")
save(fig, "quartiles-boxplot")

# ---------- 3. 1.5 x IQR on a normal distribution ----------
k = stats.norm.ppf(0.75)        # Q3 of standard normal ~ 0.674
fence = k + 1.5 * 2 * k         # ~ 2.698
xs = np.linspace(-4, 4, 500)
fig, ax = plt.subplots(figsize=(8, 3.8))
pdf = stats.norm.pdf(xs)
ax.plot(xs, pdf, color=INK)
ax.fill_between(xs, pdf, where=(xs >= -k) & (xs <= k), color=LIGHT_TEAL, label="box: middle 50%")
ax.fill_between(xs, pdf, where=xs >= fence, color=WINE, alpha=0.8)
ax.fill_between(xs, pdf, where=xs <= -fence, color=WINE, alpha=0.8, label="beyond fences: about 0.7%")
for v, t in [(-k, "Q1\n−0.674σ"), (k, "Q3\n+0.674σ"), (-fence, "lower fence\n−2.698σ"), (fence, "upper fence\n+2.698σ")]:
    ax.axvline(v, color=MUTED, ls=":", lw=1.4)
    ax.text(v, 0.43, t, ha="center", fontsize=10, color=MUTED, bbox=dict(boxstyle="round,pad=0.12", fc=PAPER, ec="none"))
ax.annotate("about 0.35%\neach side", xy=(3.0, 0.004), xytext=(3.2, 0.12), ha="center", fontsize=10, color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.set_ylim(0, 0.52)
ax.set_xlabel("standard deviations from the mean")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=10)
ax.set_title("On normal data, Tukey's fences sit at about ±2.7σ")
save(fig, "quartiles-normal-fences")

# ---------- 4. z-score masking vs IQR ----------
mean, sd = x.mean(), x.std(ddof=1)
fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
ax = axes[0]
ax.scatter(x, np.zeros_like(x), s=70, color=INK, zorder=3)
ax.axvspan(mean - 3 * sd, mean + 3 * sd, color=LIGHT_GOLD, alpha=0.7)
ax.axvline(mean, color=GOLD, lw=2)
ax.text(mean, 0.55, f"mean {mean:.1f}, SD {sd:.1f}", ha="center", color=GOLD, fontsize=10.5, bbox=dict(boxstyle="round,pad=0.15", fc=LIGHT_GOLD, ec="none"))
ax.text(41, -0.55, "z = 2.82: NOT flagged", ha="right", color=WINE, fontsize=10.5)
ax.set_title("z-score rule: the 40 inflates the mean and SD, so mean ± 3SD swallows it", fontsize=11.5, loc="left")
ax = axes[1]
ax.scatter(x, np.zeros_like(x), s=70, color=INK, zorder=3)
ax.axvspan(lo_f, hi_f, color=LIGHT_TEAL, alpha=0.7)
ax.axvline(q2, color=BLUE, lw=2)
ax.scatter([40], [0], s=200, facecolor="none", edgecolor=WINE, lw=2)
ax.text(q2, 0.55, f"median {q2:g}, IQR {iqr:g}", ha="center", color=BLUE, fontsize=10.5, bbox=dict(boxstyle="round,pad=0.15", fc=LIGHT_TEAL, ec="none"))
ax.text(41, -0.55, "beyond fence 20: flagged", ha="right", color=WINE, fontsize=10.5)
ax.set_title("IQR rule: quartiles barely notice the 40, so the fences stay tight", fontsize=11.5, loc="left")
for a in axes:
    a.set_ylim(-0.9, 0.9); a.set_yticks([]); a.spines["left"].set_visible(False)
axes[1].set_xlabel("value (shaded band = 'normal' range under each rule)")
fig.tight_layout()
save(fig, "quartiles-zscore-vs-iqr")
