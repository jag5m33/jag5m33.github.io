import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from sklearn.ensemble import IsolationForest

# the worked example: 10 half-hourly demand residuals (MW), one spike
x = np.array([-310, -180, -90, -40, 0, 30, 70, 120, 210, 2400.])

# 1. random cuts on a number line: spike vs a middle value (one hand-picked tree)
fig, axes = plt.subplots(2, 1, figsize=(10, 4.6))
def line(ax, title):
    ax.axhline(0, color=RULE, lw=2, zorder=1)
    ax.scatter(x[:-1], np.zeros(9), color=TEAL, s=55, zorder=3)
    ax.scatter([2400], [0], color=WINE, s=80, zorder=3)
    ax.set_xlim(-400, 2550); ax.set_ylim(-1, 1.25); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.set_title(title, loc="left", fontsize=12)
ax = axes[0]; line(ax, "The spike (2,400 MW): alone after 1 random cut")
ax.axvline(1000, color=GOLD, lw=2, ls="--"); ax.text(1020, 0.55, "cut 1 at 1,000", color=GOLD, fontsize=10.5)
ax.annotate("isolated", xy=(2400, 0.08), xytext=(2050, 0.6), color=WINE, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax = axes[1]; line(ax, "A normal value (0 MW): needs 6 random cuts")
cuts = [1000, 150, -120, 50, -20, 15]
for k, c in enumerate(cuts, 1):
    ax.axvline(c, color=GOLD, lw=1.6, ls="--")
    hy = {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.95, 5: 0.95, 6: 0.4}[k]
    ax.text(c + 10, hy, str(k), color=GOLD, fontsize=10.5, fontweight="bold")
ax.annotate("isolated after cut 6", xy=(0, -0.08), xytext=(330, -0.75), color=TEAL, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=TEAL))
ax.set_xlabel("demand residual (MW): what's left after removing the normal daily / weekly / yearly pattern")
fig.tight_layout(); save(fig, "isoforest-cuts")

# 2. average path length over 100 random trees (manual implementation)
def depth(vals, target, rng, d=0):
    if len(vals) <= 1 or vals.min() == vals.max(): return d
    c = rng.uniform(vals.min(), vals.max())
    return depth(vals[vals < c] if target < c else vals[vals >= c], target, rng, d + 1)
rng = np.random.default_rng(0); T = 100; paths = np.zeros((T, len(x)))
for t in range(T):
    seed = rng.integers(1e9)
    for i, v in enumerate(x): paths[t, i] = depth(x, v, np.random.default_rng(seed))
avg = paths.mean(0)
fig, ax = plt.subplots(figsize=(9, 3.6))
cols = [WINE if v == 2400 else TEAL for v in x]
bars = ax.bar(range(10), avg, color=cols)
for i, a in enumerate(avg): ax.text(i, a + 0.08, f"{a:.1f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(range(10)); ax.set_xticklabels([f"{v:,.0f}" for v in x])
ax.set_xlabel("demand residual (MW)"); ax.set_ylabel("average number of cuts")
ax.set_title("Average path length over 100 random trees: short = odd")
ax.set_ylim(0, 6.6)
save(fig, "isoforest-paths")

# 3. 2D scatter with score shading
rng = np.random.default_rng(0)
normal = rng.normal(0, 1, (200, 2)) * [1.0, 0.7]
odd = np.array([[4.2, 3.0], [-3.8, 2.6], [3.5, -3.2], [-4.0, -2.4], [0.3, 4.1], [4.6, 0.2]])
X = np.vstack([normal, odd])
m = IsolationForest(n_estimators=100, contamination=0.03, random_state=0).fit(X)
xx, yy = np.meshgrid(np.linspace(-5.5, 5.5, 110), np.linspace(-5, 5, 100))
zz = m.score_samples(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(8, 5.2))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("iso", [LIGHT_WINE, PAPER, LIGHT_TEAL])
cs = ax.contourf(xx, yy, zz, levels=8, cmap=cmap)
ax.contour(xx, yy, m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
           levels=[0], colors=GOLD, linewidths=2)
pred = m.predict(X)
ax.scatter(*X[pred == 1].T, color=TEAL, s=16, label="normal (predict = 1)")
ax.scatter(*X[pred == -1].T, color=WINE, s=55, marker="x", label="anomaly (predict = −1)")
cb = fig.colorbar(cs, ax=ax, shrink=0.85); cb.set_label("score_samples (lower = more anomalous)")
ax.text(-5.3, 4.5, "gold line: the contamination = 3% threshold", color=GOLD, fontsize=10.5)
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2"); ax.set_aspect("equal")
ax.set_title("Isolation forest on 206 points: the edges score lowest")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=10)
save(fig, "isoforest-2d")
print("flagged", (pred == -1).sum(), "of which true odd", (pred[-6:] == -1).sum())
