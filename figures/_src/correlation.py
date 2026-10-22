import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats

# 1. grid of r values
rng = np.random.default_rng(0)
targets = [-0.9, -0.5, 0, 0.5, 0.9]
fig, axes = plt.subplots(1, 5, figsize=(12, 2.9))
for ax, r in zip(axes, targets):
    xy = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], 120)
    rr = stats.pearsonr(xy[:, 0], xy[:, 1])[0]
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color=TEAL if r >= 0 else WINE, alpha=0.75, edgecolor="none")
    ax.set_title(f"target r = {r:+.1f}\n(sample r = {rr:+.2f})", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
fig.suptitle("Sign = direction, size = how tightly the points hug a line", y=1.1, fontsize=13, weight="bold")
save(fig, "correlation-grid")

# 2. curved relationship: r ~ 0
x = np.linspace(-3, 3, 41)
y = x**2 + rng.normal(0, 0.4, x.size)
r = stats.pearsonr(x, y)[0]; rho = stats.spearmanr(x, y)[0]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.scatter(x, y, s=22, color=BLUE)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"A perfect U-shape: Pearson r = {r:.2f}, Spearman ρ = {rho:.2f}")
ax.text(0, 7.5, "strong relationship, but not monotonic\n→ both coefficients say ~0", ha="center", fontsize=10.5, color=MUTED)
ax.set_ylim(-1, 10)
save(fig, "correlation-curve")
print("curve", r, rho)

# 3. outlier drags Pearson, not Spearman
x = np.arange(1, 13, dtype=float)
y = x + rng.normal(0, 0.8, x.size)
r0, s0 = stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0]
xo, yo = np.append(x, 13), np.append(y, 60)
r1, s1 = stats.pearsonr(xo, yo)[0], stats.spearmanr(xo, yo)[0]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].scatter(x, y, color=TEAL, s=26); axes[0].set_title(f"clean: Pearson {r0:.2f}, Spearman {s0:.2f}", fontsize=11.5)
axes[1].scatter(x, y, color=TEAL, s=26); axes[1].scatter([13], [60], color=WINE, s=60, zorder=3)
axes[1].annotate("one typo: 60 instead of ~13\n(to ranks it is just 'the largest')", xy=(13, 60), xytext=(1, 45), color=WINE, arrowprops=dict(arrowstyle="->", color=WINE))
axes[1].set_title(f"+1 outlier: Pearson {r1:.2f}, Spearman {s1:.2f}", fontsize=11.5)
for ax in axes: ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.suptitle("One outlier drags Pearson; Spearman (ranks) barely moves", y=1.04, fontsize=13, weight="bold")
save(fig, "correlation-outlier")
print("outlier", r0, s0, r1, s1)

# 4. worked example with deviations
x = np.array([1, 2, 3, 4, 5, 6.]); y = np.array([52, 55, 61, 60, 68, 72.])
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.axvline(x.mean(), color=MUTED, ls=":", lw=1.2); ax.axhline(y.mean(), color=MUTED, ls=":", lw=1.2)
for xi, yi in zip(x, y):
    c = TEAL if (xi - x.mean()) * (yi - y.mean()) > 0 else WINE
    ax.plot([xi, xi], [y.mean(), yi], color=c, lw=1.2)
    ax.scatter(xi, yi, color=c, s=50, zorder=3)
ax.text(x.mean() + 0.05, 70.5, "x̄ = 3.5", color=MUTED, fontsize=10)
ax.text(0.9, y.mean() + 0.6, "ȳ = 61.33", color=MUTED, fontsize=10)
ax.text(5.0, 54, "teal: both above or both below\nthe means → positive product\nred: opposite sides → negative", fontsize=9.5, color=INK)
ax.set_xlabel("hours studied (x)"); ax.set_ylabel("exam score (y)")
ax.set_title("Covariance adds up how points sit relative to both means")
save(fig, "correlation-worked")

# 5. Simpson's paradox
rng = np.random.default_rng(3)
fig, ax = plt.subplots(figsize=(7.5, 4))
X, Y = [], []
for k, c in enumerate([TEAL, GOLD, WINE]):
    xs = rng.uniform(0, 3, 25) + 3 * k
    ys = -0.8 * (xs - 3 * k) + 4 * k + rng.normal(0, 0.5, 25)
    ax.scatter(xs, ys, s=18, color=c, label=f"group {k+1}: r = {stats.pearsonr(xs, ys)[0]:.2f}")
    b = np.polyfit(xs, ys, 1); g = np.linspace(xs.min(), xs.max(), 2); ax.plot(g, np.polyval(b, g), color=c, lw=1.6)
    X += list(xs); Y += list(ys)
X, Y = np.array(X), np.array(Y)
b = np.polyfit(X, Y, 1); g = np.linspace(0, 9, 2)
ax.plot(g, np.polyval(b, g), color=INK, ls="--", lw=1.6, label=f"all pooled: r = {stats.pearsonr(X, Y)[0]:.2f}")
ax.legend(fontsize=9.5, loc="upper left")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Simpson's paradox: down within every group, up overall")
save(fig, "correlation-simpson")
