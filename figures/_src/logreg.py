import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))

# 1. why not linear regression
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
x0 = np.array([1, 2, 3, 4, 6, 7, 8, 9.0]); y0 = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
for ax, extra, title in [(axes[0], False, "8 tumours: threshold at 0.5 works"),
                         (axes[1], True, "add one huge tumour: threshold moves")]:
    x = np.append(x0, 40) if extra else x0; y = np.append(y0, 1) if extra else y0
    m, c = np.polyfit(x, y, 1); xs = np.linspace(0, 42, 100)
    ax.scatter(x[y == 0], y[y == 0], color=TEAL, s=45, zorder=3)
    ax.scatter(x[y == 1], y[y == 1], color=WINE, marker="x", s=55, zorder=3)
    ax.plot(xs, m * xs + c, color=GOLD)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    xt = (0.5 - c) / m
    ax.axvline(xt, color=BLUE, ls="--", lw=1.4)
    ax.text(xt + 0.4, -0.25, f"cut-off x = {xt:.1f}", color=BLUE, fontsize=10)
    if extra:
        ax.annotate("now x = 6 is\ncalled benign", xy=(6, 1), xytext=(12, 0.2), fontsize=10, color=WINE,
                    arrowprops=dict(arrowstyle="->", color=WINE))
    if not extra:
        ax.annotate("line goes above 1\n(not a probability)", xy=(11, m*11+c), xytext=(16, 1.05), fontsize=9.5, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_title(title); ax.set_xlabel("tumour size x"); ax.set_ylim(-0.4, 1.7); ax.set_xlim(0, 42)
axes[0].set_ylabel("y (0 = benign, 1 = malignant)")
fig.tight_layout(); save(fig, "logreg-whynotlinear")

# 2. sigmoid
fig, ax = plt.subplots(figsize=(8, 3.8))
z = np.linspace(-8, 8, 300); ax.plot(z, sig(z), color=TEAL)
ax.axhline(0.5, color=MUTED, ls=":", lw=1.2); ax.axvline(0, color=MUTED, ls=":", lw=1.2)
ax.scatter([0], [0.5], color=GOLD, s=70, zorder=4)
ax.annotate("z = 0  →  g = 0.5", xy=(0, 0.5), xytext=(1.2, 0.3), color=GOLD, arrowprops=dict(arrowstyle="->", color=GOLD))
ax.text(4, 0.88, "large +z → close to 1", color=WINE, fontsize=10.5)
ax.text(-7.8, 0.1, "large −z → close to 0", color=BLUE, fontsize=10.5)
ax.text(-7.8, 0.7, "predict y = 1 when g ≥ 0.5,\ni.e. when z ≥ 0", color=INK, fontsize=10.5)
ax.set_xlabel("z = w·x + b"); ax.set_ylabel("g(z) = P(y = 1 | x)")
ax.set_title("Sigmoid: squashes any number into (0, 1)")
save(fig, "logreg-sigmoid")

# 3. boundaries
rng = np.random.default_rng(1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
pts = rng.uniform(0, 3.2, (40, 2)); lab = pts.sum(1) >= 3
ax.scatter(*pts[~lab].T, color=TEAL, s=35, label="y = 0  (z < 0)")
ax.scatter(*pts[lab].T, color=WINE, marker="x", s=40, label="y = 1  (z ≥ 0)")
xs = np.linspace(0, 3.2, 10); ax.plot(xs, 3 - xs, color=GOLD)
ax.text(2.35, 0.35, "x₁ + x₂ = 3\n(z = 0)", color=GOLD, fontsize=10.5, ha="left", bbox=dict(fc=PAPER, ec=GOLD, alpha=0.95))
ax.set_xlim(0, 3.2); ax.set_ylim(0, 3.2); ax.set_aspect("equal")
ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_title("w = [1, 1], b = −3: a straight line")
ax.legend(loc="upper center", fontsize=9.5, bbox_to_anchor=(0.5, -0.18), ncol=2)
ax = axes[1]
pts = rng.uniform(-2, 2, (80, 2)); lab = (pts ** 2).sum(1) >= 1
ax.scatter(*pts[~lab].T, color=TEAL, s=35); ax.scatter(*pts[lab].T, color=WINE, marker="x", s=40)
t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=GOLD)
ax.text(0, -1.75, "boundary: x₁² + x₂² = 1", color=GOLD, ha="center", fontsize=10.5,
        bbox=dict(fc=PAPER, ec=GOLD, alpha=0.95))
ax.set_aspect("equal"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_title("features x₁², x₂², w = [1, 1], b = −1: a circle")
fig.tight_layout(); save(fig, "logreg-boundaries")

# 4. non-convex vs convex (computed on a tiny 1-D dataset)
x = np.array([-4, -3, -1, 1, 3, 4.0]); y = np.array([0, 0, 1, 0, 1, 1.0])
ws = np.linspace(-6, 6, 400)
mse = [np.mean((sig(w * x) - y) ** 2) / 2 for w in ws]
ll = [np.mean(-(y * np.log(sig(w * x)) + (1 - y) * np.log(1 - sig(w * x)))) for w in ws]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(ws, mse, color=WINE); axes[0].set_title("Squared error + sigmoid: NOT convex")
axes[0].annotate("flat plateaus: gradient ≈ 0,\ngradient descent stalls", xy=(-5, mse[40]), xytext=(-4.5, 0.2),
                 fontsize=10, color=WINE, arrowprops=dict(arrowstyle="->", color=WINE))
axes[1].plot(ws, ll, color=TEAL); axes[1].set_title("Log loss: convex bowl, one minimum")
i = int(np.argmin(ll)); axes[1].scatter([ws[i]], [ll[i]], color=GOLD, s=60, zorder=3)
axes[1].annotate("single global minimum", xy=(ws[i], ll[i]), xytext=(ws[i] - 4.5, ll[i] + 1.3), color=GOLD,
                 arrowprops=dict(arrowstyle="->", color=GOLD))
for ax in axes: ax.set_xlabel("weight w"); ax.set_ylabel("cost J(w)")
fig.tight_layout(); save(fig, "logreg-convex")

# 5. log loss curves
f = np.linspace(0.005, 0.995, 300)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
ax = axes[0]; ax.plot(f, -np.log(f), color=WINE)
ax.set_title("true y = 1: cost = −log(f)")
ax.annotate("confident and wrong\n(f → 0) → huge cost", xy=(0.02, -np.log(0.02)), xytext=(0.25, 4.2), color=WINE,
            fontsize=10.5, arrowprops=dict(arrowstyle="->", color=WINE))
ax.annotate("confident and right\n(f → 1) → cost ≈ 0", xy=(0.97, 0.03), xytext=(0.5, 1.5), color=TEAL,
            fontsize=10.5, arrowprops=dict(arrowstyle="->", color=TEAL))
ax = axes[1]; ax.plot(f, -np.log(1 - f), color=BLUE)
ax.set_title("true y = 0: cost = −log(1 − f)")
ax.annotate("confident and wrong\n(f → 1) → huge cost", xy=(0.98, -np.log(0.02)), xytext=(0.3, 4.2), color=BLUE,
            fontsize=10.5, arrowprops=dict(arrowstyle="->", color=BLUE))
ax.annotate("confident and right\n(f → 0) → cost ≈ 0", xy=(0.03, 0.03), xytext=(0.15, 1.5), color=TEAL,
            fontsize=10.5, arrowprops=dict(arrowstyle="->", color=TEAL))
for ax in axes: ax.set_xlabel("predicted probability f = P(y = 1)"); ax.set_ylim(0, 5.3)
axes[0].set_ylabel("loss for one example")
fig.tight_layout(); save(fig, "logreg-logloss")
