import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# 1. contours + gradient descent
def run_gd(H, start, lr, steps):
    w = np.array(start, float); path = [w.copy()]
    for _ in range(steps):
        w = w - lr * (H @ w); path.append(w.copy())
    return np.array(path)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
cases = [(np.diag([20.0, 0.8]), "unscaled: x1 in 300-2000, x2 in 0-5", 0.095, WINE),
         (np.diag([1.0, 1.0]), "scaled: both features about -1 to 1", 0.45, TEAL)]
g = np.linspace(-3, 3, 300); W1, W2 = np.meshgrid(g, g)
for ax, (H, title, lr, col) in zip(axes, cases):
    J = 0.5*(H[0, 0]*W1**2 + H[1, 1]*W2**2)
    ax.contour(W1, W2, J, levels=np.geomspace(0.05, J.max()*0.6, 9), colors=MUTED, linewidths=0.8)
    p = run_gd(H, [-0.5, 2.6], lr, 25 if col == WINE else 8)
    ax.plot(p[:, 0], p[:, 1], "-o", color=col, ms=4, lw=1.6)
    ax.plot(0, 0, marker="*", color=GOLD, ms=16, zorder=4)
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel("$w_1$ (weight on $x_1$)"); ax.set_ylabel("$w_2$ (weight on $x_2$)")
    ax.set_title(title, fontsize=12)
axes[0].text(0, -2.75, "steep across, shallow along:\nzigzags, needs many steps", ha="center", fontsize=10, color=WINE)
axes[1].text(0, -2.75, "round bowl: heads straight\nfor the minimum", ha="center", fontsize=10, color=TEAL)
fig.tight_layout(); save(fig, "scaling-contours")

# 2. one skewed feature through each scaler
rng = np.random.default_rng(0)
x = rng.lognormal(mean=3.5, sigma=0.6, size=400)
x = np.append(x, [400, 520])[:, None]
scalers = [("original", None), ("min-max", MinMaxScaler()), ("standard (z-score)", StandardScaler()),
           ("robust (median/IQR)", RobustScaler()), ("max-abs", MaxAbsScaler())]
fig, axes = plt.subplots(len(scalers), 1, figsize=(9, 8.5))
for ax, (name, s), col in zip(axes, scalers, [MUTED, BLUE, TEAL, GOLD, ORANGE]):
    v = x.ravel() if s is None else s.fit_transform(x).ravel()
    ax.hist(v, bins=60, color=col, alpha=0.85)
    lo, hi = v.min(), v.max()
    ax.text(1.01, 0.5, f"{name}\nrange {lo:.2f} to {hi:.2f}", transform=ax.transAxes, va="center", fontsize=10)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    med = np.median(v); ax.axvline(med, color=INK, lw=1, ls="--")
axes[0].set_title("Same right-skewed feature, five scalers (dashed line = median)")
axes[-1].set_xlabel("value after scaling (each panel has its own x-axis)")
fig.tight_layout(); save(fig, "scaling-skewed")

# 3. one-hot picture
fig, ax = plt.subplots(figsize=(9, 2.8)); ax.axis("off")
rows = ["red", "green", "blue", "green"]
cols = ["blue", "green", "red"]
ax.text(0.5, 4.6, "colour", ha="center", fontweight="bold")
for i, r in enumerate(rows):
    ax.text(0.5, 3.6 - i*0.9, r, ha="center")
ax.annotate("", (2.3, 2.3), (1.3, 2.3), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=MUTED))
ax.text(1.8, 2.6, "one-hot", ha="center", fontsize=10, color=MUTED)
for j, c in enumerate(cols):
    x0 = 3.2 + j*2.0
    ax.text(x0, 4.6, f"colour_{c}", ha="center", fontweight="bold", fontsize=10,
            color=MUTED if j == 0 else INK)
    for i, r in enumerate(rows):
        on = r == c
        ax.add_patch(plt.Rectangle((x0-0.6, 3.25 - i*0.9), 1.2, 0.7, fc=LIGHT_TEAL if on else "none",
                                   ec=RULE, lw=1))
        ax.text(x0, 3.6 - i*0.9, "1" if on else "0", ha="center", va="center", color=INK if on else MUTED)
ax.add_patch(plt.Rectangle((2.45, 0.45), 1.5, 4.6, fc="none", ec=WINE, lw=1.4, ls="--"))
ax.text(8.3, 2.0, "drop='first' removes\nthe dashed column:\nall zeros then means 'blue'", fontsize=10, color=WINE, va="center")
ax.set_xlim(0, 11); ax.set_ylim(0.3, 5)
save(fig, "scaling-onehot")
