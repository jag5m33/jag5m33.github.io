import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from matplotlib.patches import Circle

# Digital copies of the sketches in Jag's deep learning notes.

def network(ax, layers, labels=None, wlabels=None, out_label=None):
    xs = np.linspace(0, 1, len(layers)); pts = []
    for x, n in zip(xs, layers):
        ys = np.linspace(0.5 + 0.17 * (n - 1), 0.5 - 0.17 * (n - 1), n) if n > 1 else [0.5]
        pts.append([(x, y) for y in ys])
    for a, b in zip(pts, pts[1:]):
        for p in a:
            for q in b: ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=1, zorder=1)
    cols = [BLUE] + [GOLD] * (len(layers) - 2) + [TEAL]
    for layer, c in zip(pts, cols):
        for p in layer: ax.add_patch(Circle(p, 0.045, fc=PAPER, ec=c, lw=2.2, zorder=2))
    if labels:
        for p, t in zip(pts[0], labels): ax.text(p[0] - 0.07, p[1], t, ha="right", va="center", fontsize=11)
    if out_label: ax.text(pts[-1][0][0] + 0.07, pts[-1][0][1], out_label, va="center", fontsize=11)
    ax.set_xlim(-0.35, 1.3); ax.set_ylim(0.05, 0.95); ax.set_aspect("equal"); ax.axis("off")
    return pts

# 1. the student network: study, play, sleep -> hidden layer -> pass / fail
fig, ax = plt.subplots(figsize=(7, 3.6))
pts = network(ax, [3, 2, 1], ["study  x₁", "play  x₂", "sleep  x₃"], out_label="ŷ  (pass / fail)")
for x, t in [(0, "input layer"), (0.5, "hidden layer 1"), (1, "output layer")]:
    ax.text(x, 0.07, t, ha="center", fontsize=10, color=MUTED)
ax.text(0.25, 0.86, "weights w", ha="center", fontsize=10, color=WINE)
ax.text(0.5, 0.64 + 0.1, "Σxᵢwᵢ + b  →  act( )", ha="center", fontsize=10, color=INK)
save(fig, "dl-network")

# 2. multilayered (deep) network
fig, ax = plt.subplots(figsize=(6, 3.4))
network(ax, [3, 5, 4, 3])
ax.text(0.5, 0.0, "many hidden layers", ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)
save(fig, "dl-deep")

# 3. gradient descent on the loss curve
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = np.linspace(-3, 3, 200); ax.plot(w, w ** 2, color=INK)
for x0, c, side in [(-2.2, TEAL, "left"), (2.2, WINE, "right")]:
    ax.scatter([x0], [x0 ** 2], color=c, zorder=3, s=40)
    ax.annotate("", xy=(x0 * 0.55, (x0 * 0.55) ** 2), xytext=(x0, x0 ** 2),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
ax.text(-2.2, 6.0, "slope −ve\n→ weight ↑", color=TEAL, ha="center", fontsize=10.5)
ax.text(2.2, 6.0, "slope +ve\n→ weight ↓", color=WINE, ha="center", fontsize=10.5)
ax.scatter([0], [0], color=GOLD, zorder=3, s=50); ax.text(0, 0.6, "global minimum", ha="center", fontsize=10.5, color=GOLD)
ax.set_xlabel("weight"); ax.set_ylabel("loss"); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-0.5, 9.5)
save(fig, "dl-gradient-descent")

# 4. activation functions and their derivatives
x = np.linspace(-10, 10, 400)
sig = 1 / (1 + np.exp(-x)); th = np.tanh(x)
fns = [("Sigmoid", sig, sig * (1 - sig), "output 0 → 1", "derivative 0 → 0.25"),
       ("Tanh", th, 1 - th ** 2, "output −1 → 1", "derivative 0 → 1"),
       ("ReLU", np.maximum(0, x), (x > 0).astype(float), "max(0, x)", "derivative 0 or 1"),
       ("Leaky ReLU", np.where(x > 0, x, 0.01 * x), np.where(x > 0, 1, 0.01), "max(0.01x, x)", "derivative 0.01 or 1")]
fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
for i, (name, f, d, t1, t2) in enumerate(fns):
    xx = x if i < 2 else np.linspace(-6, 6, 400)
    if i >= 2:
        f = np.maximum(0, xx) if i == 2 else np.where(xx > 0, xx, 0.01 * xx)
        d = (xx > 0).astype(float) if i == 2 else np.where(xx > 0, 1, 0.01)
    for ax, y, col, t in [(axes[0, i], f, TEAL, t1), (axes[1, i], d, WINE, t2)]:
        ax.axhline(0, color=RULE, lw=1); ax.axvline(0, color=RULE, lw=1)
        ax.plot(xx, y, color=col); ax.set_title(t, fontsize=11, fontweight="normal")
    axes[0, i].text(0.02, 1.2, name, transform=axes[0, i].transAxes, fontsize=13, fontweight="bold")
axes[0, 0].set_ylabel("function"); axes[1, 0].set_ylabel("derivative")
fig.tight_layout(); save(fig, "dl-activations")

# 5. regression losses: MSE (quadratic), MAE (V shape), Huber (quadratic then linear)
e = np.linspace(-3, 3, 300); d = 1.0
huber = np.where(abs(e) <= d, 0.5 * e ** 2, d * abs(e) - 0.5 * d ** 2)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, y, col, t in [(axes[0], 0.5 * e ** 2, TEAL, "MSE: quadratic"), (axes[1], abs(e), WINE, "MAE: V shape"),
                      (axes[2], huber, GOLD, "Huber: quadratic, then linear")]:
    ax.plot(e, y, color=col); ax.set_title(t, fontsize=12); ax.set_xlabel("error  y − ŷ"); ax.set_yticks([])
axes[2].axvline(d, color=MUTED, ls=":", lw=1); axes[2].axvline(-d, color=MUTED, ls=":", lw=1)
axes[2].text(d + 0.1, 3.2, "δ", color=MUTED, fontsize=12)
axes[0].set_ylabel("loss")
fig.tight_layout(); save(fig, "dl-losses")
