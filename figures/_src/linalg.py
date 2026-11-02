import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np

# Linear algebra figures: digital copies of the sketches in Jag's handwritten notes.

def axes2d(ax, lim):
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

def arrow(ax, start, vec, color, label=None, off=(0.1, 0.1), ls="-", lw=2.4):
    ax.annotate("", xy=(start[0] + vec[0], start[1] + vec[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=16))
    if label:
        ax.text(start[0] + vec[0] / 2 + off[0], start[1] + vec[1] / 2 + off[1], label, color=color,
                fontsize=14, fontweight="bold")

# 1. the vector [2, 1]: 2 across, 1 up, rooted at the origin
fig, ax = plt.subplots(figsize=(4.6, 3))
axes2d(ax, ((-0.6, 2.8), (-0.6, 1.6)))
arrow(ax, (0, 0), (2, 1), TEAL)
ax.plot([0, 2], [0, 0], color=GOLD, ls="--", lw=1.6); ax.plot([2, 2], [0, 1], color=GOLD, ls="--", lw=1.6)
ax.text(1, -0.28, "2 across", color=GOLD, ha="center", fontsize=11)
ax.text(2.08, 0.45, "1 up", color=GOLD, fontsize=11)
ax.text(0.7, 0.62, r"$\begin{bmatrix}2\\1\end{bmatrix}$" if False else "[2, 1]", color=TEAL, fontsize=12)
ax.text(-0.08, -0.25, "origin", color=MUTED, fontsize=10, ha="right")
save(fig, "la-vector")

# 2. adding v + w: both from the origin, then w moved to the tip of v
v, w = np.array([1, 2]), np.array([3, -1])
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axs[0]; axes2d(ax, ((-0.5, 4.6), (-1.5, 2.6)))
arrow(ax, (0, 0), v, TEAL, "v", (-0.4, 0)); arrow(ax, (0, 0), w, WINE, "w", (0, -0.45))
ax.set_title("v and w, both rooted at the origin", fontsize=11, loc="left")
ax = axs[1]; axes2d(ax, ((-0.5, 4.6), (-1.5, 2.6)))
arrow(ax, (0, 0), v, TEAL, "v", (-0.4, 0)); arrow(ax, tuple(v), w, WINE, "w", (0.1, 0.15))
arrow(ax, (0, 0), v + w, GOLD, "v + w", (0.1, -0.5), ls="--", lw=2)
ax.set_title("move the start of w to the tip of v", fontsize=11, loc="left")
fig.tight_layout(); save(fig, "la-add")

# 3. scaling: v x 2 stretches, v x (-1.8) flips and stretches
v = np.array([1.2, 0.8])
fig, ax = plt.subplots(figsize=(5.4, 3.8))
axes2d(ax, ((-2.6, 2.8), (-1.8, 2.0)))
arrow(ax, (0, 0), 2 * v, LIGHT_TEAL, lw=5)
ax.text(2 * v[0] - 0.4, 2 * v[1] + 0.12, "v × 2", color=TEAL, fontsize=12)
arrow(ax, (0, 0), v, TEAL, "v", (-0.3, 0.12))
arrow(ax, (0, 0), -1.8 * v, WINE)
ax.text(-1.8 * v[0] + 0.15, -1.8 * v[1] - 0.35, "v × (−1.8): flip and stretch", color=WINE, fontsize=11)
save(fig, "la-scale")

# 4. linear combination 3v + 2w: keep one scalar fixed, change the other -> tip moves along a line
v, w = np.array([0.6, 0.9]), np.array([0.5, -0.35])
fig, ax = plt.subplots(figsize=(5.4, 4))
axes2d(ax, ((-1.6, 3.6), (-1.6, 3.2)))
a = np.linspace(-2.2, 3.6, 2); line = np.outer(a, v) + 2 * w
ax.plot(line[:, 0], line[:, 1], color=GOLD, ls="--", lw=1.6)
arrow(ax, (0, 0), 3 * v, TEAL, "3v", (-0.5, 0.05))
arrow(ax, tuple(3 * v), 2 * w, WINE, "2w", (0.1, 0.1))
ax.text(2.2, 0.2, "2w held constant,\nscalar on v changes:\ntip slides along\nthis dashed line", color=GOLD, fontsize=10, ha="left")
save(fig, "la-combo")

# 5. span of two 3D vectors: a flat sheet through the origin
fig = plt.figure(figsize=(5.4, 4.4)); ax = fig.add_subplot(projection="3d")
v, w = np.array([1, 0.2, 0.5]), np.array([0.1, 1, 0.6])
s, t = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
P = s[..., None] * v + t[..., None] * w
ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=LIGHT_TEAL, alpha=0.55, edgecolor=TEAL, lw=0.6)
for vec, c, name in [(v, TEAL, "v"), (w, WINE, "w")]:
    ax.quiver(0, 0, 0, *vec, color=c, lw=2.4, arrow_length_ratio=0.15)
    ax.text(*(vec * 1.15), name, color=c, fontsize=14, fontweight="bold")
for k in range(3):
    e = np.zeros(3); e[k] = 1.8
    ax.plot(*np.c_[-e, e], color=MUTED, lw=0.8)
ax.set_axis_off(); ax.view_init(elev=18, azim=-60)
ax.set_title("span of v and w in 3D: a flat sheet (plane)", fontsize=11)
save(fig, "la-plane")
