import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np

# PCA and eigenvector figures: digital copies of the sketches in Jag's PCA notes.

def bare(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

# 1. curse of dimensionality: accuracy rises from 3 to 15 features, then falls
feats = [3, 6, 15, 50, 100, 500]; acc = [0.70, 0.78, 0.84, 0.80, 0.74, 0.64]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(feats))
ax.plot(x, acc, color=TEAL, lw=2.4, marker="o")
for i, (f, a) in enumerate(zip(feats, acc)):
    ax.text(i, a + 0.018, f"M{i+1}", ha="center", color=INK, fontsize=10)
ax.axvspan(-0.3, 2.3, color=LIGHT_TEAL, alpha=0.45); ax.axvspan(2.3, 5.3, color=LIGHT_WINE, alpha=0.45)
ax.text(1, 0.62, "accuracy ↑", color=TEAL, ha="center", fontsize=11, fontweight="bold")
ax.text(4, 0.88, "accuracy ↓ (overfitting)", color=WINE, ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([str(f) for f in feats]); ax.set_yticks([])
ax.set_xlabel("number of features"); ax.set_ylabel("accuracy"); ax.set_ylim(0.58, 0.92)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
save(fig, "pca-curse")

# 2. covariance signs: effect 1 (x up, y up) vs effect 2 (x up, y down)
rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
xs = rng.uniform(0, 10, 18)
for ax, sign, col, title in [(axs[0], 1, TEAL, "+ve covariance (effect 1)\nx ↑ y ↑, x ↓ y ↓"),
                             (axs[1], -1, WINE, "−ve covariance (effect 2)\nx ↑ y ↓, x ↓ y ↑")]:
    ax.scatter(xs, 5 + sign * 0.8 * (xs - 5) + rng.normal(0, 0.9, xs.size), color=col, s=26)
    bare(ax); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title, fontsize=11, loc="left")
fig.tight_layout(); save(fig, "pca-cov")

# 3. feature selection on the housing data: keep house size, drop fountain size
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
hs = rng.uniform(50, 150, 16)
axs[0].scatter(hs, 2 * hs + rng.normal(0, 12, hs.size), color=TEAL, s=26)
axs[0].set_title("house size: linear, corr close to +1\n→ keep", fontsize=11, loc="left")
fs = rng.uniform(0, 5, 16)
axs[1].scatter(fs, 200 + rng.normal(0, 40, fs.size), color=WINE, s=26)
axs[1].set_title("fountain size: corr ≈ 0 to 0.25\n→ not important, drop it", fontsize=11, loc="left")
for ax, lab in zip(axs, ["house size", "fountain size"]):
    bare(ax); ax.set_xlabel(lab); ax.set_ylabel("price")
fig.tight_layout(); save(fig, "pca-select")

# 4. projecting onto the x axis vs onto PC1 (size')
size = rng.uniform(1, 9, 14); rooms = 0.8 * size + rng.normal(0, 0.6, size.size)
P = np.c_[size, rooms]; m = P.mean(0)
w, V = np.linalg.eigh(np.cov(P.T)); pc1 = V[:, -1] * np.sign(V[0, -1])
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
ax = axs[0]
ax.scatter(P[:, 0], P[:, 1], color=INK, s=22, zorder=3)
for p in P: ax.plot([p[0], p[0]], [p[1], 0], color=MUTED, lw=0.7, ls=":")
ax.scatter(P[:, 0], np.zeros(len(P)), color=TEAL, s=22, zorder=3)
ax.set_title("project onto the x axis:\nkeep size, lose the rooms spread", fontsize=11, loc="left")
ax = axs[1]
ax.scatter(P[:, 0], P[:, 1], color=INK, s=22, zorder=3)
t = np.linspace(-6, 6, 2); L = m + np.outer(t, pc1)
ax.plot(L[:, 0], L[:, 1], color=GOLD, lw=2.2); ax.text(L[1, 0] - 0.3, L[1, 1] + 0.25, "PC1 (size')", color=GOLD, fontsize=11, fontweight="bold", ha="right")
pc2 = np.array([-pc1[1], pc1[0]]); L2 = m + np.outer(np.linspace(-1.6, 1.6, 2), pc2)
ax.plot(L2[:, 0], L2[:, 1], color=WINE, lw=1.6, ls="--"); ax.text(L2[0, 0] + 0.15, L2[0, 1] - 0.3, "PC2 (rooms')", color=WINE, fontsize=10)
for p in P:
    q = m + ((p - m) @ pc1) * pc1
    ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.7, ls=":"); ax.scatter(*q, color=TEAL, s=22, zorder=3)
ax.set_title("project onto PC1:\nmost of the spread kept", fontsize=11, loc="left")
for ax in axs:
    bare(ax); ax.set_xlabel("house size"); ax.set_ylabel("no. of rooms"); ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.6, 9.5)
fig.tight_layout(); save(fig, "pca-project")

# 5. one point P1 projected onto the unit vector u
fig, ax = plt.subplots(figsize=(5.2, 3.8))
p = np.array([2.2, 3.0]); u = np.array([np.cos(0.4), np.sin(0.4)])
s = p @ u; q = s * u
ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
ax.plot([0, 4.6 * u[0]], [0, 4.6 * u[1]], color=LIGHT_GOLD, lw=6, solid_capstyle="round")
ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=WINE, lw=2.6, mutation_scale=16))
ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, ls="--", lw=1.2)
ax.scatter(*p, color=INK, s=30, zorder=4); ax.scatter(*q, color=TEAL, s=40, zorder=4)
ax.text(p[0] + 0.1, p[1] + 0.1, "P₁ (x₁, y₁): data point", color=INK, fontsize=10)
ax.text(u[0] + 0.15, u[1] - 0.35, "u: unit vector", color=WINE, fontsize=10)
ax.text(q[0] + 0.15, q[1] - 0.35, f"P₁′ = P₁ · u = {s:.2f}\n(distance along u)", color=TEAL, fontsize=10)
ax.set_xlim(-0.4, 5); ax.set_ylim(-0.4, 3.8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
save(fig, "pca-dot")

# 6. eigenvector: A keeps v on its own line and only scales it by lambda
A = np.array([[1, 0.971], [0.971, 1]])
v = np.array([1, 1]) / np.sqrt(2); w_ = np.array([1, 0.2]); w_ = w_ / np.linalg.norm(w_)
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, vec, title, col in [(axs[0], v, "eigenvector: Av stays on v's line,\njust scaled by λ = 1.971", TEAL),
                            (axs[1], w_, "not an eigenvector: Aw is\nknocked off w's line", WINE)]:
    Av = A @ vec
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    ax.plot([-0.4 * vec[0] * 4, vec[0] * 4], [-0.4 * vec[1] * 4, vec[1] * 4], color=LIGHT_GOLD, lw=5)
    ax.annotate("", xy=Av, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, ls="--", mutation_scale=14))
    ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6, mutation_scale=16))
    ax.text(*(vec * 0.5 + np.array([0.12, -0.25])), "v" if vec is v else "w", color=INK, fontsize=13, fontweight="bold")
    ax.text(*(Av + np.array([0.08, 0.05])), "Av" if vec is v else "Aw", color=col, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.6, 2.4); ax.set_ylim(-0.6, 2.4); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=11, loc="left")
fig.tight_layout(); save(fig, "eigen-line")
