import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from sklearn.cluster import KMeans
rng = np.random.default_rng(0)
centres = np.array([[2, 2], [7, 3], [4.5, 7.5]])
P = np.vstack([c + rng.normal(0, 0.9, (40, 2)) for c in centres])
lab = KMeans(3, n_init=10, random_state=0).fit_predict(P)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
axes[0].scatter(P[:, 0], P[:, 1], s=28, color=MUTED, alpha=0.8)
axes[0].set_title("What the algorithm is given: no labels")
cols = [TEAL, WINE, GOLD]
for k in range(3):
    axes[1].scatter(P[lab == k, 0], P[lab == k, 1], s=28, color=cols[k], alpha=0.85, label=f"cluster {k+1}")
axes[1].set_title("What it finds: groups (K-means, k = 3)")
axes[1].legend(loc="upper right", fontsize=10)
for ax in axes: ax.set_xlabel("feature 1")
axes[0].set_ylabel("feature 2")
fig.text(0.5, -0.04, "The colours are not true labels: they are the algorithm's own grouping, which we must interpret.",
         ha="center", fontsize=10.5, color=MUTED)
save(fig, "unsupintro-clusters")

# supervised vs unsupervised side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
y = (P[:, 0] + 0.3 * P[:, 1] > 5.5).astype(int)
ax = axes[0]
ax.scatter(P[y == 1, 0], P[y == 1, 1], s=26, color=TEAL, marker="o", label="label = 1 (given)")
ax.scatter(P[y == 0, 0], P[y == 0, 1], s=26, color=WINE, marker="s", label="label = 0 (given)")
xs = np.linspace(0, 10, 10); ax.plot(xs, (5.5 - xs) / 0.3, color=INK, ls="--", lw=1.4)
ax.set_ylim(-1, 11); ax.set_xlim(-1, 10)
ax.set_title("Supervised: learn x → y from labels"); ax.legend(loc="upper right", fontsize=9.5)
ax = axes[1]
ax.scatter(P[:, 0], P[:, 1], s=26, color=MUTED, alpha=0.8)
far = np.argsort(-np.min(np.linalg.norm(P[:, None] - centres[None], axis=2), axis=1))[:4]
ax.scatter(P[far, 0], P[far, 1], s=110, facecolors="none", edgecolors=ORANGE, lw=2, label="unusual points")
ax.set_ylim(-1, 11); ax.set_xlim(-1, 10)
ax.set_title("Unsupervised: find structure in x alone"); ax.legend(loc="upper right", fontsize=9.5)
for ax in axes: ax.set_xlabel("feature 1")
axes[0].set_ylabel("feature 2")
save(fig, "unsupintro-supvsunsup")

# PCA picture
X = rng.multivariate_normal([0, 0], [[3, 2.4], [2.4, 2.4]], 150)
from sklearn.decomposition import PCA
pca = PCA(2).fit(X)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.scatter(X[:, 0], X[:, 1], s=18, color=MUTED, alpha=0.7)
for comp, var, col, name in zip(pca.components_, pca.explained_variance_, [TEAL, WINE], ["PC1", "PC2"]):
    v = comp * 2 * np.sqrt(var)
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    ax.text(v[0] * 1.12, v[1] * 1.12, name, color=col, fontsize=12, fontweight="bold")
ax.set_aspect("equal"); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_title(f"PCA: PC1 keeps {pca.explained_variance_ratio_[0]:.0%} of the spread")
save(fig, "unsupintro-pca")
print(pca.explained_variance_ratio_)
