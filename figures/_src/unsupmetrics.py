import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
rng = np.random.default_rng(0)

# ---------- shared blob data: 4 true groups ----------
centres = np.array([[2, 2], [8, 2.5], [5, 7.5], [9.5, 8]])
P = np.vstack([c + rng.normal(0, 0.8, (50, 2)) for c in centres])

# ---------- 1. elbow ----------
ks = range(1, 10); inert = []
for k in ks:
    inert.append(KMeans(k, n_init=10, random_state=0).fit(P).inertia_)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ks, inert, "o-", color=TEAL)
ax.scatter([4], [inert[3]], s=160, facecolors="none", edgecolors=WINE, lw=2, zorder=3)
ax.annotate("elbow at k = 4:\nadding clusters after this\nbarely lowers WCSS", xy=(4, inert[3]), xytext=(5.3, inert[1] * 0.75),
            color=WINE, fontsize=10.5, arrowprops=dict(arrowstyle="->", color=WINE))
ax.set_xlabel("number of clusters k"); ax.set_ylabel("inertia / WCSS")
ax.set_title("Elbow method: WCSS always falls as k grows; look for the bend")
save(fig, "unsupmetrics-elbow")
print("inertia", np.round(inert, 1))

# ---------- 2. silhouette worked example ----------
W = np.array([[1, 1], [2, 1], [1, 2], [6, 5], [7, 5], [6, 6], [4, 3]], float)
lab = np.array([0, 0, 0, 1, 1, 1, 1])
s = silhouette_samples(W, lab)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
for k, col in [(0, TEAL), (1, WINE)]:
    ax.scatter(W[lab == k, 0], W[lab == k, 1], s=120, color=col, zorder=3, label=f"cluster {'A' if k == 0 else 'B'}")
for i, (x, y) in enumerate(W):
    ax.text(x + 0.18, y + 0.18, f"P{i+1}", fontsize=10)
G = W[6]
for j in range(6):
    ax.plot([G[0], W[j, 0]], [G[1], W[j, 1]], color=TEAL if lab[j] == 0 else WINE, lw=0.9, ls=":" if lab[j] == 0 else "-", alpha=0.7)
ax.annotate("P7: borderline,\ns = −0.04", xy=(4, 3), xytext=(3.4, 5.6), color=ORANGE, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.set_xlim(0, 8); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=10)
ax.set_title("Seven points, two clusters"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax = axes[1]
order = [0, 1, 2, 3, 4, 5, 6]
cols = [TEAL] * 3 + [WINE] * 4
ax.barh(np.arange(7)[::-1], s[order], color=[ORANGE if v < 0 else c for v, c in zip(s[order], cols)])
for yi, v, i in zip(np.arange(7)[::-1], s[order], order):
    ax.text(v + 0.02 if v > 0 else 0.02, yi, f"{v:.2f}", va="center", fontsize=10)
ax.set_yticks(np.arange(7)[::-1]); ax.set_yticklabels([f"P{i+1}" for i in order])
ax.axvline(s.mean(), color=INK, ls="--", lw=1.3)
ax.text(s.mean() + 0.02, -0.75, f"mean = {s.mean():.2f}", fontsize=10)
ax.set_ylim(-1, 6.6)
ax.set_xlim(-0.2, 1); ax.set_xlabel("silhouette s(i)"); ax.set_title("Silhouette of each point")
save(fig, "unsupmetrics-silworked")

# ---------- 3. silhouette plot for k = 3, 4, 5 ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True)
for ax, k in zip(axes, [3, 4, 5]):
    lb = KMeans(k, n_init=10, random_state=0).fit_predict(P)
    sv = silhouette_samples(P, lb); y0 = 0
    for c in range(k):
        v = np.sort(sv[lb == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=CYCLE[c % len(CYCLE)], alpha=0.85)
        y0 += len(v) + 6
    m = sv.mean(); ax.axvline(m, color=INK, ls="--", lw=1.3)
    ax.set_title(f"k = {k}: mean s = {m:.2f}", color=TEAL if k == 4 else INK)
    ax.set_yticks([]); ax.set_xlabel("silhouette s(i)")
    ax.set_xlim(-0.2, 1)
fig.suptitle("Silhouette plots: one band per cluster, sorted by score (4-blob data)", y=1.03, fontsize=13, fontweight="bold")
save(fig, "unsupmetrics-silplot")

# ---------- 4. scree ----------
X = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 8)) + 0.4 * rng.normal(size=(300, 8))
pca = PCA().fit((X - X.mean(0)) / X.std(0))
r = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, 9), r, color=LIGHT_TEAL, edgecolor=TEAL, label="explained variance ratio")
ax.plot(range(1, 9), np.cumsum(r), "o-", color=WINE, label="cumulative")
ax.axhline(0.9, color=MUTED, ls=":", lw=1.3); ax.text(8.4, 0.905, "90%", color=MUTED, fontsize=10, va="bottom", ha="right")
for i, v in enumerate(np.cumsum(r)[:4]):
    ax.text(i + 1, v + 0.035, f"{v:.1%}", ha="center", fontsize=9.5, color=WINE)
ax.set_xlabel("principal component"); ax.set_ylabel("share of total variance"); ax.set_ylim(0, 1.1)
ax.legend(loc="center right"); ax.set_title("Scree plot: 3 components keep almost all the variance")
save(fig, "unsupmetrics-scree")
print("scree", r.round(3), np.cumsum(r).round(3))

# ---------- 5. anomaly score distribution ----------
from sklearn.ensemble import IsolationForest
N = rng.normal(0, 1, (980, 2)); A = rng.uniform(-6, 6, (20, 2)); A = A[np.linalg.norm(A, axis=1) > 3.5]
D = np.vstack([N, A]); truth = np.r_[np.zeros(len(N)), np.ones(len(A))]
iso = IsolationForest(contamination=0.02, random_state=0).fit(D)
score = -iso.score_samples(D)
thr = np.quantile(score, 0.98)
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(score.min(), score.max(), 40)
ax.hist(score[truth == 0], bins=bins, color=LIGHT_TEAL, edgecolor=TEAL, label="normal points")
ax.hist(score[truth == 1], bins=bins, color=WINE, alpha=0.85, label="true anomalies (normally unknown)")
ax.axvline(thr, color=INK, ls="--", lw=1.4)
ax.text(thr + 0.005, 120, "threshold = 98th percentile\n(contamination 2%)\nflagged →", fontsize=10)
ax.set_xlabel("anomaly score (higher = more unusual)"); ax.set_ylabel("count")
ax.set_ylim(0, ax.get_ylim()[1] * 1.35); ax.legend(loc="upper left", fontsize=9.5, bbox_to_anchor=(0.12, 1))
ax.set_title("Anomaly scores: a long right tail, cut at a chosen threshold")
save(fig, "unsupmetrics-anomscores")
k = 20; top = np.argsort(-score)[:k]
print("n anomalies", len(A), "precision@20", truth[top].mean(), "recall@20", truth[top].sum() / truth.sum())
