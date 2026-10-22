import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
import warnings; warnings.simplefilter("ignore")
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Krish-style two-point example
x = np.array([1.0, 2.0]); y = np.array([2.0, 4.0])
xt = np.array([0.5, 1.5, 2.5, 3.5]); yt = np.array([1.6, 2.7, 4.0, 5.0])
xs = np.linspace(0, 4, 50)
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(xs, 2 * xs, color=WINE, label="perfect fit: ŷ = 2x  (training cost 0)")
ax.plot(xs, 1.2 * xs + 1.2, color=TEAL, label="ridge-style: ŷ = 1.2x + 1.2  (flatter)")
ax.scatter(x, y, s=90, color=INK, zorder=4, label="2 training points")
ax.scatter(xt, yt, s=60, facecolor="none", edgecolor=ORANGE, lw=2, zorder=4, label="new (test) points")
for xi, yi in zip(xt, yt):
    ax.plot([xi, xi], [yi, 2 * xi], color=LIGHT_WINE, lw=3, zorder=1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_xlim(0, 4); ax.set_ylim(0, 8.3)
ax.legend(fontsize=10, loc="upper left")
ax.text(3.95, 1.0, "pink bars = test misses of the\nperfect-fit line (test SSE 5.45);\nteal line's test SSE is 0.33", ha="right", fontsize=10, color=MUTED)
ax.set_title("Two training points: the perfect line overfits")
save(fig, "regularisation-twopoints")

# 2. cost breakdown vs slope (lambda = 1, b at its best for each slope)
ws = np.linspace(0, 2.2, 200)
bs = y.mean() - ws * x.mean()
sse = np.array([((y - (w * x + b)) ** 2).sum() for w, b in zip(ws, bs)])
pen = ws ** 2
fig, ax = plt.subplots(figsize=(8, 3.9))
ax.plot(ws, sse, color=BLUE, ls="--", label="training error Σ(y − ŷ)²")
ax.plot(ws, pen, color=GOLD, ls="--", label="penalty λ·slope²  (λ = 1)")
ax.plot(ws, sse + pen, color=INK, label="total ridge cost")
for w, c in [(2.0, WINE), (1.2, TEAL)]:
    ax.scatter([w], [((y - (w * x + y.mean() - w * x.mean())) ** 2).sum() + w ** 2], color=c, s=70, zorder=3)
ax.annotate("perfect fit: 0 + 4.00 = 4.00", (2.0, 4.0), (1.25, 4.6), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
ax.annotate("w = 1.2: 0.32 + 1.44 = 1.76", (1.2, 1.76), (0.95, 3.2), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
wm = 2 / 3
ax.annotate("ridge minimum w = 0.67\n(total 1.33)", (wm, 4 / 3), (0.05, 2.6), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
ax.set_xlabel("slope w (intercept re-fitted for each slope)"); ax.set_ylabel("cost"); ax.set_ylim(0, 5.2)
ax.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.06))
ax.set_title("Ridge cost = error + penalty")
save(fig, "regularisation-costparts")

# 3. geometry: circle vs diamond
w_ols = np.array([2.2, 0.9])
A = np.array([[1.0, 0.55], [0.55, 0.6]])
g = np.linspace(-1.8, 3.2, 300); W1, W2 = np.meshgrid(g, g)
D = np.stack([W1 - w_ols[0], W2 - w_ols[1]], -1)
Q = np.einsum("...i,ij,...j->...", D, A, D)
def touch(norm):
    best, arg = 1e9, None
    th = np.linspace(0, 2 * np.pi, 4000)
    if norm == 2: pts = np.c_[np.cos(th), np.sin(th)]
    else:
        pts = np.c_[np.cos(th), np.sin(th)]; pts = pts / np.abs(pts).sum(1, keepdims=True)
    d = pts - w_ols; q = np.einsum("ni,ij,nj->n", d, A, d)
    i = np.argmin(q); return pts[i], q[i]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (norm, t, col) in zip(axs, [(2, "Ridge (L2): w1² + w2² ≤ t", TEAL), (1, "Lasso (L1): |w1| + |w2| ≤ t", WINE)]):
    p, q = touch(norm)
    ax.contour(W1, W2, Q, levels=[q * 0.25, q * 0.6, q, q * 1.7, q * 2.6], colors=[MUTED], linewidths=1)
    ax.contour(W1, W2, Q, levels=[q], colors=[BLUE], linewidths=2)
    if norm == 2: ax.add_patch(plt.Circle((0, 0), 1, color=col, alpha=0.25, ec=col, lw=2))
    else: ax.add_patch(plt.Polygon([[1, 0], [0, 1], [-1, 0], [0, -1]], color=col, alpha=0.25, ec=col, lw=2))
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.scatter(*w_ols, color=INK, s=50, zorder=4); ax.annotate("unpenalised\nbest (w1, w2)", w_ols, (1.75, 2.45), fontsize=10, arrowprops=dict(arrowstyle="->", color=INK), bbox=dict(fc=PAPER, ec="none", alpha=0.9))
    ax.scatter(*p, color=GOLD, s=110, marker="*", zorder=5, edgecolor=INK, lw=0.5)
    lab = f"touches at ({p[0]:.2f}, {p[1]:.2f})"
    ax.annotate(lab + ("\nboth weights shrunk,\nneither zero" if norm == 2 else "\nw2 is exactly 0:\na corner"), p, (-1.7, -1.6) if norm == 2 else (-1.7, -1.6), fontsize=10, arrowprops=dict(arrowstyle="->", color=INK), bbox=dict(fc=PAPER, ec="none", alpha=0.9))
    ax.set_xlim(-1.8, 3.2); ax.set_ylim(-1.8, 3.2); ax.set_aspect("equal")
    ax.set_xlabel("w1"); ax.set_ylabel("w2"); ax.set_title(t, fontsize=12)
    print(norm, p)
fig.tight_layout(); save(fig, "regularisation-geometry")

# 4. coefficient paths (diabetes-like synthetic data, scaled)
rng = np.random.default_rng(0)
n, p = 120, 8
Z = rng.normal(size=(n, p)); Z[:, 1] = Z[:, 0] * 0.9 + rng.normal(0, 0.45, n)  # 0 and 1 correlated
true = np.array([3.0, 0.0, -2.0, 1.5, 0.0, 0.8, 0.0, -0.5])
yy = Z @ true + rng.normal(0, 2.0, n)
Zs = StandardScaler().fit_transform(Z)
lr = np.logspace(-2, 4, 60); ll = np.logspace(-3, 0.7, 60)
cr = np.array([Ridge(alpha=a).fit(Zs, yy).coef_ for a in lr])
cl = np.array([Lasso(alpha=a, max_iter=50000).fit(Zs, yy).coef_ for a in ll])
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
cols = [TEAL, WINE, GOLD, BLUE, ORANGE, MUTED, "#5b8c5a", "#9b6bb0"]
for j in range(p):
    axs[0].plot(lr, cr[:, j], color=cols[j], lw=1.6, label=f"w{j+1}")
    axs[1].plot(ll, cl[:, j], color=cols[j], lw=1.6)
axs[0].set_xscale("log"); axs[1].set_xscale("log")
axs[0].set_title("Ridge: weights shrink smoothly, never hit 0", fontsize=11)
axs[1].set_title("Lasso: weights hit exactly 0 one by one", fontsize=11)
for ax in axs: ax.axhline(0, color=INK, lw=0.8); ax.set_xlabel("λ (sklearn alpha), log scale  →  stronger")
axs[0].set_ylabel("coefficient value"); axs[0].legend(fontsize=8, ncol=2, loc="lower left")
fig.tight_layout(); save(fig, "regularisation-paths")
print("lasso zeros at alpha=0.3:", (Lasso(alpha=0.3).fit(Zs, yy).coef_ == 0).sum())

# 5. train / val error vs lambda (polynomial ridge)
rng = np.random.default_rng(7)
def truth(x): return np.sin(2 * np.pi * x) + 0.5 * x
xtr = rng.uniform(0, 1, 15); ytr = truth(xtr) + rng.normal(0, 0.3, 15)
xva = rng.uniform(0, 1, 200); yva = truth(xva) + rng.normal(0, 0.3, 200)
lams = np.logspace(-8, 2, 50); tr, va = [], []
for a in lams:
    m = make_pipeline(PolynomialFeatures(12, include_bias=False), StandardScaler(), Ridge(alpha=a)).fit(xtr[:, None], ytr)
    tr.append(mean_squared_error(ytr, m.predict(xtr[:, None]))); va.append(mean_squared_error(yva, m.predict(xva[:, None])))
tr, va = np.array(tr), np.array(va); k = np.argmin(va)
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.plot(lams, tr, color=TEAL, label="training error"); ax.plot(lams, va, color=WINE, label="validation error")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(lams[k], color=GOLD, ls="--", lw=1.4)
ax.annotate(f"best λ ≈ {lams[k]:.0e}", (lams[k], va[k]), (lams[k] * 30, va[k] * 0.55), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
ax.text(lams[2], va.max() * 0.7, "λ too small:\noverfit (high variance)\nbig gap", fontsize=10, va="top")
ax.text(lams[-1], 0.27, "λ too big:\nunderfit (high bias)\nboth high", fontsize=10, ha="right", va="center")
ax.set_xlabel("λ (log scale)"); ax.set_ylabel("MSE (log scale)"); ax.legend(fontsize=10, loc="lower right")
ax.set_title("Degree-12 polynomial, 15 points: choosing λ")
fig.tight_layout(); save(fig, "regularisation-lambda")
print("best lam", lams[k], tr[0], va[0], tr[-1], va[-1], va[k])
