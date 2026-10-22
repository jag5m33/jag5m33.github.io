import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
# this file is called xgboost.py, so drop its own folder from the path before importing the real library
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [q for q in sys.path if os.path.abspath(q or '.') != _here]
import xgboost as xgb

X = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])  # pointy, round, whiskers
y = np.array([1,1,0,0,1,1,0,1,0,0])
names = ["Ear shape", "Face shape", "Whiskers"]
lam, eta = 1.0, 0.3

def box(ax, x, yy, text, fc, w=1.6, h=0.7, fs=10.5):
    ax.add_patch(FancyBboxPatch((x-w/2, yy-h/2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=INK, lw=1.3))
    ax.text(x, yy, text, ha="center", va="center", fontsize=fs)

def arrow(ax, x0, y0, x1, y1, col=MUTED):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))

def mini_tree(ax, x, yy, fc=LIGHT_GOLD, s=0.55):
    ax.add_patch(Circle((x, yy+0.35*s), 0.16*s, fc=fc, ec=INK, lw=1))
    for dx in (-0.35, 0.35):
        ax.plot([x, x+dx*s], [yy+0.35*s, yy-0.25*s], color=MUTED, lw=1)
        ax.add_patch(Circle((x+dx*s, yy-0.3*s), 0.12*s, fc=LIGHT_TEAL, ec=INK, lw=1))

# ---------- 1. bagging vs boosting ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
box(ax, 0, 3.2, "training data", LIGHT_GOLD, w=2.2)
xs = [-2.1, -0.7, 0.7, 2.1]
for i, x in enumerate(xs):
    arrow(ax, 0, 2.85, x, 2.35)
    ax.text(x, 2.15, f"bootstrap\nsample {i+1}", ha="center", va="center", fontsize=8.5, color=MUTED)
    mini_tree(ax, x, 1.3)
    arrow(ax, x, 0.95, 0, 0.35)
box(ax, 0, 0.05, "average the votes", LIGHT_TEAL, w=2.6)
ax.text(0, -0.85, "trees built independently, in parallel\naveraging lowers VARIANCE", ha="center", fontsize=10.5, color=TEAL)
ax.set_xlim(-3, 3); ax.set_ylim(-1.4, 3.7); ax.axis("off"); ax.set_title("Bagging (random forest)")
ax = axes[1]
xs = [-2.2, -0.75, 0.75, 2.2]
lbl = ["tree 1\nfits y", "tree 2\nfits errors\nof 1", "tree 3\nfits errors\nof 1+2", "tree 4\n..."]
for i, x in enumerate(xs):
    mini_tree(ax, x, 2.2)
    ax.text(x, 1.35, lbl[i], ha="center", va="top", fontsize=8.5, color=MUTED)
    if i < 3:
        arrow(ax, x+0.35, 2.2, xs[i+1]-0.35, 2.2, WINE)
        ax.text((x+xs[i+1])/2, 2.45, "residuals", ha="center", fontsize=8, color=WINE)
box(ax, 0, 0.05, "sum: base + η·tree1 + η·tree2 + ...", LIGHT_TEAL, w=4.4)
ax.text(0, -0.85, "trees built one after another\neach corrects the last: lowers BIAS", ha="center", fontsize=10.5, color=WINE)
ax.set_xlim(-3, 3); ax.set_ylim(-1.4, 3.7); ax.axis("off"); ax.set_title("Boosting (gradient boosting, XGBoost)")
save(fig, "xgboost-bagboost")

# ---------- 2. residuals shrinking round by round (real xgboost, depth 2) ----------
rounds = [0, 1, 3, 10, 30]
fig, axes = plt.subplots(1, len(rounds), figsize=(11, 3.4), sharey=True)
for ax, n in zip(axes, rounds):
    if n == 0:
        p = np.full(10, 0.5)
    else:
        m = xgb.XGBClassifier(n_estimators=n, max_depth=2, learning_rate=eta, reg_lambda=lam, gamma=0,
                              base_score=0.5, min_child_weight=0).fit(X, y)
        p = m.predict_proba(X)[:, 1]
    r = y - p
    ax.bar(np.arange(1, 11), r, color=[TEAL if v > 0 else WINE for v in r])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title("start (p = 0.5)" if n == 0 else f"after {n} tree{'s' if n > 1 else ''}", fontsize=11)
    ax.set_xticks([1, 5, 10]); ax.set_xlabel("animal #", fontsize=10)
    ax.text(5.5, -0.62, f"mean |r| = {np.abs(r).mean():.2f}", ha="center", fontsize=9.5, color=INK)
axes[0].set_ylabel("residual y − p"); axes[0].set_ylim(-0.7, 0.6)
fig.suptitle("Each new tree fits what is left over, so the residuals shrink (cat data, depth-2 trees, η = 0.3)",
             y=1.04, fontsize=12.5, fontweight="bold")
save(fig, "xgboost-residuals")

# ---------- 3. similarity: agree vs cancel ----------
p0 = 0.5
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
for ax, (res, title, col) in zip(axes, [([0.5, 0.5, 0.5, 0.5], "Residuals agree (pure node)", TEAL),
                                         ([0.5, -0.5, 0.5, -0.5], "Residuals cancel (mixed node)", WINE)]):
    res = np.array(res)
    ax.bar(range(1, 5), res, color=[TEAL if v > 0 else WINE for v in res], width=0.6)
    ax.axhline(0, color=MUTED, lw=1)
    sim = res.sum()**2 / (len(res)*p0*(1-p0) + lam)
    ax.set_xlabel(f"sum = {res.sum():+.1f}    similarity = {abs(res.sum()):.1f}² / ({len(res)}×0.25 + 1) = {sim:.2f}",
                  fontsize=10.5, color=col)
    ax.set_ylim(-0.7, 0.7); ax.set_xticks([]); ax.set_title(title, color=col)
    ax.set_ylabel("residual")
save(fig, "xgboost-similarity")

# ---------- 4. gain per feature, round 1 ----------
p = np.full(10, 0.5); r = y - p
def sim(rr, pp): return rr.sum()**2 / ((pp*(1-pp)).sum() + lam)
root = sim(r, p)
gains, sl, sr = [], [], []
for j in range(3):
    m = X[:, j] == 1
    sl.append(sim(r[m], p[m])); sr.append(sim(r[~m], p[~m])); gains.append(sl[-1] + sr[-1] - root)
fig, ax = plt.subplots(figsize=(8, 3.6))
yy = np.arange(3)[::-1]
ax.barh(yy, gains, color=[TEAL, MUTED, MUTED], height=0.55)
for yi, g, a, b in zip(yy, gains, sl, sr):
    ax.text(g + 0.04, yi, f"gain = {a:.3f} + {b:.3f} − {root:.0f} = {g:.3f}", va="center", fontsize=10.5)
ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlim(0, 3.2)
ax.set_xlabel("gain = similarity(left) + similarity(right) − similarity(root)")
ax.set_title("Tree 1, root split: ear shape wins again")
save(fig, "xgboost-gains")

# ---------- 5. early stopping curve ----------
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
Xs, ys = make_classification(n_samples=1500, n_features=20, n_informative=6, flip_y=0.12, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(Xs, ys, test_size=0.3, random_state=0)
m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.1, max_depth=6, eval_metric="logloss",
                      early_stopping_rounds=50)
m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
ev = m.evals_result()
tr, va = ev["validation_0"]["logloss"], ev["validation_1"]["logloss"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tr, color=GOLD, label="training log-loss")
ax.plot(va, color=TEAL, label="validation log-loss")
b = m.best_iteration
ax.axvline(b, color=WINE, ls="--", lw=1.4)
ax.annotate(f"best round = {b}\n(validation stops improving)", xy=(b, va[b]), xytext=(b - 30, va[b] + 0.13),
            color=WINE, fontsize=10, arrowprops=dict(arrowstyle="->", color=WINE))
ax.text(len(va) - 1, va[-1] + 0.03, "stops 50 rounds later", ha="right", fontsize=9.5, color=MUTED)
ax.set_xlabel("boosting round (number of trees)"); ax.set_ylabel("log-loss (lower is better)")
ax.legend(loc="upper right"); ax.set_title("Early stopping picks the number of trees for you")
save(fig, "xgboost-earlystop")
print("best iteration", b, "rounds run", len(va))
