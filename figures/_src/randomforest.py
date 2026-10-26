import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np, warnings
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
plt.rcParams["savefig.dpi"] = 130   # resolution for rasterised shading only

# ---------- the cat data ----------
X = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])
y = np.array([1,1,0,0,1,1,0,1,0,0])
NAMES = ["Ear shape", "Face shape", "Whiskers"]

# ---------- 1. which feature becomes the root? 100 trees each ----------
bag = BaggingClassifier(DecisionTreeClassifier(criterion="entropy"), n_estimators=100, random_state=0).fit(X, y)
rf = RandomForestClassifier(n_estimators=100, max_features=1, criterion="entropy", random_state=0).fit(X, y)
rb = np.bincount([t.tree_.feature[0] for t in bag.estimators_], minlength=3)
rr = np.bincount([t.tree_.feature[0] for t in rf.estimators_], minlength=3)
print("root counts bagging", rb, "random forest", rr)
fig, ax = plt.subplots(figsize=(8, 3.8))
xs = np.arange(3); w = 0.36
b1 = ax.bar(xs - w/2, rb, w, color=GOLD, label="Bagged trees (all 3 features at every split)")
b2 = ax.bar(xs + w/2, rr, w, color=TEAL, label="Random forest (max_features=1)")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1, str(int(r.get_height())), ha="center", fontsize=11)
ax.axhline(100/3, color=MUTED, ls=":", lw=1.2)
ax.text(2.62, 100/3 + 1.5, "1/3 each", color=MUTED, fontsize=9.5, ha="right")
ax.set_xticks(xs, NAMES); ax.set_ylabel("trees with this root (of 100)")
ax.set_ylim(0, 75); ax.legend(fontsize=9.5, loc="upper right")
ax.set_title("Root feature of 100 trees grown on the cat table")
save(fig, "randomforest-roots")

# ---------- 2. diagram: a fresh random feature subset at EVERY node ----------
def box(ax, x, yy, text, fc, w=2.7, h=0.95, fs=9.5):
    ax.add_patch(FancyBboxPatch((x - w/2, yy - h/2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=INK, lw=1.2))
    ax.text(x, yy, text, ha="center", va="center", fontsize=fs, color=INK)
def arr(ax, x0, y0, x1, y1, lab):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    ax.text((x0 + x1)/2 + (-0.15 if x1 < x0 else 0.15), (y0 + y1)/2 + 0.1, lab, fontsize=9, color=MUTED,
            style="italic", ha="right" if x1 < x0 else "left")
fig, ax = plt.subplots(figsize=(10, 4.6))
box(ax, 0, 3, "ROOT\ndrawn this time: {Whiskers}\nsplit on Whiskers", LIGHT_GOLD, w=3.2)
box(ax, -3, 1.4, "NODE\nnew draw: {Ear shape}\nsplit on Ear shape", LIGHT_GOLD, w=3.0)
box(ax, 3, 1.4, "NODE\nnew draw: {Face shape}\nsplit on Face shape", LIGHT_GOLD, w=3.0)
arr(ax, -0.6, 2.5, -2.5, 1.9, "present"); arr(ax, 0.6, 2.5, 2.5, 1.9, "absent")
for x, t, c in [(-4.2, "leaf", LIGHT_TEAL), (-1.8, "leaf", LIGHT_WINE), (1.8, "leaf", LIGHT_TEAL), (4.2, "...", RULE)]:
    box(ax, x, 0, t, c, w=1.3, h=0.55, fs=10)
for x0, x1 in [(-3.3, -4.1), (-2.7, -1.9), (2.7, 1.9), (3.3, 4.1)]:
    ax.annotate("", xy=(x1, 0.3), xytext=(x0, 0.92), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
ax.text(0, -0.75, "Each gold box = one split. Before EVERY split, the forest draws a fresh random subset of "
        "features\n(size max_features) and picks the best split among those only. The root is just the first draw.",
        ha="center", va="center", fontsize=10, color=INK)
ax.set_xlim(-5.2, 5.2); ax.set_ylim(-1.2, 3.65); ax.axis("off")
ax.set_title("One tree in a random forest (illustration, max_features = 1)")
save(fig, "randomforest-everysplit")

# ---------- 3. decision boundaries: tree vs bagging vs random forest ----------
Xm, ym = make_moons(n_samples=400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=0)
models = [("One deep tree", DecisionTreeClassifier(random_state=0)),
          ("Bagging, 200 trees", BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=0)),
          ("Random forest, 200 trees", RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0))]
xx, yy = np.meshgrid(np.linspace(-2.2, 3.2, 250), np.linspace(-1.7, 2.2, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
for ax, (name, m) in zip(axes, models):
    m.fit(Xtr, ytr)
    acc = m.score(Xte, yte); print(name, "test", acc)
    P = m.predict_proba(grid)[:, 1].reshape(xx.shape)
    cs = ax.contourf(xx, yy, P, levels=[0, 0.5, 1], colors=[LIGHT_WINE, LIGHT_TEAL], alpha=0.6)
    cs.set_rasterized(True)
    ax.contour(xx, yy, P, levels=[0.5], colors=INK, linewidths=1.2)
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=9, color=TEAL)
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=9, color=WINE)
    ax.set_title(f"{name}\ntest accuracy {acc:.3f}", fontsize=11.5)
    ax.set_xlabel("feature 1")
axes[0].set_ylabel("feature 2")
print("RF OOB moons", models[2][1].oob_score_)
save(fig, "randomforest-boundaries")

# ---------- 4. correlation between trees and accuracy, 20-feature data ----------
Xc, yc = make_classification(n_samples=1000, n_features=20, n_informative=6, flip_y=0.1, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
t = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
b = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, oob_score=True, random_state=0, n_jobs=-1).fit(Xtr, ytr)
r = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0, n_jobs=-1).fit(Xtr, ytr)
def rho(ms):
    P = np.array([m.predict_proba(Xte)[:, 1] for m in ms]); C = np.corrcoef(P)
    return C[np.triu_indices(len(ms), 1)].mean()
rb_, rr_ = rho(b.estimators_), rho(r.estimators_)
accs = [t.score(Xte, yte), b.score(Xte, yte), r.score(Xte, yte)]
print("rho bag", rb_, "rho rf", rr_, "acc", accs, "oob", b.oob_score_, r.oob_score_)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
bars = ax.bar(["Bagged trees", "Random forest"], [rb_, rr_], color=[GOLD, TEAL], width=0.5)
for bb, v in zip(bars, [rb_, rr_]):
    ax.text(bb.get_x() + bb.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=11)
ax.set_ylim(0, 0.65); ax.set_ylabel(r"average correlation $\rho$")
ax.set_title("How alike are the trees?", fontsize=12)
ax = axes[1]
bars = ax.bar(["One tree", "Bagging", "Random forest"], accs, color=[MUTED, GOLD, TEAL], width=0.55)
for bb, v in zip(bars, accs):
    ax.text(bb.get_x() + bb.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
ax.set_ylim(0.7, 0.9); ax.set_ylabel("test accuracy")
ax.set_title("Accuracy on 300 held-out rows", fontsize=12)
fig.suptitle("20-feature synthetic data: random features at each split make trees less alike", y=1.03,
             fontsize=12.5, fontweight="bold")
save(fig, "randomforest-rho")

# ---------- 5. impurity vs permutation importance ----------
rng = np.random.default_rng(0)
Xi, yi = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=0, shuffle=False,
                             flip_y=0.1, random_state=0)
Xi = np.column_stack([Xi, rng.random(1000), rng.integers(0, 2, 1000)])
names = ["useful 1", "useful 2", "useful 3", "noise\n(continuous)", "noise\n(continuous)",
         "random ID\n(unique per row)", "random\nyes/no"]
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.3, random_state=0)
m = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pi = permutation_importance(m, Xte, yte, n_repeats=20, random_state=0)
print("impurity", m.feature_importances_.round(3), "perm", pi.importances_mean.round(3), "test", m.score(Xte, yte))
fig, ax = plt.subplots(figsize=(10, 4))
xs = np.arange(len(names)); w = 0.38
ax.bar(xs - w/2, m.feature_importances_, w, color=GOLD, label="impurity-based (feature_importances_, train)")
ax.bar(xs + w/2, pi.importances_mean, w, color=TEAL, label="permutation (accuracy drop, test set)")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(xs, names, fontsize=9.5); ax.set_ylabel("importance")
ax.legend(fontsize=9.5)
ax.annotate("pure noise, but impurity\nimportance gives it 0.09", xy=(5 - w/2, m.feature_importances_[5]),
            xytext=(5.25, 0.25), fontsize=9.5, color=WINE, arrowprops=dict(arrowstyle="->", color=WINE))
ax.set_title("Impurity importance rewards many-valued noise; permutation importance doesn't")
save(fig, "randomforest-importance")
