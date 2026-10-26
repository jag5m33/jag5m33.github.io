import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np, warnings
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
plt.rcParams["savefig.dpi"] = 130   # resolution for the rasterised shading only

# ---------- the cat data (same rows as the trees chapter) ----------
X = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])
y = np.array([1,1,0,0,1,1,0,1,0,0])
NAMES = ["Ear shape", "Face shape", "Whiskers"]

# the three bootstrap samples used in the chapter (seed 2)
rng = np.random.default_rng(2)
S = [np.sort(rng.integers(0, 10, 10)) for _ in range(3)]
for s in S:
    print("sample (1-based):", list(s + 1), "unique", len(set(s)), "OOB", sorted(set(range(1, 11)) - set(s + 1)))

def cell(ax, x, yy, txt, fc, w=0.8, h=0.8, ec="none", tc="white", fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x - w/2, yy - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, yy, txt, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")

# ---------- 1. bootstrap samples ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.text(-0.9, 4.4, "Original\n10 rows", ha="right", va="center", fontsize=11, fontweight="bold")
for i in range(10):
    cell(ax, i, 4.4, str(i + 1), TEAL if y[i] else WINE)
for k, s in enumerate(S):
    yy = 2.6 - 1.15 * k
    ax.text(-0.9, yy, f"Sample {k+1}", ha="right", va="center", fontsize=11, fontweight="bold")
    counts = np.bincount(s, minlength=10)
    for j, r in enumerate(s):
        repeat = counts[r] > 1
        cell(ax, j, yy, str(r + 1), TEAL if y[r] else WINE, ec=GOLD if repeat else "none")
    oob = sorted(set(range(10)) - set(s))
    ax.text(10.0, yy, "missing (OOB): " + ", ".join(str(o + 1) for o in oob),
            ha="left", va="center", fontsize=10.5, color=MUTED)
ax.annotate("", xy=(4.5, 3.05), xytext=(4.5, 3.95), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
ax.text(4.75, 3.5, "draw 10 times, with replacement", ha="left", va="center", fontsize=10, color=MUTED, style="italic")
ax.add_patch(FancyBboxPatch((0.3, -1.25), 0.4, 0.35, boxstyle="round,pad=0.02", fc="none", ec=GOLD, lw=1.6))
ax.text(0.95, -1.08, "gold outline = row drawn more than once", va="center", fontsize=10, color=INK)
cell(ax, 8.6, -1.08, "", TEAL, w=0.35, h=0.35); ax.text(8.9, -1.08, "cat", va="center", fontsize=10)
cell(ax, 9.9, -1.08, "", WINE, w=0.35, h=0.35); ax.text(10.2, -1.08, "not cat", va="center", fontsize=10)
ax.set_xlim(-2.6, 14.6); ax.set_ylim(-1.5, 5.0); ax.axis("off")
ax.set_title("Bootstrap sampling: same size as the data, some rows repeated, some left out")
save(fig, "bagging-bootstrap")

# ---------- 2. (1 - 1/n)^n ----------
n = np.arange(1, 101)
p = (1 - 1/n) ** n
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(n, p, color=TEAL, marker="o", ms=3, lw=1.5)
ax.axhline(np.exp(-1), color=WINE, ls="--", lw=1.4)
ax.text(100, np.exp(-1) + 0.012, r"limit $e^{-1}$ = 0.368", color=WINE, ha="right", va="bottom", fontsize=11)
for nn in [2, 10]:
    v = (1 - 1/nn) ** nn
    ax.scatter([nn], [v], color=GOLD, zorder=3, s=55)
    ax.annotate(f"n = {nn}: {v:.3f}", xy=(nn, v), xytext=(nn + 12, v - 0.06 if nn == 10 else v - 0.03),
                fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="->", color=GOLD))
ax.set_xlabel("n (number of rows, and number of draws)")
ax.set_ylabel("P(a given row is never drawn)")
ax.set_ylim(0.2, 0.42)
ax.set_title("The chance a row is left out settles at about 36.8%")
save(fig, "bagging-oob")

# ---------- 3. one deep tree vs bagged trees (two moons) ----------
Xm, ym = make_moons(n_samples=400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=0)
tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, oob_score=True, random_state=0).fit(Xtr, ytr)
print("moons tree test", tree.score(Xte, yte), "bag test", bag.score(Xte, yte), "bag OOB", bag.oob_score_)
xx, yy2 = np.meshgrid(np.linspace(-2.2, 3.2, 250), np.linspace(-1.7, 2.2, 250))
grid = np.c_[xx.ravel(), yy2.ravel()]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, m, title in [(axes[0], tree, f"One deep tree (test acc. {tree.score(Xte, yte):.3f})"),
                     (axes[1], bag, f"200 bagged trees (test acc. {bag.score(Xte, yte):.3f})")]:
    P = m.predict_proba(grid)[:, 1].reshape(xx.shape)
    cs = ax.contourf(xx, yy2, P, levels=[0, 0.5, 1], colors=[LIGHT_WINE, LIGHT_TEAL], alpha=0.6)
    cs.set_rasterized(True)   # keep the SVG small
    ax.contour(xx, yy2, P, levels=[0.5], colors=INK, linewidths=1.3)
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=12, color=TEAL, label="class 1")
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=12, color=WINE, label="class 0")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("feature 1")
axes[0].set_ylabel("feature 2")
axes[0].legend(loc="lower left", fontsize=9)
save(fig, "bagging-boundary")

# ---------- 4. variance of an average of B models ----------
B = np.arange(1, 51)
fig, ax = plt.subplots(figsize=(8, 4))
for rho, col in [(0.0, TEAL), (0.3, BLUE), (0.6, GOLD), (1.0, WINE)]:
    v = rho + (1 - rho) / B
    ax.plot(B, v, color=col, label=rf"$\rho$ = {rho}")
    ax.text(50.8, v[-1], f"{v[-1]:.2f}", color=col, va="center", fontsize=10)
ax.set_xlabel("B (number of trees averaged)")
ax.set_ylabel(r"variance of the average ($\sigma^2$ = 1)")
ax.set_ylim(0, 1.3)
ax.legend(title="correlation between trees", fontsize=10, title_fontsize=10, loc="upper right", ncol=4,
          bbox_to_anchor=(1.0, 1.02))
ax.annotate(r"floor = $\rho\sigma^2$: more trees can't go below it", xy=(30, 0.6 + 0.4/30), xytext=(12, 0.8),
            fontsize=10, color=GOLD, arrowprops=dict(arrowstyle="->", color=GOLD))
ax.set_title(r"Var(average) = $\rho\sigma^2 + (1-\rho)\sigma^2/B$")
save(fig, "bagging-variance")

# ---------- 5. three stumps, three votes ----------
new = np.array([[0, 1, 1]])   # floppy, round, whiskers present
fig, ax = plt.subplots(figsize=(10, 3.6))
votes = []
for k, s in enumerate(S):
    t = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0).fit(X[s], y[s])
    f = t.tree_.feature[0]
    pc = t.predict_proba(new)[0, 1]; v = int(t.predict(new)[0]); votes.append(v)
    x0 = k * 3.4
    cell(ax, x0, 2.0, f"Tree {k+1}: {NAMES[f]}?", LIGHT_GOLD, w=2.6, h=0.6, tc=INK, fs=10.5)
    lab = {0: ("floppy", "pointy"), 1: ("not round", "round"), 2: ("absent", "present")}[f]
    for b, dx in [(0, -0.75), (1, 0.75)]:
        m = X[s][:, f] == b
        nc, nn = y[s][m].sum(), m.sum()
        on_path = new[0, f] == b
        cell(ax, x0 + dx, 0.85, f"{nc} cats / {nn}", LIGHT_TEAL if nc / nn > 0.5 else LIGHT_WINE,
             w=1.3, h=0.55, tc=INK, fs=9.5, ec=INK if on_path else "none", bold=on_path)
        ax.annotate("", xy=(x0 + dx, 1.13), xytext=(x0 + dx * 0.4, 1.7),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        ax.text(x0 + dx * 1.15, 1.45, lab[b], fontsize=9, color=MUTED, style="italic",
                ha="right" if dx < 0 else "left")
    ax.text(x0, 0.05, f"P(cat) = {pc:.2f}  ->  vote: {'cat' if v else 'not cat'}", ha="center", fontsize=10.5,
            color=TEAL if v else WINE, fontweight="bold")
ax.text(3.4, -0.6, f"Majority vote: {sum(votes)} cat vs {3 - sum(votes)} not cat  ->  not cat",
        ha="center", fontsize=12, fontweight="bold")
ax.set_xlim(-1.6, 8.4); ax.set_ylim(-0.9, 2.4); ax.axis("off")
ax.set_title("New animal: floppy ears, round face, whiskers. Outlined leaf = where it lands.", fontsize=12)
save(fig, "bagging-votes")
