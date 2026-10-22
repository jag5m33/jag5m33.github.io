import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

# ---------- the cat data (verified against Ng's counts in the chapter) ----------
ear      = np.array([1,0,0,1,1,1,0,1,0,0])   # 1 = pointy
face     = np.array([1,0,1,0,1,1,0,1,1,1])   # 1 = round
whisk    = np.array([1,1,0,1,1,0,0,0,0,0])   # 1 = present
y        = np.array([1,1,0,0,1,1,0,1,0,0])   # 1 = cat
assert (ear.sum(), y[ear==1].sum(), y[ear==0].sum()) == (5, 4, 1)
assert (face.sum(), y[face==1].sum(), y[face==0].sum()) == (7, 4, 1)
assert (whisk.sum(), y[whisk==1].sum(), y[whisk==0].sum()) == (4, 3, 2)

def H(p):
    return 0.0 if p in (0, 1) else -p*np.log2(p) - (1-p)*np.log2(1-p)

def box(ax, x, y, text, fc, ec=INK, w=2.2, h=0.8, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=INK)

def edge(ax, x0, y0, x1, y1, label=None, side=-1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    if label:
        ax.text((x0+x1)/2 + side*0.25, (y0+y1)/2, label, ha="right" if side < 0 else "left", va="center",
                fontsize=10, color=MUTED, style="italic")

def tokens(ax, x, y, n_cat, n_dog, r=0.13, gap=0.3):
    """draw small circles: teal C = cat, wine D = not cat"""
    items = ["C"]*n_cat + ["D"]*n_dog
    x0 = x - gap*(len(items)-1)/2
    for i, t in enumerate(items):
        col = TEAL if t == "C" else WINE
        ax.add_patch(Circle((x0 + i*gap, y), r, fc=col, ec="none"))
        ax.text(x0 + i*gap, y, t, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

# ---------- 1. anatomy ----------
fig, ax = plt.subplots(figsize=(9, 4.6))
box(ax, 0, 3, "Ear shape?", LIGHT_GOLD, bold=True)
box(ax, -2.6, 1.5, "Face shape?", LIGHT_GOLD, bold=True)
box(ax, 2.6, 1.5, "Whiskers?", LIGHT_GOLD, bold=True)
for x, t in [(-3.9, "Cat"), (-1.3, "Not cat"), (1.3, "Cat"), (3.9, "Not cat")]:
    box(ax, x, 0, t, LIGHT_TEAL if t == "Cat" else LIGHT_WINE, w=1.6)
edge(ax, -0.4, 2.6, -2.2, 1.9, "pointy", -1); edge(ax, 0.4, 2.6, 2.2, 1.9, "floppy", 1)
edge(ax, -2.9, 1.1, -3.8, 0.4, "round", -1); edge(ax, -2.3, 1.1, -1.4, 0.4, "not round", 1)
edge(ax, 2.3, 1.1, 1.4, 0.4, "present", -1); edge(ax, 2.9, 1.1, 3.8, 0.4, "absent", 1)
ax.annotate("root node\n(first question, all data)", xy=(1.15, 3), xytext=(3.2, 3.35), fontsize=10, color=GOLD,
            arrowprops=dict(arrowstyle="->", color=GOLD), va="center")
ax.annotate("decision node\n(another question)", xy=(-3.75, 1.5), xytext=(-6.3, 2.5), fontsize=10, color=GOLD,
            arrowprops=dict(arrowstyle="->", color=GOLD), va="center")
ax.annotate("branch\n(one answer)", xy=(1.3, 2.25), xytext=(-0.2, 1.55), fontsize=10, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED), ha="center", va="center")
ax.annotate("leaf node\n(prediction)", xy=(4.75, 0), xytext=(5.4, 1.1), fontsize=10, color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL), va="center")
for d, yy in [(0, 3), (1, 1.5), (2, 0)]:
    ax.text(-6.6, yy, f"depth {d}", color=MUTED, fontsize=9, va="center")
ax.set_xlim(-6.8, 6.8); ax.set_ylim(-0.6, 3.7); ax.axis("off")
ax.set_title("Parts of a decision tree (read from the top down)")
save(fig, "trees-anatomy")

# ---------- 2. purity ----------
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
for ax, (nc, nd, title) in zip(axes, [(5, 0, "Pure: all cats"), (3, 2, "Mixed: 3 cats, 2 not"), (0, 5, "Pure: no cats")]):
    tokens(ax, 0, 0.1, nc, nd, r=0.12, gap=0.32)
    p = nc/(nc+nd)
    ax.text(0, -0.45, f"p(cat) = {p:.1f}    entropy H = {H(p):.2f}", ha="center", fontsize=11)
    ax.set_title(title, color=TEAL if nd == 0 or nc == 0 else WINE)
    ax.set_xlim(-1, 1); ax.set_ylim(-0.7, 0.45); ax.axis("off")
fig.suptitle("A node is pure when every example in it has the same class", y=1.06, fontsize=13, fontweight="bold")
save(fig, "trees-purity")

# ---------- 3. entropy curve ----------
p = np.linspace(0.0005, 0.9995, 400)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(p, [H(v) for v in p], color=TEAL)
ax.fill_between(p, [H(v) for v in p], color=LIGHT_TEAL, alpha=0.4)
for v, lab in [(0.5, "5 cats / 5 not\nH = 1 (most mixed)"), (0.8, "4 cats / 1 not\nH = 0.72")]:
    ax.scatter([v], [H(v)], color=WINE, zorder=3, s=50)
    ax.annotate(lab, xy=(v, H(v)), xytext=(v + 0.07, H(v) - 0.28 if v == 0.8 else 1.06), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=WINE), color=WINE)
ax.scatter([0, 1], [0, 0], color=TEAL, s=60, zorder=3, clip_on=False)
ax.text(0.02, 0.06, "pure (no cats)\nH = 0", fontsize=10, color=TEAL)
ax.text(0.98, 0.06, "pure (all cats)\nH = 0", fontsize=10, color=TEAL, ha="right")
ax.set_xlabel("$p_1$ = fraction of examples in the node that are cats")
ax.set_ylabel("entropy $H(p_1)$ (bits)")
ax.set_ylim(0, 1.25); ax.set_xlim(0, 1)
ax.set_title("Entropy: 0 for a pure node, 1 for a 50/50 node")
save(fig, "trees-entropy")

# ---------- 4. the three candidate root splits ----------
feats = [("Ear shape", ear, "pointy", "floppy"), ("Face shape", face, "round", "not round"),
         ("Whiskers", whisk, "present", "absent")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (name, f, l, r) in zip(axes, feats):
    yl, yr = y[f == 1], y[f == 0]
    hl, hr = H(yl.mean()), H(yr.mean())
    w = len(yl)/10*hl + len(yr)/10*hr
    gain = 1 - w
    box(ax, 0, 2, name + "?", LIGHT_GOLD, w=2.4, bold=True)
    edge(ax, -0.3, 1.6, -1.2, 0.95, l, -1); edge(ax, 0.3, 1.6, 1.2, 0.95, r, 1)
    tokens(ax, -1.35, 0.6, yl.sum(), len(yl)-yl.sum(), r=0.11, gap=0.25)
    tokens(ax, 1.35, 0.6, yr.sum(), len(yr)-yr.sum(), r=0.11, gap=0.25)
    ax.text(-1.35, 0.2, f"{yl.sum()}/{len(yl)} cats\nH = {hl:.2f}", ha="center", va="top", fontsize=10)
    ax.text(1.35, 0.2, f"{yr.sum()}/{len(yr)} cats\nH = {hr:.2f}", ha="center", va="top", fontsize=10)
    best = name == "Ear shape"
    ax.text(0, -0.75, f"gain = {gain:.2f}", ha="center", fontsize=13, fontweight="bold",
            color=TEAL if best else MUTED)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1, 2.5); ax.axis("off")
fig.suptitle("Three candidate root splits: ear shape leaves the purest children", y=1.02, fontsize=13, fontweight="bold")
save(fig, "trees-rootsplits")

# ---------- 5. final tree ----------
fig, ax = plt.subplots(figsize=(9.5, 4.8))
box(ax, 0, 3, "Ear shape?\n10 animals, 5 cats, H = 1.00", LIGHT_GOLD, w=3.6, h=1.0, fs=10.5)
box(ax, -2.8, 1.4, "Face shape?\n5 animals, 4 cats, H = 0.72", LIGHT_GOLD, w=3.4, h=1.0, fs=10.5)
box(ax, 2.8, 1.4, "Whiskers?\n5 animals, 1 cat, H = 0.72", LIGHT_GOLD, w=3.4, h=1.0, fs=10.5)
leaves = [(-4.1, "Cat\n4 of 4 cats", True), (-1.5, "Not cat\n0 of 1", False),
          (1.5, "Cat\n1 of 1", True), (4.1, "Not cat\n0 of 4", False)]
for x, t, c in leaves:
    box(ax, x, -0.2, t, LIGHT_TEAL if c else LIGHT_WINE, w=2.0, h=0.9, fs=10.5)
edge(ax, -0.6, 2.5, -2.2, 1.9, "pointy", -1); edge(ax, 0.6, 2.5, 2.2, 1.9, "floppy", 1)
edge(ax, -3.2, 0.9, -3.9, 0.25, "round", -1); edge(ax, -2.4, 0.9, -1.7, 0.25, "not round", 1)
edge(ax, 2.4, 0.9, 1.7, 0.25, "present", -1); edge(ax, 3.2, 0.9, 3.9, 0.25, "absent", 1)
ax.text(1.95, 3.0, "gain 0.28", color=TEAL, fontsize=10, fontweight="bold", va="center")
ax.text(-2.8, 2.0, "gain 0.72", color=TEAL, fontsize=10, fontweight="bold", ha="center")
ax.text(2.8, 2.0, "gain 0.72", color=TEAL, fontsize=10, fontweight="bold", ha="center")
ax.text(0, -1.05, "Every leaf is pure (H = 0), so the tree stops growing.", ha="center", color=MUTED, fontsize=10.5)
ax.set_xlim(-5.4, 5.4); ax.set_ylim(-1.3, 3.6); ax.axis("off")
ax.set_title("The finished tree learned from the 10 animals")
save(fig, "trees-final")

# ---------- 6. continuous feature: try every threshold ----------
wt = np.array([7.2, 8.4, 9.6, 15.0, 7.6, 13.0, 18.0, 8.8, 20.0, 11.0])
order = np.sort(np.unique(wt)); ths = (order[:-1] + order[1:]) / 2
gains = []
for t in ths:
    l, r = y[wt <= t], y[wt > t]
    gains.append(1 - (len(l)/10*H(l.mean()) + len(r)/10*H(r.mean())))
gains = np.array(gains); bi = gains.argmax()
fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2]))
ax = axes[0]
ax.scatter(wt[y == 1], np.zeros((y == 1).sum()), color=TEAL, s=70, label="cat", zorder=3)
ax.scatter(wt[y == 0], np.zeros((y == 0).sum()), color=WINE, s=70, marker="s", label="not cat", zorder=3)
ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.legend(loc="upper right", ncol=2, fontsize=10)
ax.set_ylim(-0.5, 0.8)
ax = axes[1]
ax.plot(ths, gains, "o-", color=GOLD)
ax.axvline(ths[bi], color=TEAL, ls="--", lw=1.5); axes[0].axvline(ths[bi], color=TEAL, ls="--", lw=1.5)
ax.annotate(f"best threshold: weight <= {ths[bi]:.1f}\ngain = {gains[bi]:.2f}", xy=(ths[bi], gains[bi]),
            xytext=(ths[bi] + 2.5, gains[bi] - 0.05), color=TEAL, fontsize=10.5, arrowprops=dict(arrowstyle="->", color=TEAL))
ax.set_xlabel("weight (lb), candidate threshold = midpoint between neighbouring values")
ax.set_ylabel("information gain")
axes[0].set_title("Continuous feature: try each midpoint, keep the one with the highest gain")
save(fig, "trees-threshold")
print("threshold gains", dict(zip(ths.round(2), gains.round(3))))

# ---------- 7. overfitting: depth vs accuracy ----------
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, yy = make_moons(n_samples=600, noise=0.35, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(X, yy, test_size=0.4, random_state=0)
ds = range(1, 16); tr, va = [], []
for d in ds:
    m = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0).fit(Xtr, ytr)
    tr.append(m.score(Xtr, ytr)); va.append(m.score(Xva, yva))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ds, tr, "o-", color=GOLD, label="training accuracy")
ax.plot(ds, va, "o-", color=TEAL, label="validation accuracy")
b = int(np.argmax(va)) + 1
ax.axvline(b, color=MUTED, ls=":", lw=1.4)
ax.text(b + 0.25, 0.80, f"best validation\n(depth {b})", color=MUTED, fontsize=10)
ax.annotate("", xy=(13, tr[12]), xytext=(13, va[12]), arrowprops=dict(arrowstyle="<->", color=WINE, lw=1.5))
ax.text(12.8, (tr[12] + va[12]) / 2, "gap = overfitting:\nthe tree is memorising\nnoise in the training set", color=WINE, fontsize=10, ha="right", va="center")
ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(loc="lower right")
ax.set_title("Deeper trees always fit training data better, not new data")
save(fig, "trees-overfit")
print("depth acc", list(zip(ds, np.round(tr, 3), np.round(va, 3))))
