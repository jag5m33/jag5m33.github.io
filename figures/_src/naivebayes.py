import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

def box(ax, x, yy, text, fc, w=2.4, h=0.8, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w/2, yy - h/2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=INK, lw=1.2))
    ax.text(x, yy, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal")
def arr(ax, x0, y0, x1, y1, lab=None, side=1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    if lab:
        ax.text((x0 + x1)/2 + 0.12*side, (y0 + y1)/2 + 0.05, lab, fontsize=9.5, color=MUTED, style="italic",
                ha="left" if side > 0 else "right")

# ---------- 1. medical test as natural frequencies ----------
N, prev, sens, fpr = 10000, 0.01, 0.90, 0.05
sick = N * prev; healthy = N - sick
tp, fn = sick * sens, sick * (1 - sens)
fp, tn = healthy * fpr, healthy * (1 - fpr)
post = tp / (tp + fp)
print("sick", sick, "tp", tp, "fp", fp, "positives", tp + fp, "posterior", post)
fig, ax = plt.subplots(figsize=(10, 4.4))
box(ax, 0, 3, f"{N:,.0f} people", LIGHT_GOLD, bold=True)
box(ax, -2.6, 1.6, f"{sick:,.0f} have the disease", LIGHT_WINE)
box(ax, 2.6, 1.6, f"{healthy:,.0f} healthy", LIGHT_TEAL)
arr(ax, -0.5, 2.6, -2.2, 2.0, "1% (prior)", -1); arr(ax, 0.5, 2.6, 2.2, 2.0, "99%", 1)
leaves = [(-3.9, f"{tp:,.0f} test +", WINE, True), (-1.3, f"{fn:,.0f} test −", RULE, False),
          (1.3, f"{fp:,.0f} test +", WINE, True), (3.9, f"{tn:,.0f} test −", RULE, False)]
for x, t, c, pos in leaves:
    box(ax, x, 0.2, t, LIGHT_WINE if pos else "white", w=2.0, fs=10.5, bold=pos)
arr(ax, -2.9, 1.2, -3.7, 0.6, "90% (sensitivity)", -1); arr(ax, -2.3, 1.2, -1.5, 0.6, "10%", 1)
arr(ax, 2.3, 1.2, 1.5, 0.6, "5% (false +)", -1); arr(ax, 2.9, 1.2, 3.7, 0.6, "95%", 1)
ax.text(0, -0.75, f"Everyone who tests positive: {tp:,.0f} + {fp:,.0f} = {tp + fp:,.0f}.   "
        f"Of those, really ill: {tp:,.0f} / {tp + fp:,.0f} = {post:.3f}",
        ha="center", fontsize=11.5, fontweight="bold", color=INK)
ax.set_xlim(-5.2, 5.2); ax.set_ylim(-1.1, 3.5); ax.axis("off")
ax.set_title("Bayes' theorem as counts: a positive test means only a 15% chance of disease")
save(fig, "naivebayes-medical")

# ---------- 2. smoothed word likelihoods for the toy corpus ----------
docs = ["pay fee invoice", "pay fee notice", "terminate notice", "terminate agreement notice"]
lab = np.array(["Payment", "Payment", "Termination", "Termination"])
cv = CountVectorizer(); Xc = cv.fit_transform(docs).toarray(); vocab = list(cv.get_feature_names_out())
V = len(vocab)
probs = {}
for c in ["Payment", "Termination"]:
    cnt = Xc[lab == c].sum(0)
    probs[c] = (cnt + 1) / (cnt.sum() + V)
    print(c, dict(zip(vocab, cnt)), "total", cnt.sum(), "smoothed", probs[c].round(4))
new = ["terminate", "fee", "notice"]
fig, ax = plt.subplots(figsize=(9, 4))
xs = np.arange(V); w = 0.38
b1 = ax.bar(xs - w/2, probs["Payment"], w, color=GOLD, label="P(word | Payment), denominator 6 + 6 = 12")
b2 = ax.bar(xs + w/2, probs["Termination"], w, color=TEAL, label="P(word | Termination), denominator 5 + 6 = 11")
fr = {"Payment": ["1/12", "3/12", "2/12", "2/12", "3/12", "1/12"],
      "Termination": ["2/11", "1/11", "1/11", "3/11", "1/11", "3/11"]}
for bars, c in [(b1, "Payment"), (b2, "Termination")]:
    for bb, f in zip(bars, fr[c]):
        ax.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 0.005, f, ha="center", fontsize=8.5)
ax.set_xticks(xs, vocab)
for t in ax.get_xticklabels():
    if t.get_text() in new:
        t.set_fontweight("bold"); t.set_color(WINE)
ax.set_ylabel("smoothed probability")
ax.set_ylim(0, 0.36); ax.legend(fontsize=9.5, loc="upper left")
ax.set_title("Word likelihoods with Laplace smoothing (alpha = 1). Bold = words in the new sentence")
save(fig, "naivebayes-likelihoods")

# ---------- 3. underflow: product vs sum of logs ----------
n = np.arange(1, 201)
with np.errstate(under="ignore"):
    prod = np.array([np.prod(np.full(k, 0.001)) for k in n])
first_zero = n[prod == 0][0]
print("first n where product of 0.001s is exactly 0:", first_zero, "last nonzero", prod[first_zero - 2])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
ok = prod > 0
ax.plot(n[ok], np.log10(prod[ok]), color=GOLD)
ax.axvline(first_zero, color=WINE, ls="--", lw=1.3)
ax.text(first_zero + 4, -150, f"n = {first_zero}:\nproduct becomes\nexactly 0.0", color=WINE, fontsize=10)
ax.set_xlabel("number of words n"); ax.set_ylabel(r"$\log_{10}$ of the stored product")
ax.set_xlim(0, 200)
ax.set_title("Multiplying: 0.001 × 0.001 × ...", fontsize=12)
ax = axes[1]
ax.plot(n, n * np.log(0.001), color=TEAL)
ax.set_xlabel("number of words n"); ax.set_ylabel("sum of natural logs")
ax.text(105, -300, f"n = 200: {200 * np.log(0.001):.1f}\n(an ordinary number)", color=TEAL, fontsize=10)
ax.set_title("Adding logs: ln 0.001 + ln 0.001 + ...", fontsize=12)
save(fig, "naivebayes-underflow")

# ---------- 4. complement counts for LEDGAR (from Jag's EDA notebook) ----------
total = 60000      # train rows when the class counts were taken in eda.ipynb
classes = [("Governing Laws", 3167), ("Books", 23)]
fig, ax = plt.subplots(figsize=(9, 3.6))
ys = np.arange(len(classes))
for i, (name, k) in enumerate(classes):
    ax.barh(i + 0.18, k, 0.34, color=GOLD)
    ax.barh(i - 0.18, total - k, 0.34, color=TEAL)
    ax.text(k * 1.15, i + 0.18, f"{k:,} clauses (own class: MultinomialNB)", va="center", fontsize=10)
    ax.text((total - k) * 1.15, i - 0.18, f"{total - k:,} (complement: ComplementNB)", va="center", fontsize=10)
ax.set_xscale("log"); ax.set_xlim(10, 10**7.3)
ax.set_yticks(ys, [c for c, _ in classes])
ax.set_xlabel("training clauses used to estimate that class's word weights (log scale)")
ax.set_title("ComplementNB estimates each class from everything that ISN'T that class")
save(fig, "naivebayes-complement")

# ---------- 5. duplicated (correlated) features make NB overconfident ----------
rng = np.random.default_rng(0)
x = np.r_[rng.normal(0, 1, 500), rng.normal(1.5, 1, 500)]
yy = np.r_[np.zeros(500), np.ones(500)]
copies = [1, 2, 3, 4, 5]
pp = []
for k in copies:
    g = GaussianNB().fit(np.tile(x[:, None], (1, k)), yy)
    pp.append(g.predict_proba(np.full((1, k), 1.2))[0, 1])
print("P(class 1 | x=1.2) with copies", dict(zip(copies, np.round(pp, 4))))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(copies, pp, color=WINE, marker="o")
for k, v in zip(copies, pp):
    if k == 1:
        ax.text(k + 0.08, v - 0.035, f"{v:.3f}", ha="left", fontsize=10.5)
    else:
        ax.text(k - 0.08, v + 0.012, f"{v:.3f}", ha="right", fontsize=10.5)
ax.axhline(pp[0], color=TEAL, ls="--", lw=1.3)
ax.text(6.25, pp[0] - 0.02, "same information, so it should stay here", color=TEAL, fontsize=9.5, va="top", ha="right")
ax.set_xticks(copies)
ax.set_xlabel("number of identical copies of the feature")
ax.set_ylabel("P(class 1 | x = 1.2)")
ax.set_ylim(0.6, 1.04); ax.set_xlim(0.7, 6.3)
ax.set_title("GaussianNB counts each copy as new evidence and grows overconfident")
save(fig, "naivebayes-overconfident")
