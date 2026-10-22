import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
from nlp_common import *
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.patches import FancyBboxPatch

# 1. sliding window on S1
toks = LEM[0].split()
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(-0.3, 10.5); ax.set_ylim(-0.4, 3.6)
for n, y, col, lab in [(1, 2.7, TEAL, "unigrams (n=1)"), (2, 1.5, GOLD, "bigrams (n=2)"), (3, 0.3, WINE, "trigrams (n=3)")]:
    ax.text(-0.2, y + 0.25, lab, fontsize=10.5, color=col, fontweight="bold", va="center")
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    x = 2.4
    for g in grams:
        w = 0.105*len(g) + 0.3
        ax.add_patch(FancyBboxPatch((x, y), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec=col, lw=1.3))
        ax.text(x + w/2, y + 0.25, g, ha="center", va="center", fontsize=9.5)
        x += w + 0.15
    ax.text(10.4, y + 0.25, f"{len(grams)}", ha="right", va="center", fontsize=11, color=col, fontweight="bold")
ax.text(10.4, 3.4, "count", ha="right", fontsize=9, color=MUTED)
ax.set_title("S1 lemmatised: 'the party shall pay the fee' (6 tokens -> 6 - n + 1 n-grams)")
save(fig, "ngrams-window")

# 2. explosion
ns = [1, 2, 3, 4, 5]
exact = [len(CountVectorizer(ngram_range=(n, n), token_pattern=r"(?u)\b\w+\b").fit(LEM).get_feature_names_out()) for n in ns]
cum = [len(CountVectorizer(ngram_range=(1, n), token_pattern=r"(?u)\b\w+\b").fit(LEM).get_feature_names_out()) for n in ns]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
ax.bar(np.array(ns) - 0.2, exact, 0.4, color=GOLD, label="only n-grams of size n")
ax.bar(np.array(ns) + 0.2, cum, 0.4, color=TEAL, label="ngram_range=(1, n)")
for x, a, b in zip(ns, exact, cum):
    ax.text(x - 0.2, a + 1, str(a), ha="center", fontsize=9.5); ax.text(x + 0.2, b + 1, str(b), ha="center", fontsize=9.5)
ax.set_xticks(ns); ax.set_xlabel("n"); ax.set_ylabel("vocabulary size (columns)")
ax.set_title("Three sentences (19 tokens)"); ax.legend(fontsize=9.5, loc="upper left")
# LEDGAR-scale illustration with real numbers from his README
ax = axes[1]
ax.axis("off")
rows = [("LEDGAR train (59,933 clauses)", ""),
        ("ngram_range=(1, 3), min_df=4,", ""),
        ("max_df=0.9, custom stop words", "95,559 columns"),
        ("", ""),
        ("each clause uses about (his notes)", "73 columns"),
        ("share of matrix that is zero", "> 99.9%")]
for i, (a, b) in enumerate(rows):
    ax.text(0.02, 0.9 - i*0.14, a, fontsize=11, transform=ax.transAxes)
    ax.text(0.98, 0.9 - i*0.14, b, fontsize=11, ha="right", fontweight="bold", color=WINE, transform=ax.transAxes)
ax.set_title("Same idea at full size (his project)")
fig.tight_layout(); save(fig, "ngrams-explosion")
