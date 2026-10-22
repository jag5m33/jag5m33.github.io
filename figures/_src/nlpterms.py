import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
from nlp_common import *
from matplotlib.patches import FancyBboxPatch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def box(ax, x, y, w, h, fc, ec=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw))

# 1. corpus -> documents -> tokens -> vocabulary -> DTM
fig, ax = plt.subplots(figsize=(13, 5.6)); ax.axis("off"); ax.set_xlim(0, 26); ax.set_ylim(0, 11.5)
box(ax, 0.2, 0.3, 10.4, 10.6, "none", MUTED, 1.4)
ax.text(5.4, 10.4, "CORPUS = all the documents", ha="center", fontweight="bold", color=MUTED)
toks = [c.split() for c in CLEAN]
for i, (s, tk) in enumerate(zip(S, toks)):
    y = 7.3 - i*3.2
    box(ax, 0.6, y, 9.6, 2.6, PAPER, TEAL)
    ax.text(0.9, y + 2.1, f"DOCUMENT {IDX[i]}", fontsize=9, color=TEAL, fontweight="bold")
    ax.text(0.9, y + 1.45, f"“{s}”", fontsize=9.5, style="italic")
    x = 0.9
    for t in tk:
        w = 0.15*len(t) + 0.28
        box(ax, x, y + 0.25, w, 0.65, LIGHT_GOLD, GOLD, 0.8)
        ax.text(x + w/2, y + 0.57, t, ha="center", va="center", fontsize=8)
        x += w + 0.1
ax.text(10.0, 0.45, "gold = tokens", fontsize=8.5, color=GOLD, ha="right")
ax.annotate("", (11.5, 5.6), (10.75, 5.6), arrowprops=dict(arrowstyle="-|>", lw=1.6, color=MUTED))
ax.text(11.12, 7.4, "unique tokens", ha="center", fontsize=9, color=MUTED, rotation=90)
v = CountVectorizer().fit(CLEAN); vocab = list(v.get_feature_names_out())
box(ax, 11.6, 0.3, 2.6, 10.6, LIGHT_TEAL, TEAL)
ax.text(12.9, 10.4, "VOCABULARY", ha="center", fontsize=9, fontweight="bold", color=TEAL)
for k, w in enumerate(vocab):
    ax.text(12.9, 9.7 - k*0.58, w, ha="center", fontsize=8.5)
ax.text(12.9, 0.55, f"{len(vocab)} words", ha="center", fontsize=8.5, color=TEAL)
ax.annotate("", (15.3, 5.6), (14.35, 5.6), arrowprops=dict(arrowstyle="-|>", lw=1.6, color=MUTED))
ax.text(14.85, 7.2, "each word\n= a\nfeature", ha="center", fontsize=9, color=MUTED)
X = v.transform(CLEAN).toarray()
ins = ax.inset_axes([16.3/26, 3.4/11.5, 9.8/26, 3.4/11.5])
heatmap(ins, X, vocab, fontsize=8.5)
ax.text(20.9, 9.6, "DOCUMENT-TERM MATRIX", ha="center", fontsize=9, fontweight="bold")
ax.text(20.9, 9.0, "rows = documents\ncolumns = vocabulary\ncell = count", ha="center", fontsize=8.5, color=MUTED, va="top")
save(fig, "nlpterms-hierarchy")

# 2. pipeline with S1 at every stage
sw = custom_stop_words()
s1_lem = LEM[0].split()
s1_kept = [w for w in s1_lem if w not in sw]
tv = TfidfVectorizer(stop_words=list(sw)).fit(LEM)
row = tv.transform(LEM[:1]).toarray()[0]
nums = ", ".join(f"{w} {x:.2f}" for w, x in zip(tv.get_feature_names_out(), row) if x > 0)
stages = [("raw text", S[0], MUTED),
          ("clean (regex)", CLEAN[0], BLUE),
          ("tokens", " | ".join(CLEAN[0].split()), GOLD),
          ("lemmas (spaCy)", " | ".join(s1_lem), ORANGE),
          ("stop words removed", " | ".join(s1_kept), WINE),
          ("numbers (TF-IDF)", nums, TEAL),
          ("model", "scores for each clause type -> predicted label", INK)]
fig, ax = plt.subplots(figsize=(10, 5.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, len(stages))
for i, (name, txt, col) in enumerate(stages):
    y = len(stages) - 1 - i
    box(ax, 0.1, y + 0.22, 2.6, 0.56, "none", col, 1.6)
    ax.text(1.4, y + 0.5, name, ha="center", va="center", fontsize=10.5, color=col, fontweight="bold")
    ax.text(3.0, y + 0.5, txt, va="center", fontsize=10.5, family="monospace" if 0 < i < 6 else "serif")
    if i < len(stages) - 1:
        ax.annotate("", (1.4, y - 0.2), (1.4, y + 0.2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
save(fig, "nlpterms-pipeline")
