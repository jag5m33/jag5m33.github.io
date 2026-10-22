import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
from nlp_common import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
SW = list(ENGLISH_STOP_WORDS)
cv = CountVectorizer(stop_words=SW); C = cv.fit_transform(LEM).toarray(); cols = list(cv.get_feature_names_out())
N = 3; df = (C > 0).sum(0)
idf_raw = np.log(N/df)
T_hand = C*idf_raw
tv = TfidfVectorizer(stop_words=SW); T = tv.fit_transform(LEM).toarray()

# 1. three tables as heatmaps: TF, IDF, TF-IDF by hand
fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 3], hspace=1.0)
a = fig.add_subplot(gs[0]); heatmap(a, C, cols, title="1. TF = raw count of the word in the sentence")
a = fig.add_subplot(gs[1]); heatmap(a, idf_raw[None, :], cols, rows=["IDF"], cmap=CMAP_W, fmt="{:.3f}", vmax=1.2,
                                     highlight=[cols.index("party")], title="2. IDF = ln(N / df), N = 3    (party: ln(3/3) = 0)")
a = fig.add_subplot(gs[2]); heatmap(a, T_hand, cols, cmap=CMAP_W, fmt="{:.3f}", vmax=1.2, highlight=[cols.index("party")],
                                     title="3. TF x IDF   ('party' column is all zeros)")
save(fig, "tfidf-hand")

# 2. BoW vs sklearn TF-IDF
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(wspace=0.12))
heatmap(axes[0], C, cols, title="Bag of Words (counts)")
heatmap(axes[1], T, cols, cmap=CMAP_W, fmt="{:.2f}", vmax=0.8, title="TF-IDF (sklearn defaults: smoothed IDF, L2 norm)")
axes[1].set_yticklabels([])
save(fig, "tfidf-vs-bow")

# 3. idf curves
dfr = np.arange(1, 11); Nn = 10
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
ax.plot(dfr, np.log(Nn/dfr), marker="o", color=WINE, label="textbook: ln(N/df)")
ax.plot(dfr, np.log((1+Nn)/(1+dfr)) + 1, marker="o", color=TEAL, label="sklearn: ln((1+N)/(1+df)) + 1")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("df = number of documents containing the word (N = 10)"); ax.set_ylabel("IDF")
ax.legend(fontsize=9.5); ax.set_title("IDF: rare words weigh more")
ax.annotate("in every document:\ntextbook gives 0,\nsklearn gives 1", (10, 0.05), (6.4, 1.6), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax = axes[1]
c = np.arange(1, 21)
ax.plot(c, c, color=WINE, label="raw count (sublinear_tf=False)")
ax.plot(c, 1 + np.log(c), color=TEAL, label="1 + ln(count) (sublinear_tf=True)")
ax.set_xlabel("times the word appears in one clause"); ax.set_ylabel("TF used")
for k, off in [(1, (2.2, 11.5)), (10, (12, 8.5)), (20, (16.2, 6.8))]:
    ax.annotate(f"{k} -> {1+np.log(k):.2f}", (k, 1+np.log(k)), off, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=MUTED))
ax.legend(fontsize=9.5, loc="upper left"); ax.set_title("sublinear_tf: diminishing returns")
fig.tight_layout(); save(fig, "tfidf-curves")

# 4. L2 / cosine: arrows
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[1, 1]))
ax = axes[0]
ax.annotate("", (3, 4), (0, 0), arrowprops=dict(arrowstyle="-|>", color=WINE, lw=2))
ax.annotate("", (0.6, 0.8), (0, 0), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.4))
th = np.linspace(0, np.pi/2, 100); ax.plot(np.cos(th), np.sin(th), color=MUTED, ls="--", lw=1)
ax.text(3.05, 4.05, "(3, 4)\nlength 5", fontsize=10, color=WINE)
ax.text(0.75, 0.55, "(0.6, 0.8)\nlength 1", fontsize=10, color=TEAL)
ax.set_xlim(0, 4.6); ax.set_ylim(0, 4.8); ax.set_aspect("equal")
ax.set_xlabel("count of word A"); ax.set_ylabel("count of word B"); ax.set_title("L2: divide by length -> unit arrow")
ax = axes[1]
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(T)
heatmap(ax, sim, IDX, rows=IDX, cmap=CMAP_W, fmt="{:.3f}", vmax=1.0, title="Cosine similarity of the TF-IDF rows")
ax.set_xticklabels(IDX, rotation=0, ha="center"); ax.set_aspect("equal")
fig.tight_layout(); save(fig, "tfidf-cosine")
