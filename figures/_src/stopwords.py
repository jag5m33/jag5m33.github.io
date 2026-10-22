import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
from nlp_common import *
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# document frequency bars
b = CountVectorizer(binary=True); Xb = b.fit_transform(LEM)
df = np.asarray(Xb.mean(axis=0)).ravel(); words = b.get_feature_names_out()
order = np.argsort(-df, kind="stable"); sw = custom_stop_words()
fig, ax = plt.subplots(figsize=(10, 3.6))
cols = []
for w in words[order]:
    cols.append(WINE if w in LEGAL else (MUTED if w in ENGLISH_STOP_WORDS else TEAL))
ax.bar(range(len(order)), df[order]*100, color=cols)
ax.set_xticks(range(len(order))); ax.set_xticklabels(words[order], rotation=50, ha="right")
ax.set_ylabel("% of sentences\ncontaining the word"); ax.set_ylim(0, 115)
ax.axhline(90, color=INK, ls="--", lw=1); ax.text(2.6, 104, "max_df = 0.9 drops words above the dashed line", fontsize=9.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=MUTED, label="in sklearn's list"), Patch(color=WINE, label="added legal word"),
                   Patch(color=TEAL, label="kept (content word)")], loc="upper right", bbox_to_anchor=(1, 0.78), fontsize=9.5)
ax.set_title("Document frequency for the three lemmatised sentences")
save(fig, "stopwords-df")

# heatmaps: before / sklearn / custom
def dtm(docs, stop=None):
    v = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop); X = v.fit_transform(docs).toarray()
    return X, list(v.get_feature_names_out())
X0, c0 = dtm(LEM); X1, c1 = dtm(LEM, list(ENGLISH_STOP_WORDS)); X2, c2 = dtm(LEM, list(sw))
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[len(c1), len(c2)+2], height_ratios=[1, 1], hspace=0.95, wspace=0.3)
a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])
heatmap(a0, X0, c0, highlight=[c0.index(w) for w in c0 if w in ENGLISH_STOP_WORDS], hl_colour=MUTED,
        title=f"Lemmatised: {len(c0)} columns (grey labels are in sklearn's stop-word list)")
heatmap(a1, X1, c1, highlight=[c1.index(w) for w in ["party", "shall", "agreement"]],
        title=f"sklearn list removed: {len(c1)} columns")
heatmap(a2, X2, c2, cmap=CMAP_COUNT, title=f"+ legal words: {len(c2)} columns")
save(fig, "stopwords-heatmap")
