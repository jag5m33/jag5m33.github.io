import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
from nlp_common import *
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

v = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS)); X = v.fit_transform(LEM).toarray(); cols = list(v.get_feature_names_out())
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[len(cols), 1.2], wspace=0.08))
heatmap(axes[0], X, cols, title="Document-term matrix (BoW counts)")
axes[0].set_yticklabels([f"{i}  " for i in IDX])
tot = X.sum(axis=0)
heatmap(axes[1], X.sum(axis=1, keepdims=True), ["words kept"], cmap=CMAP_W, fmt="{:.0f}", vmax=8)
axes[1].set_yticks([])
save(fig, "bow-dtm")

# one-hot per word vs BoW for S1
s1 = LEM[0].split()
vocab = list(CountVectorizer(token_pattern=r"(?u)\b\w+\b").fit(LEM).get_feature_names_out())
OH = np.array([[1 if w == c else 0 for c in vocab] for w in s1])
fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw=dict(height_ratios=[len(s1), 1.4], hspace=0.25))
heatmap(axes[0], OH, vocab, rows=s1, title="One-hot: one row per WORD of S1 (6 rows x 15 columns)")
axes[0].set_xticklabels([]); axes[0].tick_params(axis="x", length=0)
heatmap(axes[1], OH.sum(axis=0, keepdims=True), vocab, rows=["S1 (BoW)"], cmap=CMAP_COUNT)
axes[1].set_title("Bag of Words: add the rows up -> one row per SENTENCE ('the' = 2)", fontsize=11.5)
save(fig, "bow-onehot")
