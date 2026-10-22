import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
from nlp_common import *
from sklearn.feature_extraction.text import CountVectorizer

def dtm(docs, **kw):
    v = CountVectorizer(token_pattern=r"(?u)\b\w+\b", **kw); X = v.fit_transform(docs).toarray()
    return X, list(v.get_feature_names_out())

X0, c0 = dtm(S, lowercase=False)
X1, c1 = dtm(S)
X2, c2 = dtm(LEM)
fig, axes = plt.subplots(3, 1, figsize=(10, 8.6))
heatmap(axes[0], X0, c0, highlight=[c0.index(w) for w in ["The", "the", "Party", "party", "parties"]],
        title=f"1. Raw tokens (case kept): {len(c0)} columns")
heatmap(axes[1], X1, c1, highlight=[c1.index(w) for w in ["the", "party", "parties", "fees", "notices", "given", "writing"]],
        title=f"2. Lowercased: {len(c1)} columns  (The+the merge, Party+party merge)")
heatmap(axes[2], X2, c2, highlight=[c2.index(w) for w in ["party", "fee", "notice", "give", "write"]], hl_colour=TEAL,
        title=f"3. Lemmatised with spaCy: {len(c2)} columns  (parties -> party, so 'party' is now in every row)")
fig.tight_layout(h_pad=1.6); save(fig, "tokenise-merge")
