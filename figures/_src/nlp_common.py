"""Shared running example for the NLP chapters: Jag's three legal sentences,
cleaned with his clean.py logic and lemmatised with spaCy en_core_web_sm."""
import re, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from figstyle import *

S = ["The Party shall pay the Fees.",
     "The parties may terminate this Agreement.",
     "Notices shall be given to the party in writing."]
IDX = ["S1", "S2", "S3"]

def clean_text(text):          # copy of py_legal_data/src/clean.py
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[$£€]\s?\d[\d,]*(\.\d+)?", " moneytoken ", text)
    text = re.sub(r"\(\s*([a-z]|[ivx]+|\d+)\s*\)", " ", text, flags=re.I)
    text = re.sub(r"\d+([.,]\d+)*", " numtoken ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def lemmatise(texts):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return [" ".join(t.lemma_ for t in doc) for doc in nlp.pipe(texts)]

CLEAN = [clean_text(s) for s in S]
LEM = lemmatise(CLEAN)

EXEMPT = {'no', 'not', 'nor', 'never', 'neither', 'none', 'nothing', 'nobody',
          'without', 'cannot', 'further', 'interest'}
LEGAL = {'shall', 'party', 'agreement', 'hereof', 'hereunder', 'herein'}

def custom_stop_words():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return (set(ENGLISH_STOP_WORDS) | LEGAL) - EXEMPT

CMAP_COUNT = LinearSegmentedColormap.from_list("cnt", [PAPER, LIGHT_TEAL, TEAL])
CMAP_W = LinearSegmentedColormap.from_list("w", [PAPER, LIGHT_GOLD, GOLD, WINE])

def heatmap(ax, M, cols, rows=IDX, cmap=CMAP_COUNT, fmt="{:.0f}", vmax=None, highlight=(), hl_colour=WINE,
            fontsize=10, title=None, zero_grey=True):
    M = np.asarray(M, float)
    vmax = vmax if vmax is not None else max(M.max(), 2)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = v / vmax > 0.6
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color=("white" if dark else (MUTED if (v == 0 and zero_grey) else INK)))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=55, ha="right", fontsize=fontsize)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=fontsize)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=RULE, lw=1.2); ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for j in highlight:
        ax.get_xticklabels()[j].set_color(hl_colour); ax.get_xticklabels()[j].set_fontweight("bold")
    if title: ax.set_title(title, fontsize=11.5)
