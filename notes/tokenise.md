# Tokenise, stem & lemmatise

Cutting text into tokens, then reducing each token to a base form, so that "parties" and "party" end up in the same column.

## 1. Tokenisation

**Tokenisation** means splitting text into **tokens**, the pieces that become columns. "the parties agreed" → `the` · `parties` · `agreed`.

You don't write this yourself: spaCy and sklearn do it. But each tool cuts text slightly differently.

**Input:** `The Party can't pay U.S. $5,000.`

| tokeniser | tokens |
|---|---|
| spaCy | The · Party · ca · n't · pay · U.S. · \$ · 5,000 · . |
| sklearn default | The · Party · can · pay · 000 |

- spaCy keeps punctuation, knows `U.S.` is one thing, and keeps `n't` (the negation) as its own token.
- sklearn drops punctuation and every **one-letter** token, so `t`, `U`, `S` and `5` vanish.

> **Trap:** sklearn's default silently drops one-character tokens. Usually fine for English, but worth knowing. See [TF-IDF](#/t/tfidf).

## 2. POS tagging

**Part-of-speech (POS) tagging** labels each token with its grammar role: noun, verb, determiner and so on. spaCy does this using the words around it.

| S3 | Notices | shall | be | given | to | the | party | in | writing |
|---|---|---|---|---|---|---|---|---|---|
| POS | NOUN | AUX | AUX | VERB | ADP | DET | NOUN | ADP | VERB |
| lemma | notice | shall | be | **give** | to | the | party | in | **write** |

The lemmatiser (next section) uses these tags.

## 3. Stemming vs lemmatisation

Both reduce a word to a base form, so that different forms count as the same column.

- **Stemming:** chops endings off by fixed rules. Fast, but the result (the **stem**) may not be a real word.
- **Lemmatisation:** looks up the dictionary form (the **lemma**), using the word's POS. Always a real word.

Porter stemmer (NLTK) vs spaCy, word by word:

| word | Porter stem | spaCy lemma |
|---|---|---|
| parties | parti | party |
| terminating | termin | terminate |
| notices | notic | notice |
| given | given | give |
| paid | paid | pay |
| was | wa | be |

- Stems like `parti` and `wa` aren't words, so they're hard to read later.
- Stemming misses irregular words (`given`, `paid`, `was`); the lemmatiser knows them.

| | stemming | lemmatisation |
|---|---|---|
| How | chops endings by rule | looks up the real word |
| Speed | fast | slower |
| Result | sometimes not a real word | always a real word |
| Tool | NLTK `PorterStemmer` | spaCy `token.lemma_` |

**He used lemmatisation (spaCy)** because legal words need to stay readable for the error analysis.

### Why the lemmatiser needs the POS

| sentence | token | POS | lemma |
|---|---|---|---|
| We had a **meeting**. | meeting | NOUN | meeting |
| They are **meeting** the client. | meeting | VERB | meet |

Same spelling, different word. Without the tag, spaCy can't tell which one it is.

> **Trap:** taggers make mistakes. "in writing" in S3 was tagged VERB, giving `write`. Lowercased, short text gives the tagger less to go on.

## His Lemmatizer class

From [py_legal_data/src/lemmetizer.py](https://github.com/jag5m33/py_legal_data/blob/main/src/lemmetizer.py) (comments shortened):

```python
import spacy

class Lemmatizer:
    def __init__(self, model="en_core_web_sm", batch_size=500):
        # load spaCy once; switch off parser and ner (not needed, so faster)
        self.nlp = spacy.load(model, disable=["parser", "ner"])
        self.batch_size = batch_size

    def lemmatize_clauses(self, df):
        results = []
        # nlp.pipe processes the whole column, 500 clauses at a time
        for doc in self.nlp.pipe(df["clean_text"], batch_size=self.batch_size):
            results.append(" ".join(t.lemma_ for t in doc))  # "parties" -> "party"
        df["lemmatized_text"] = results
        return df
```

- `nlp.pipe` runs tokeniser → tagger → lemmatiser on many texts in batches, in the same order as the rows. Much faster than calling `nlp()` 60,000 times.
- The **parser** (grammar tree) and **ner** (finding names of people and companies) are switched off because lemmas don't need them.

## Heatmap: columns merging

The count table for the three sentences at three stages.

![Document-term matrix of the three sentences at three stages. Raw: 18 columns, with The/the and Party/party/parties split. Lowercased: 16 columns. Lemmatised: 15 columns, and party now has a 1 in every row.](figures/tokenise-merge.svg)

Top to bottom:

1. **Raw (18 columns):** `The`/`the` are separate, and so are `Party`, `party`, `parties`. Three columns for one idea.
2. **Lowercased (16):** `The`+`the` merge, and `Party`+`party` merge.
3. **Lemmatised (15):** `parties` → `party`, so `party` now has a 1 in every row. Also `fees` → `fee`, `notices` → `notice`.

> **Key idea:** fewer, fuller columns. Each merge gives the model more examples of the same feature, so it learns it better.

## Code

```python
import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
ps = PorterStemmer()

doc = nlp("the parties may terminate this agreement")   # tokenise, tag, lemmatise
for token in doc:
    print(token.text, token.pos_, token.lemma_, ps.stem(token.text))
# parties NOUN party parti
# terminate VERB terminate termin
```

## Advantages / disadvantages

| | advantages | disadvantages |
|---|---|---|
| Stemming | very fast, no model needed | non-words; misses irregular words |
| Lemmatisation | real words; handles irregulars | slower; tagger mistakes become lemma mistakes |

## Limitations

- Lemmatising doesn't merge **synonyms** (`terminate` / `cancel`).
- Contractions after punctuation removal can go wrong (`won t` → `win t`). LEDGAR had none.

## Quick reference

| term | one line |
|---|---|
| Token | one piece the tokeniser produced |
| POS tag | grammar role: NOUN, VERB… (`token.pos_`) |
| Stem | ending chopped by rule; may not be a word (`parti`) |
| Lemma | dictionary form (`party`), via `token.lemma_` |
| `nlp.pipe(texts, batch_size)` | process many texts in batches, in order |
| `disable=["parser","ner"]` | skip parts you don't need, for speed |
