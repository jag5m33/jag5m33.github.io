# TF-IDF & its parameters

Bag of Words, but rare words count more and common words count less. Then each setting in his TfidfVectorizer, in plain words.

## The idea

[Bag of Words](#/t/bow) gives the biggest numbers to the most common words, which are the least useful. **TF-IDF** multiplies each count by a weight that is **high for rare words and low for common ones**.

- **TF** (term frequency) = how often the word is in **this** document.
- **IDF** (inverse document frequency) = how rare the word is **across all** documents.
- **TF-IDF = TF × IDF.** High only if the word is frequent here *and* rare elsewhere.

## The formula

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \times \ln\frac{N}{\text{df}(t)}
$$

- $t$ = a word (term); $d$ = a document.
- $\text{tf}(t, d)$ = how many times $t$ appears in $d$.
- $N$ = number of documents.
- $\text{df}(t)$ = number of documents that contain $t$ (see [Stop words](#/t/stopwords)).
- $\ln(N/\text{df})$ is the IDF. $N/\text{df}$ means "one in how many documents has this word"; the log just squashes big values (see [Logs & exponents](#/t/logs)). A word in every document gets $\ln 1 = 0$.

## Worked by hand: three tables

The three lemmatised sentences with sklearn's stop words removed (so *party* and *shall* are still there, to see what IDF does to them). $N = 3$.

### Table 1: TF (counts)

| | agreement | fee | notice | party | pay | shall | terminate | write |
|---|---|---|---|---|---|---|---|---|
| S1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| S2 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| S3 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |

### Table 2: IDF

| word | df | $N/\text{df}$ | $\ln(N/\text{df})$ |
|---|---|---|---|
| party | 3 | 3/3 = 1 | **0.000** |
| shall | 2 | 3/2 = 1.5 | **0.405** |
| agreement, fee, notice, pay, terminate, write | 1 | 3/1 = 3 | **1.099** |

### Table 3: TF × IDF

| | agreement | fee | notice | party | pay | shall | terminate | write |
|---|---|---|---|---|---|---|---|---|
| S1 | 0 | 1.099 | 0 | **0** | 1.099 | 0.405 | 0 | 0 |
| S2 | 1.099 | 0 | 0 | **0** | 0 | 0 | 1.099 | 0 |
| S3 | 0 | 0 | 1.099 | **0** | 0 | 0.405 | 0 | 1.099 |

![The three tables as heatmaps. TF counts on top; IDF in the middle, with party at 0; TF × IDF at the bottom, where the party column is all zeros and shall is pale.](figures/tfidf-hand.svg)

In the bottom panel, the `party` column has gone blank: it is in every sentence, so $\ln(3/3) = 0$ wipes it out. *shall* (in 2 of 3) is pale. The dark cells are the words that make each sentence different.

> **Say it like this:** "party" scores 0 because it's in every sentence, so it can't tell them apart. TF-IDF found the stop word for me.

## What sklearn does differently

sklearn adds 1 in a couple of places so that a word in every document gets weight **1 instead of 0** (it isn't deleted completely). The order stays the same: rarer still means higher. On our sentences, sklearn's IDF is party = 1.000, shall = 1.288, the rest = 1.693.

## L2 normalisation

After TF × IDF, sklearn makes every row the same "length". Think of each row as an arrow; its **length** (L2 norm) is the square root of the sum of squares. Divide each value by the length.

**From my notes:** $(3, 4)$ has length $\sqrt{9 + 16} = 5$, so it becomes $(3/5, 4/5) = (0.6, 0.8)$.

**S1** (fee, party, pay, shall = 1.693, 1, 1.693, 1.288): length = 2.897, so fee becomes $1.693 / 2.897 = 0.584$, party $0.345$, shall $0.445$.

**Why:** a long clause has bigger counts than a short clause on the same topic. After L2, only the *mix* of words matters, not the length.

![Left: BoW counts, where every present word is 1. Right: TF-IDF, where fee, pay, terminate, notice and write are darkest (0.58 to 0.65), shall is middling (0.44) and party is lightest (0.35 to 0.39).](figures/tfidf-vs-bow.svg)

In BoW every present word is 1. In TF-IDF, compare `party` (pale, 0.345) with `fee` (dark, 0.584) in the same row: same count, very different weight.

**Cosine similarity** measures how alike two rows are, from 0 (no shared words) to 1 (same words in the same mix).

![Left: the (3, 4) arrow shrunk to the unit arrow (0.6, 0.8). Right: cosine similarities between the TF-IDF rows; S1 and S3 are most alike (0.317), S2 is 0.133 from both.](figures/tfidf-cosine.svg)

S1 and S3 are most alike (0.317) because they share *party* and *shall*.

## Each parameter in his TfidfVectorizer

From [py_legal_data/src/tf_idf.py](https://github.com/jag5m33/py_legal_data/blob/main/src/tf_idf.py):

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfid(stop_words_set):
    vectorizer = TfidfVectorizer(stop_words= (stop_words_set),
                                 lowercase = True,
                                 ngram_range = (1,3),
                                 min_df = 4,
                                 sublinear_tf = True,
                                 max_df = 0.9,
                                 norm = 'l2')
    return vectorizer
```

| parameter | what it does, in plain words |
|---|---|
| `stop_words` | Removes his custom list (*the, shall, party*…) but keeps negations. Must be passed as a **list**, not a set. See [Stop words](#/t/stopwords). |
| `lowercase=True` | Lowercases everything so `Party` and `party` are one column. His cleaning already did this, so it's a safety net. |
| `ngram_range=(1,3)` | Uses single words, pairs and triples, to catch phrases like *governing law*. See [N-grams](#/t/ngrams). |
| `min_df=4` | Drops any term in fewer than 4 documents: typos, one-off names, rare trigrams. |
| `max_df=0.9` | Drops any term in more than 90% of documents: common words his list missed. |
| `sublinear_tf=True` | Uses $1 + \ln(\text{count})$ instead of the raw count, so a word repeated 10 times doesn't count 10 times as much. |
| `norm='l2'` | Gives every row length 1 (above), so long and short clauses are comparable. |

**`sublinear_tf` in numbers:** count 1 → 1, count 2 → 1.693, count 10 → 3.303. In his own words, it *"implements diminishing returns"*, and the $+1$ is there because $\ln 1 = 0$: without it, *"any word appearing exactly once in a document would get a term frequency (TF) score of 0"*.

> **Trap:** these settings depend on corpus size. `min_df=4` is sensible for 60,000 clauses but impossible on 3 sentences (no word can be in 4 of 3 documents), and sklearn raises an error.

## Fit on train only

`fit` learns the **vocabulary** and the **IDF** values. Those are learned from data, just like a scaler's mean (see [Scaling & encoding](#/t/scaling)).

```python
vectorizer = tfid(list(stop_words_set))

x_train = vectorizer.fit_transform(proc_train.lemmatized_text)  # learn vocabulary + IDF from TRAIN
x_val   = vectorizer.transform(proc_val.lemmatized_text)        # reuse them: no leakage
x_test  = vectorizer.transform(proc_test.lemmatized_text)

print(x_train.shape)    # (59933, 95559)
```

If the vectoriser saw validation or test text, scores would look better than they really are: **data leakage**.

## Code: the three sentences end to end

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

tv = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))   # sklearn IDF + L2 by default
X = tv.fit_transform(lemmas)

print(pd.DataFrame(X.toarray(), columns=tv.get_feature_names_out(),
                   index=["S1", "S2", "S3"]).round(3))
#     agreement    fee  notice  party    pay  shall  terminate  write
# S1      0.000  0.584   0.000  0.345  0.584  0.445      0.000  0.000
```

## BoW vs TF-IDF

| | Bag of Words | TF-IDF |
|---|---|---|
| Cell value | raw count | count × rarity weight, then L2 |
| Common words | biggest numbers | turned down |
| Rare, specific words | same as any other | boosted |
| His use | finding stop words | the model's input |

## Caveats

- **Rare = important** is usually true, but typos are rare too; `min_df` handles them.
- **Still a bag:** word order (beyond n-grams) and meaning are lost.
- **IDF depends on the corpus.** Refit on new data; never fit on test.

## Use cases

- Fast, strong starting model for text classification (clause type, spam, topic).
- Search: find the documents most similar to a query.
- Keywords: the top TF-IDF words of a document (his waiver example: *presentment, protest, maturity, waive presentment*).

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Turns down common words automatically | No word order or meaning |
| Simple, fast, scales to large corpora | Very many columns with n-grams |
| Readable: you can list top words per class | Several settings to tune |
| Works very well with logistic regression | Usually beaten by transformers on meaning-heavy tasks |

## Quick reference

| term | one line |
|---|---|
| TF | count of the word in this document |
| IDF | $\ln(N/\text{df})$; 0 for a word in every document |
| TF-IDF | TF × IDF |
| sklearn's IDF | adds 1s, so the minimum is 1, not 0 |
| L2 norm | every row scaled to length 1 |
| `min_df=4` / `max_df=0.9` | drop very rare / very common terms |
| `sublinear_tf=True` | count becomes $1 + \ln(\text{count})$ |
| Fit on train only | vocabulary and IDF are learned |
