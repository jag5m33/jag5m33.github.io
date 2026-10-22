# TF-IDF & its parameters

Bag of Words, but rare words count more and common words count less. Then every setting in his TfidfVectorizer, explained.

## The idea

[Bag of Words](#/t/bow) gives the biggest numbers to the most common words, which are the least useful. **TF-IDF** (term frequency × inverse document frequency) multiplies each count by a weight that is **high for rare words and low for common ones**.

- **TF** (term frequency) = how often the word is in **this** document.
- **IDF** (inverse document frequency) = how rare the word is **across all** documents.
- **TF-IDF = TF × IDF.** High only if the word is frequent here *and* rare elsewhere.

## The formulas

$$
\text{tf}(t, d) = \text{count of term } t \text{ in document } d
$$

$$
\text{idf}(t) = \ln\frac{N}{\text{df}(t)}
$$

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
$$

$N$ = number of documents; $\text{df}(t)$ = number of documents containing $t$ (see [Stop words](#/t/stopwords)).

**Where IDF comes from:** $N / \text{df}$ is "one in how many documents has this word". A word in 1 of 1,000 documents gives 1,000; a word in every document gives 1. Taking the **log** squashes that huge range so a very rare word doesn't get an absurd weight (see [Logs & exponents](#/t/logs)). And $\ln 1 = 0$: a word in every document gets weight zero.

## Worked by hand: three tables

Input: the three lemmatised sentences with sklearn's stop words removed (so *party* and *shall* are still there, to show what IDF does to them). $N = 3$.

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

In the bottom panel, the `party` column has gone blank: it's in every sentence, so $\ln(3/3) = 0$ wipes it out. *shall* (in 2 of 3) is pale. The dark cells are the words that make each sentence different.

> **Say it like this:** "party" scores 0 because it's in every sentence, so it can't tell them apart. TF-IDF found the stop word for me.

## sklearn's version: smoothed IDF

`TfidfVectorizer` with its default `smooth_idf=True` uses

$$
\text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1
$$

| word | df | textbook $\ln(N/\text{df})$ | sklearn $\ln\frac{1+N}{1+\text{df}} + 1$ |
|---|---|---|---|
| party | 3 | 0.000 | $\ln(4/4) + 1 =$ **1.000** |
| shall | 2 | 0.405 | $\ln(4/3) + 1 =$ **1.288** |
| fee, pay, … | 1 | 1.099 | $\ln(4/2) + 1 =$ **1.693** |

Why it differs:

- **The $+1$ inside** (on top and bottom) acts as if there were one extra document containing every word. It stops division by zero if a word has $\text{df} = 0$ (possible when you pass a fixed `vocabulary`), and slightly softens extreme weights.
- **The $+1$ outside** means a word in every document gets weight 1, not 0. sklearn chose not to delete such words completely; you remove them with stop words or `max_df` instead.
- The **ranking is the same**: rarer still means higher.

![Left: IDF against document frequency for N = 10. Textbook (wine) falls to 0; sklearn's smoothed version (teal) falls to 1. Right: raw counts vs 1 + ln(count) for sublinear_tf.](figures/tfidf-curves.svg)

In the left panel both curves fall as df rises; only the floor differs.

## BoW vs TF-IDF on the three sentences

sklearn `TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)` with all other defaults (smoothed IDF, then L2 normalisation, below):

![Left: BoW counts, where every present word is 1. Right: TF-IDF, where fee, pay, terminate, notice and write are darkest (0.58 to 0.65), shall is middling (0.44) and party is lightest (0.35 to 0.39).](figures/tfidf-vs-bow.svg)

In BoW every present word gets the same 1. In TF-IDF, compare the `party` column (pale, 0.35) with `fee` (dark, 0.58) in the same row: same count, very different weight.

| S1 | fee | party | pay | shall |
|---|---|---|---|---|
| raw TF × smoothed IDF | 1.693 | 1.000 | 1.693 | 1.288 |
| ÷ length 2.897 (L2) | **0.584** | **0.345** | **0.584** | **0.445** |

S2 has only three words, so each gets a slightly bigger share after L2: 0.652, 0.385, 0.652.

## L2 normalisation and cosine similarity

### L2 normalisation

Treat each row as an **arrow** (a vector). Its **length** (L2 norm) is

$$
\lVert x \rVert_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
$$

Divide every value by the length, and the arrow has length 1.

**From my notes:** $(3, 4) \to \sqrt{9 + 16} = 5 \to (3/5, 4/5) = (0.6, 0.8)$.

**S1:** $\sqrt{1.693^2 + 1^2 + 1.693^2 + 1.288^2} = \sqrt{8.391} = 2.897$, then $1.693 / 2.897 = 0.584$, $1 / 2.897 = 0.345$, $1.288 / 2.897 = 0.445$. These are exactly the numbers in the heatmap.

**Why:** a 300-word clause has bigger counts than a 30-word clause on the same topic. Without normalising, length alone would push it far from its short twin. After L2, only the **direction** (the mix of words) matters.

### Cosine similarity

$$
\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert}
$$

$a \cdot b$ (the dot product) = multiply matching positions and add. After L2 normalisation both lengths are 1, so cosine **is** just the dot product.

**S1 vs S3** share *party* and *shall*:

$$
0.345 \times 0.345 + 0.445 \times 0.445 = 0.119 + 0.198 = 0.317
$$

![Left: the (3, 4) arrow shrunk to the unit arrow (0.6, 0.8). Right: cosine similarities between the TF-IDF rows; S1 and S3 are most alike (0.317), S2 is 0.133 from both.](figures/tfidf-cosine.svg)

| pair | shared words | cosine, BoW | cosine, TF-IDF |
|---|---|---|---|
| S1–S3 | party, shall | 0.500 | 0.317 |
| S1–S2 | party | 0.289 | 0.133 |
| S2–S3 | party | 0.289 | 0.133 |

TF-IDF lowers every similarity because the words they share are the common, down-weighted ones. **0** = no shared words, **1** = the same words in the same proportions.

## Every parameter in his TfidfVectorizer

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

### stop_words = his custom list

- **Does:** drops these tokens before counting (and before forming n-grams).
- **Why:** removes words in nearly every clause (*the, shall, party*) while keeping negations and clause-name words. See [Stop words](#/t/stopwords).
- **Without it:** thousands of extra n-grams like `the party shall`; the matrix is bigger and noisier. IDF down-weights them but doesn't remove them.
- **Note:** the notebook passes `list(stop_words_set)`, which is required: current sklearn accepts only a list, `'english'` or `None`, and raises `InvalidParameterError` for a set.

### lowercase = True

- **Does:** lowercases text before tokenising. It's the default.
- **Why:** `Party` and `party` become one column (18 → 16 columns on the raw sentences; see [Tokenise](#/t/tokenise)).
- **Here:** his `clean_text` already lowercased, so it's a harmless safety net.

### ngram_range = (1, 3)

- **Does:** builds unigrams, bigrams and trigrams.
- **Why:** captures phrases like *governing law* and *not exceed*. See [N-grams](#/t/ngrams).
- **Without it** (default `(1, 1)`): no phrases. **Too large:** the vocabulary explodes, which is why `min_df` is next.

### min_df = 4

- **Does:** drops any term that appears in **fewer than 4 documents**. An integer means a count of documents; a float like `0.01` means a proportion.
- **Why:** removes typos, one-off names and the long tail of rare trigrams. A feature seen in 1–3 clauses of 60,000 can't be learned reliably, and invites overfitting.
- **Without it:** with `(1, 3)` the vocabulary would be several times larger, mostly noise.
- **On the 3 sentences:** `min_df=4` fails with *"max_df corresponds to < documents than min_df"*: no word can appear in 4 of 3 documents. Settings must suit the corpus size.

### max_df = 0.9

- **Does:** drops any term that appears in **more than 90% of documents**. A float is a proportion; an integer would be a count.
- **Why:** an automatic safety net for corpus-specific stop words the hand list missed.
- **On the 3 sentences:** `max_df=0.9` removes exactly *party* and *the* (in 3/3 = 100%), the same words the df chart flagged.

### sublinear_tf = True

- **Does:** replaces the raw count with $1 + \ln(\text{tf})$ (for tf > 0).
- **Why, in his own docstring:** it *"prevents a word repeated 10 times from carrying 10 times the statistical weight"* and *"implements diminishing returns"*.
- **Why the +1 (his explanation):** $\ln 1 = 0$, so without the $+1$ *"any word appearing exactly once in a document would get a term frequency (TF) score of 0"* and be deleted from the dataset. With it, a single appearance keeps a baseline weight of 1.

| count | raw TF | $1 + \ln(\text{count})$ (sklearn, natural log) | $1 + \log_{10}(\text{count})$ (his docstring example) |
|---|---|---|---|
| 1 | 1 | 1.000 | 1.0 |
| 2 | 2 | 1.693 | 1.301 |
| 10 | 10 | 3.303 | 2.0 |
| 100 | 100 | 5.605 | 3.0 |

His docstring uses base 10 to make the numbers round, and notes sklearn uses the natural log; the shape is the same. In S1, *the* (count 2) would go from 2 to 1.693. In the right panel of the curves figure above, the teal line flattens while the raw count keeps climbing.

- **Without it:** a long clause repeating *indemnify* 15 times dominates its vector, even after L2.

### norm = 'l2'

- **Does:** divides each row by its L2 length, so every clause vector has length 1 (the default).
- **Why:** long and short clauses become comparable; cosine similarity is then just a dot product; logistic regression trains more stably on rows of similar size.
- **Without it** (`norm=None`): S1's values would be 1.693, 1, 1.693, 1.288 instead of 0.584, 0.345, 0.584, 0.445, and long clauses would have far larger values.

### Defaults he didn't set (but should know)

| parameter | default | what it does | when to change |
|---|---|---|---|
| `smooth_idf` | True | uses $\ln\frac{1+N}{1+\text{df}} + 1$ | False gives $\ln\frac{N}{\text{df}} + 1$ (fee would be 2.099, party 1) |
| `use_idf` | True | multiply by IDF at all | False = normalised term frequencies only (S1 becomes 0.5 for each of its 4 words) |
| `max_features` | None | keep only the top-$k$ terms by total frequency | cap memory, e.g. `max_features=50000` |
| `token_pattern` | `(?u)\b\w\w+\b` | regex for a token: 2+ word characters | `r"(?u)\b\w+\b"` to keep 1-letter tokens |
| `binary` | False | presence 0/1 instead of counts | short texts |
| `vocabulary` | None | use a fixed vocabulary instead of learning one | compare models on the same columns |

> **Trap:** `max_features` keeps the most *frequent* terms, which are often the least informative. On the three sentences, `max_features=5` keeps *the, party, shall, agreement, give*: the stop words win. Prefer `min_df` / `max_df` and a good stop-word list.

## Fit on train only

`fit` learns the **vocabulary** and the **IDF** values from the documents it sees. Those are learned parameters, just like a scaler's mean (see [Scaling & encoding](#/t/scaling)).

```python
vectorizer = tfid(list(stop_words_set))

x_train = vectorizer.fit_transform(proc_train.lemmatized_text)   # learn vocabulary + IDF from TRAIN
x_val   = vectorizer.transform(proc_val.lemmatized_text)         # reuse them: no leakage
x_test  = vectorizer.transform(proc_test.lemmatized_text)

print(x_train.shape)    # (59933, 95559)
```

If the vectoriser saw validation or test text, their rare words would get vocabulary columns and IDF values, and validation scores would look better than they really are: **data leakage**. A new word at transform time is simply ignored: `"the party shall arbitrate the dispute"` keeps only *party* (0.613) and *shall* (0.790).

## Code: the three sentences end to end

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

tv = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))   # smoothed IDF + L2 by default
X = tv.fit_transform(lemmas)                                 # sparse 3 x 8

print(dict(zip(tv.get_feature_names_out(), tv.idf_.round(3))))
# {'agreement': 1.693, 'fee': 1.693, ..., 'party': 1.0, 'shall': 1.288, ...}

print(pd.DataFrame(X.toarray(), columns=tv.get_feature_names_out(),
                   index=["S1", "S2", "S3"]).round(3))
#     agreement    fee  notice  party    pay  shall  terminate  write
# S1      0.000  0.584   0.000  0.345  0.584  0.445      0.000  0.000
# ...

print(cosine_similarity(X).round(3))     # S1-S3 = 0.317
```

## BoW vs TF-IDF

| | Bag of Words | TF-IDF |
|---|---|---|
| Cell value | raw count | count × rarity weight (then L2) |
| Common words | biggest numbers | down-weighted (IDF ≈ its minimum) |
| Rare, specific words | same as any other | boosted |
| Length effect | long docs have bigger counts | removed by L2 |
| Learns from data | vocabulary | vocabulary **and** IDF |
| His use | finding stop words | the model's input |
| Good with | Naive Bayes (expects counts) | logistic regression, linear SVM, similarity search |

## Assumptions & caveats

- **Rare = important.** Usually true, but typos and one-off names are rare too; `min_df` handles them.
- **Still a bag:** word order (beyond n-grams), synonyms and context are lost, as in BoW.
- **IDF depends on the corpus.** Weights learned on contracts don't transfer to tweets. Refit on new data; never fit on test.
- **Sparse and high-dimensional:** use models that handle sparse input (linear models), and regularise.
- **Settings depend on corpus size:** `min_df=4` is sensible for 60,000 clauses and impossible for 3.

## Use cases

- Strong, fast baseline for text classification (clause type, spam, topic).
- Search and similarity: rank documents by cosine similarity to a query.
- Keyword extraction: the top TF-IDF terms of a document (his waiver example: *presentment, protest, maturity, waive presentment*).

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Down-weights common words automatically | No word order or meaning (synonyms unrelated) |
| Simple, fast, scales to large corpora (sparse) | Very many columns with n-grams |
| Readable features: you can list the top words per class | Several parameters to tune (`min_df`, `max_df`, n-grams) |
| Works very well with linear models | Usually beaten by transformer embeddings on meaning-heavy tasks |

## Quick reference

| term | one line |
|---|---|
| TF | count of the word in this document |
| IDF (textbook) | $\ln(N/\text{df})$; 0 for a word in every document |
| IDF (sklearn) | $\ln\frac{1+N}{1+\text{df}} + 1$; minimum 1 |
| TF-IDF | TF × IDF |
| `sublinear_tf=True` | TF becomes $1 + \ln(\text{tf})$ |
| `min_df=4` | drop terms in fewer than 4 documents |
| `max_df=0.9` | drop terms in more than 90% of documents |
| `ngram_range=(1,3)` | uni-, bi- and trigrams |
| `norm='l2'` | every row has length 1 |
| Cosine similarity | dot product of L2-normalised rows; 0 to 1 here |
| Fit on train only | vocabulary and IDF are learned parameters |
