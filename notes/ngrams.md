# N-grams

Using short runs of neighbouring words as features, so phrases like "governing law" and "not exceed" become columns of their own.

## Definition

An **n-gram** is a run of $n$ words that sit next to each other in the text.

| n | name | example from S1 |
|---|---|---|
| 1 | **unigram** | `pay` |
| 2 | **bigram** | `pay the` |
| 3 | **trigram** | `party shall pay` |

[Bag of Words](#/t/bow) with unigrams counts single words. Adding bigrams and trigrams adds columns for word *pairs* and *triples*, which brings back a little of the word order that BoW throws away.

## Worked example: sliding a window

Take S1 after cleaning and lemmatising: `the party shall pay the fee` (6 tokens). Slide a window of width $n$ along it, one word at a time.

![S1 cut into unigrams (6), bigrams (5) and trigrams (4). Each box is one n-gram; a sentence of 6 tokens gives 6 − n + 1 of them.](figures/ngrams-window.svg)

A document with $T$ tokens has $T - n + 1$ n-grams of size $n$: here $6 - 2 + 1 = 5$ bigrams and $6 - 3 + 1 = 4$ trigrams. In the figure, each bigram overlaps the next by one word.

All three sentences (bigrams only, no stop words removed, listed in sentence order):

| sentence | bigrams |
|---|---|
| S1 | the party · party shall · shall pay · pay the · the fee |
| S2 | the party · party may · may terminate · terminate this · this agreement |
| S3 | notice shall · shall be · be give · give to · to the · the party · party in · in write |

Only `the party` is shared by more than one sentence. Most n-grams are rare, which is both their value (specific) and their cost (sparse).

## Why they help

Some meaning lives in the phrase, not the words.

**1. Fixed phrases.** *governing law*, *force majeure*, *new york*, *in accordance with*. Run on *"this agreement shall be govern by the law of the state of new york"* with his stop words, `ngram_range=(1, 2)` produces `govern law`, `law state`, `new york`. `new york` as one feature is far more useful than `new` and `york` separately.

**2. Negation.** Two clauses, his stop-word list (which keeps *not*):

- A: `fee shall not exceed moneytoken`
- B: `fee shall exceed moneytoken`

| features | A | B | can the model tell them apart? |
|---|---|---|---|
| unigrams: exceed, fee, moneytoken, not | 1,1,1,1 | 1,1,1,0 | only through the lone word *not* |
| + bigrams: **not exceed**, **fee exceed**, fee not, … | `not exceed` = 1 | `fee exceed` = 1 | yes: the phrase itself differs |

With unigrams, *not* is one weight shared by every negated clause. With bigrams, *not exceed* gets its own weight.

## The feature explosion

Every extra $n$ adds a new batch of mostly unique columns.

![Left: vocabulary size of the three sentences by n. Gold = only n-grams of that size; teal = all sizes from 1 to n (what ngram_range=(1, n) builds). Right: his LEDGAR TF-IDF at full scale.](figures/ngrams-explosion.svg)

In the left panel, read the teal bars: 15 columns with unigrams, 31 with `(1, 2)`, 46 with `(1, 3)`, 67 with `(1, 5)`, all from 19 tokens. Almost every new n-gram appears in exactly one sentence. On LEDGAR, the README reports 95,559 columns for `ngram_range=(1, 3)`, *after* `min_df=4` removed the rarest ones.

| why it explodes | consequence |
|---|---|
| a vocabulary of $V$ words allows up to $V^2$ bigrams and $V^3$ trigrams | many more columns than unigrams |
| most combinations occur once | very sparse; easy to overfit on rare phrases |
| memory and training time grow with columns | slower fits |

This is why `ngram_range` almost always comes with **`min_df`** (drop n-grams in fewer than $k$ documents) or **`max_features`**. See [TF-IDF parameters](#/t/tfidf).

## `ngram_range` in sklearn

`CountVectorizer` and `TfidfVectorizer` take `ngram_range=(min_n, max_n)`:

| setting | columns built |
|---|---|
| `(1, 1)` (default) | unigrams only |
| `(1, 2)` | unigrams + bigrams (his notes' first setting) |
| `(1, 3)` | unigrams + bigrams + trigrams (his final `tf_idf.py`) |
| `(2, 2)` | bigrams only |

```python
from sklearn.feature_extraction.text import CountVectorizer

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

for rng in [(1, 1), (1, 2), (1, 3)]:
    cv = CountVectorizer(ngram_range=rng,
                         token_pattern=r"(?u)\b\w+\b")   # allow 1-letter tokens, as in the figure
    cv.fit(lemmas)                                       # vocabulary of all n-grams in range
    print(rng, len(cv.get_feature_names_out()))
# (1, 1) 15
# (1, 2) 31
# (1, 3) 46

cv = CountVectorizer(ngram_range=(2, 2)).fit(lemmas[:1])
print(cv.get_feature_names_out())
# ['party shall' 'pay the' 'shall pay' 'the fee' 'the party']
```

## Caveats

- **Stop words are removed before n-grams are formed.** This makes "skip" n-grams: with his list, S3 `notice shall be give to the party in write` becomes `notice write`, so the bigram `notice write` is created even though the words weren't neighbours. Usually helpful (it keeps the content words together), but be aware it isn't a true phrase.
- **Sparsity and overfitting.** A trigram seen in 3 training clauses can get a large weight. Use `min_df`, regularisation (smaller `C`), and check train vs validation.
- **Correlated features.** `governing`, `law` and `governing law` usually appear together, so their weights share the credit. Fine for prediction, harder to interpret.
- **Bigger n rarely pays off.** For classic models, `(1, 2)` or `(1, 3)` is the usual limit; beyond that the columns are almost all unique.

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Captures phrases and negation (`not exceed`) | Vocabulary grows very fast |
| Still simple: same BoW/TF-IDF machinery | Most n-grams are rare: sparse, noisy |
| Often a solid accuracy gain for text classification | Needs `min_df` / `max_features` to stay manageable |
| Features remain readable | Only local order (n words); long-range meaning still lost |

## Quick reference

| term | one line |
|---|---|
| n-gram | run of $n$ neighbouring tokens |
| unigram / bigram / trigram | 1 / 2 / 3 words |
| count per document | $T - n + 1$ for $T$ tokens |
| `ngram_range=(1, 3)` | all 1-, 2- and 3-grams |
| Explosion | 15 → 31 → 46 columns for the 3 sentences; 95,559 on LEDGAR |
| Control it with | `min_df`, `max_df`, `max_features` |
| Stop words first | n-grams form over what remains (`notice write`) |
