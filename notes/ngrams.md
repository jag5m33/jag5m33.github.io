# N-grams

Using short runs of neighbouring words as features, so phrases like "governing law" and "not exceed" get columns of their own.

## Definition

An **n-gram** is a run of $n$ words that sit next to each other.

| n | name | example from S1 |
|---|---|---|
| 1 | **unigram** | `pay` |
| 2 | **bigram** | `pay the` |
| 3 | **trigram** | `party shall pay` |

[Bag of Words](#/t/bow) counts single words. Adding bigrams and trigrams adds columns for word *pairs* and *triples*, which brings back a little of the word order BoW throws away.

## Worked example: sliding a window

Take S1 after cleaning and lemmatising: `the party shall pay the fee` (6 tokens). Slide a window of $n$ words along it, one word at a time.

![S1 cut into unigrams (6), bigrams (5) and trigrams (4). Each box is one n-gram; a sentence of 6 tokens gives 6 − n + 1 of them.](figures/ngrams-window.svg)

A sentence of $T$ tokens has $T - n + 1$ n-grams. Here: $6 - 2 + 1 = 5$ bigrams and $6 - 3 + 1 = 4$ trigrams.

Bigrams of all three sentences:

| sentence | bigrams |
|---|---|
| S1 | the party · party shall · shall pay · pay the · the fee |
| S2 | the party · party may · may terminate · terminate this · this agreement |
| S3 | notice shall · shall be · be give · give to · to the · the party · party in · in write |

Only `the party` appears in more than one sentence. Most n-grams are rare.

## Why they help

Some meaning lives in the phrase, not the single words.

**1. Fixed phrases.** *governing law*, *force majeure*, *new york*. `new york` as one column is far more useful than `new` and `york` separately.

**2. Negation.** Two clauses (his stop-word list keeps *not*):

- A: `fee shall not exceed moneytoken`
- B: `fee shall exceed moneytoken`

With single words, A and B differ only by the lone word *not*. With bigrams, A gets its own `not exceed` column, so the model can learn that phrase directly.

## The feature explosion

Every extra $n$ adds a big batch of new, mostly one-off columns.

![Left: vocabulary size of the three sentences by n. Gold = only n-grams of that size; teal = all sizes from 1 to n (what ngram_range=(1, n) builds). Right: his LEDGAR TF-IDF at full scale.](figures/ngrams-explosion.svg)

In the left panel, read the teal bars: 15 columns with single words, 31 with `(1, 2)`, 46 with `(1, 3)`, all from three short sentences. On LEDGAR, `ngram_range=(1, 3)` gave 95,559 columns, even *after* `min_df=4` removed the rarest ones.

- More columns → more memory and slower training.
- Most n-grams appear once → easy to **overfit** (learn rare phrases that don't generalise).

That's why `ngram_range` almost always comes with **`min_df`** (drop n-grams in fewer than $k$ documents). See [TF-IDF](#/t/tfidf).

## `ngram_range` in sklearn

| setting | columns built |
|---|---|
| `(1, 1)` (default) | single words only |
| `(1, 2)` | single words + bigrams |
| `(1, 3)` | single words + bigrams + trigrams (his `tf_idf.py`) |
| `(2, 2)` | bigrams only |

```python
from sklearn.feature_extraction.text import CountVectorizer

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

for rng in [(1, 1), (1, 2), (1, 3)]:
    cv = CountVectorizer(ngram_range=rng,
                         token_pattern=r"(?u)\b\w+\b")  # keep 1-letter words, as in the figure
    cv.fit(lemmas)
    print(rng, len(cv.get_feature_names_out()))         # 15, 31, 46 columns
```

## Caveats

- **Stop words are removed first**, so n-grams can join words that weren't neighbours. With his list, S3 becomes `notice write`, so the bigram `notice write` is made. Usually fine, but it isn't a real phrase.
- **Rare n-grams overfit.** Use `min_df` and check train vs validation scores.
- **Bigger n rarely helps.** `(1, 2)` or `(1, 3)` is the usual limit.

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Captures phrases and negation (`not exceed`) | Number of columns grows very fast |
| Same simple BoW / TF-IDF tools | Most n-grams are rare |
| Often improves text classification | Needs `min_df` to stay manageable |

## Quick reference

| term | one line |
|---|---|
| n-gram | run of $n$ neighbouring words |
| unigram / bigram / trigram | 1 / 2 / 3 words |
| how many per sentence | $T - n + 1$ for $T$ tokens |
| `ngram_range=(1, 3)` | all 1-, 2- and 3-word runs |
| Explosion | 15 → 31 → 46 columns on the 3 sentences; 95,559 on LEDGAR |
| Control it with | `min_df` |
