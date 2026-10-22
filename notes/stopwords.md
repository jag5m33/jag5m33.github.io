# Stop words

Removing the very common words that don't help tell documents apart, and building a list that fits legal text instead of trusting a generic one.

## Definition

**Stop words** are very common words that carry little information for the task, so they are removed before counting: *the, of, and, which, this, to, be…*

They appear in almost every document, so they can't help separate one clause type from another. Leaving them in adds columns (more to learn, more noise) and, in [Bag of Words](#/t/bow), gives them the biggest counts.

> **Key idea:** "stop word" is relative to the task. *party* is meaningful English, but in a corpus of contracts it is in nearly every clause, so it can't tell a Notices clause from a Payments clause.

## Step 1: start with sklearn's list

`sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` is a frozen set of **318** words (checked with `len()`). It includes words you'd expect (*the, a, of, this, be*) and some you might not:

| surprise in the list | why it matters for contracts |
|---|---|
| no, not, nor, never, none, nothing, without, cannot | **negations**: removing them flips meaning |
| further, interest | part of clause names (Further Assurances, Interests) |
| may, must, will, should, would, could | modal verbs: *may* (permission) vs *must* (obligation) can matter legally |
| amount, due, full, first, per, against, within, upon | moderately common in contracts; he left these in the list to test later |
| herein, hereby, whereas, therefore | legal boilerplate, fine to remove |

And it **doesn't** include the words that are everywhere in contracts: *shall, party, agreement, hereof, hereunder* (only *herein* is already there).

## Step 2: find domain stop words with document frequency

**Document frequency (df)** of a word = the share of documents that contain it at least once:

$$
\text{df}(w) = \frac{\text{number of documents containing } w}{N}
$$

It counts *documents*, not occurrences: a word used five times in one clause still counts once. A word with df near 100% is in nearly every clause, so it's a candidate stop word.

**Worked on the three lemmatised sentences** ($N = 3$):

| word | in S1? | S2? | S3? | df |
|---|---|---|---|---|
| party | yes | yes | yes | 3/3 = **100%** |
| the | yes | yes | yes | 3/3 = **100%** |
| shall | yes | | yes | 2/3 = **67%** |
| agreement | | yes | | 1/3 = 33% |
| pay, fee, terminate, notice, write… | one each | | | 33% |

![Document frequency of every word in the three lemmatised sentences. party and the are in all three; shall in two. Grey bars are already in sklearn's list; wine bars are the legal words he added; teal bars are the content words that survive.](figures/stopwords-df.svg)

In the figure, the two tallest bars are *party* (wine, a legal word sklearn doesn't remove) and *the* (grey, sklearn removes it). The teal bars that survive are exactly the words that say what each sentence is *about*: pay, fee, terminate, notice, write.

### Code: document frequency with `CountVectorizer(binary=True)`

`binary=True` puts 1 if the word is present at all, instead of the count. Then the column mean **is** the document frequency.

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

cv = CountVectorizer(binary=True)                 # 1 = word present, 0 = absent
X = cv.fit_transform(lemmas)                      # sparse 3 x 15 matrix of 0/1
df = pd.Series(np.asarray(X.mean(axis=0)).ravel(),   # column mean = share of docs
               index=cv.get_feature_names_out())
print(df.sort_values(ascending=False).head())
# party 1.000, the 1.000, shall 0.667, ...
```

On LEDGAR he did this on the training clauses only, and printed the percentage for each doubtful word.

> **Trap (from his notebook fix list):** `df.text.str.contains("per")` also matches *per*form, *per*iod, pro*per*ty, so it reported 49.8% for "per", far too high. Use whole-word matching, `str.contains(r"\bper\b")`, or the `binary=True` method above, which works on tokens and can't make this mistake.

## Step 3: his custom legal list

From his model notebook ([py_legal_data/notebooks/model.ipynb](https://github.com/jag5m33/py_legal_data/blob/main/notebooks/model.ipynb)):

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words_set = set(ENGLISH_STOP_WORDS)

# KEEP in text (remove them from the stop-word list)
exempt_words = {'no', 'not', 'nor', 'never', 'neither', 'none', 'nothing',
                'nobody', 'without', 'cannot', 'further', 'interest'}

# Common legal words to REMOVE from the text (add them to the stop-word list)
legal_words = {'shall', 'party', 'agreement', 'hereof', 'hereunder', 'herein'}

# Final list: built-in + legal words - exemptions
stop_words_set = (stop_words_set | legal_words) - exempt_words
```

- `|` is set **union** (add), `-` is set **difference** (take away). All 12 exempt words were in sklearn's list; 5 of the 6 legal words were new.
- Size: $318 + 5 - 12 = 311$ words (checked).

| decision | words | why |
|---|---|---|
| **Add** | shall, party, agreement, hereof, hereunder, herein | in most clauses (high df), so they don't separate clause types |
| **Keep: negations** | no, not, nor, never, neither, none, nothing, nobody, without, cannot | they flip meaning; *"No Waivers"* would look like *"Waivers"* |
| **Keep: clause-name words** | further, interest | they name classes: *Further Assurances*, *Interests* |

> **Trap (his notebook bug):** `stop_words_set = stop_words_set.difference_update(exempt)` sets the variable to `None`, because `difference_update` changes the set in place and returns nothing. Use the `(a | b) - c` expression above, which returns a new set.

## The "No Waivers" vs "Waivers" trap

LEDGAR has two labels that differ only by a negation, and his EDA found the same text labelled both ways. Here are two clauses run through `CountVectorizer` with each list:

- A: *no waiver of any breach shall be deemed a waiver of any other breach*
- B: *waiver of any breach shall be in writing*

| list | A becomes | B becomes |
|---|---|---|
| sklearn's (removes *no*) | breach 2, deemed 1, shall 1, waiver 2 | breach 1, shall 1, waiver 1, writing 1 |
| his custom (keeps *no*) | breach 2, deemed 1, **no 1**, waiver 2 | breach 1, waiver 1, writing 1 |

With sklearn's list the word that distinguishes "No Waivers" is gone. With his list *no* survives as a feature (and with bigrams, `no waiver` survives as a phrase; see [N-grams](#/t/ngrams)). His README shows `Waivers → No Waivers` is still one of the most common confusions, so this distinction is genuinely hard even with *no* kept.

## Heatmap: before and after

![Top: lemmatised matrix, 15 columns, with grey labels marking words in sklearn's list. Bottom left: after sklearn's list, 8 columns; party, shall and agreement (wine) survive. Bottom right: after his legal list, 5 columns, each sentence showing only its content words.](figures/stopwords-heatmap.svg)

Read it top to bottom:

1. **15 columns.** The grey-labelled ones (*be, give, in, may, the, this, to*) are in sklearn's list. Note *give* is in the list, which removes a real verb from S3.
2. **sklearn's list → 8 columns.** *party* is still in every row and *shall* in two; they add nothing that separates the sentences.
3. **His list → 5 columns.** Each sentence keeps only what it is about: S1 = fee, pay; S2 = terminate; S3 = notice, write. The rows no longer overlap at all.

## Code: using the list

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words=list(stop_words_set))   # must be a LIST: a set raises an error
X = cv.fit_transform(lemmas)
print(cv.get_feature_names_out())    # ['fee' 'notice' 'pay' 'terminate' 'write']
```

> **Trap:** stop words are matched against tokens *after* lowercasing and tokenising, but *not* lemmatising: the vectoriser doesn't lemmatise. Since his text is already lemmatised, his list must hold lemmas (*party*, not *parties*).

## Assumptions & caveats

- **Assumes frequent = uninformative.** True for *the*; false for *not*. Always read the list.
- **The list is a modelling choice.** Choose it with training data only, and test changes on validation (e.g. keeping *may* vs *must*).
- **Interacts with n-grams.** Removing words before making bigrams joins words that weren't neighbours: *terminate this agreement* → *terminate agreement*.
- **TF-IDF already down-weights common words** via IDF, and `max_df` can drop them automatically. A hand list is still useful because it's explicit and protects the negations.
- **Transformers** (BERT etc.) are usually given the full text with no stop-word removal: they use word order and every small word.

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Fewer columns, less noise, faster | Can delete meaning (negations, modals) |
| Top features become readable | Generic lists don't fit specialist text |
| Domain list encodes real knowledge | Choosing the list takes judgement and testing |

## Quick reference

| term | one line |
|---|---|
| Stop word | very common word removed because it doesn't separate documents |
| `ENGLISH_STOP_WORDS` | sklearn's list, 318 words |
| Document frequency | share of documents containing the word |
| `CountVectorizer(binary=True)` | 0/1 presence; column mean = df |
| Add | shall, party, agreement, hereof, hereunder, herein |
| Keep | no, not, nor, never, neither, none, nothing, nobody, without, cannot, further, interest |
| Final list | sklearn's list + legal words − exempt words = 311 words |
| `difference_update` | modifies in place, returns `None` |
