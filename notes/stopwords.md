# Stop words

Removing very common words that don't help tell documents apart, and building a list that fits legal text.

## Definition

**Stop words** are very common words that carry little information for the task, so they are removed before counting: *the, of, and, this, to, be…*

They are in almost every document, so they can't help separate one clause type from another. Leaving them in just adds noisy columns.

> **Key idea:** "stop word" depends on the task. *party* is a real word, but in contracts it is in nearly every clause, so it can't tell a Notices clause from a Payments clause.

## Step 1: start with sklearn's list

sklearn's `ENGLISH_STOP_WORDS` has **318** words. Some are surprising:

| in the list | why it matters for contracts |
|---|---|
| no, not, nor, never, none, without, cannot | **negations**: removing them flips the meaning |
| further, interest | part of clause names (Further Assurances, Interests) |

And it **doesn't** include the words that are everywhere in contracts: *shall, party, agreement, hereof, hereunder*.

## Step 2: find legal stop words with document frequency

**Document frequency (df)** = the share of documents that contain the word at least once:

$$
\text{df}(w) = \frac{\text{documents containing } w}{N}
$$

- $w$ = the word; $N$ = total number of documents.
- A word used five times in one clause still counts once.
- df near 100% → in nearly every clause → a stop-word candidate.

**Worked on the three lemmatised sentences** ($N = 3$):

| word | S1 | S2 | S3 | df |
|---|---|---|---|---|
| party | yes | yes | yes | 3/3 = **100%** |
| the | yes | yes | yes | 3/3 = **100%** |
| shall | yes | | yes | 2/3 = **67%** |
| pay, fee, terminate, notice, write… | one each | | | 33% |

![Document frequency of every word in the three lemmatised sentences. party and the are in all three; shall in two. Grey bars are already in sklearn's list; wine bars are the legal words he added; teal bars are the content words that survive.](figures/stopwords-df.svg)

In the figure, the tallest bars are *party* (wine: a legal word sklearn doesn't remove) and *the* (grey: sklearn removes it). The teal bars that survive are exactly the words that say what each sentence is *about*.

```python
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

cv = CountVectorizer(binary=True)     # 1 = word present, 0 = absent
X = cv.fit_transform(lemmas)
df = pd.Series(np.asarray(X.mean(axis=0)).ravel(),   # column average = df
               index=cv.get_feature_names_out())
print(df.sort_values(ascending=False).head(3))       # party 1.0, the 1.0, shall 0.667
```

On LEDGAR he did this on the **training** clauses only.

> **Trap (from his notebook):** `str.contains("per")` also matches *per*iod and pro*per*ty, so it gave 49.8% for "per". Match whole words (`r"\bper\b"`) or use the `binary=True` method above.

## Step 3: his custom legal list

From his notebook ([py_legal_data/notebooks/model.ipynb](https://github.com/jag5m33/py_legal_data/blob/main/notebooks/model.ipynb)):

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# KEEP these in the text
exempt_words = {'no', 'not', 'nor', 'never', 'neither', 'none', 'nothing',
                'nobody', 'without', 'cannot', 'further', 'interest'}
# REMOVE these too: common legal words
legal_words = {'shall', 'party', 'agreement', 'hereof', 'hereunder', 'herein'}

# sklearn's list + legal words - exemptions  (| = add, - = take away)
stop_words_set = (set(ENGLISH_STOP_WORDS) | legal_words) - exempt_words
```

Final size: $318 + 5 - 12 = 311$ words (*herein* was already in sklearn's list).

| decision | words | why |
|---|---|---|
| **Add** | shall, party, agreement, hereof, hereunder, herein | in most clauses, so they don't separate clause types |
| **Keep** | no, not, never, without, cannot… | they flip meaning: *No Waivers* vs *Waivers* |
| **Keep** | further, interest | they name classes |

> **Trap (his notebook bug):** `s = s.difference_update(x)` sets `s` to `None`, because `difference_update` changes the set in place and returns nothing. Use `(a | b) - c` instead.

## The "No Waivers" vs "Waivers" trap

LEDGAR has two labels that differ only by *no*.

- A: *no waiver of any breach shall be deemed a waiver of any other breach*
- B: *waiver of any breach shall be in writing*

| list | A becomes | B becomes |
|---|---|---|
| sklearn's (removes *no*) | breach 2, deemed 1, shall 1, waiver 2 | breach 1, shall 1, waiver 1, writing 1 |
| his (keeps *no*) | breach 2, deemed 1, **no 1**, waiver 2 | breach 1, waiver 1, writing 1 |

With sklearn's list, the one word that marks "No Waivers" is gone. His README shows Waivers → No Waivers is still a common confusion, so it stays hard even with *no* kept.

## Heatmap: before and after

![Top: lemmatised matrix, 15 columns, with grey labels marking words in sklearn's list. Bottom left: after sklearn's list, 8 columns; party, shall and agreement (wine) survive. Bottom right: after his legal list, 5 columns, each sentence showing only its content words.](figures/stopwords-heatmap.svg)

Top to bottom:

1. **15 columns.** Grey labels are in sklearn's list.
2. **sklearn's list → 8 columns.** *party* is still in every row; it adds nothing.
3. **His list → 5 columns.** S1 = fee, pay; S2 = terminate; S3 = notice, write. Each sentence keeps only what it is about.

## Using the list

```python
cv = CountVectorizer(stop_words=list(stop_words_set))   # must be a LIST, not a set
X = cv.fit_transform(lemmas)
print(cv.get_feature_names_out())    # ['fee' 'notice' 'pay' 'terminate' 'write']
```

> **Trap:** the vectoriser doesn't lemmatise, so the list must hold lemmas (*party*, not *parties*).

## Caveats

- **Assumes common = useless.** True for *the*, false for *not*. Always read the list.
- **Choose the list with training data only**, and test changes on validation.
- **TF-IDF already turns down common words**, but a hand list is explicit and protects negations.
- **Transformers** (BERT etc.) usually keep every word, stop words included.

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Fewer columns, less noise, faster | Can delete meaning (negations) |
| Top features become readable | Generic lists don't fit specialist text |
| A legal list adds real domain knowledge | Needs judgement and testing |

## Quick reference

| term | one line |
|---|---|
| Stop word | very common word removed because it doesn't separate documents |
| `ENGLISH_STOP_WORDS` | sklearn's list, 318 words |
| Document frequency | share of documents containing the word |
| `CountVectorizer(binary=True)` | 0/1 presence; column average = df |
| His list | sklearn's + legal words − negations etc. = 311 words |
| `difference_update` | changes in place, returns `None` |
