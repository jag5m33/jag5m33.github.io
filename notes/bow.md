# Bag of Words

The simplest way to turn text into numbers: count how often each word appears in each document.

## Definition

**Bag of Words (BoW)** turns each document into a row of **word counts**, one position per vocabulary word. Think of tipping the words into a bag: you keep *which* words and *how many*, but lose the order.

- One **row** per document (clause).
- One **column** per vocabulary word.
- Each **cell** = how many times that word appears in that document.

That table is the **document-term matrix** from [Corpus, documents & cleaning](#/t/nlpterms).

## Step 1: build the vocabulary

Start from the cleaned, lemmatised sentences:

| id | lemmatised text |
|---|---|
| S1 | the party shall pay the fee |
| S2 | the party may terminate this agreement |
| S3 | notice shall be give to the party in write |

Count every different word across all three (15 words):

| word | total count | in how many sentences |
|---|---|---|
| the | 4 | 3 |
| party | 3 | 3 |
| shall | 2 | 2 |
| every other word | 1 | 1 |

The top words are *the*, *party*, *shall*: they say nothing about what a clause is about. That's the BoW problem in one table.

Now remove sklearn's stop words (so *party* and *shall* still survive). The vocabulary becomes 8 words, in alphabetical order:

**agreement, fee, notice, party, pay, shall, terminate, write**

## Step 2: one row per sentence

Go through the vocabulary in order and write each word's count:

| | agreement | fee | notice | party | pay | shall | terminate | write |
|---|---|---|---|---|---|---|---|---|
| S1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| S2 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| S3 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |

So **S1 = (0, 1, 0, 1, 1, 1, 0, 0)**.

![BoW document-term matrix for the three sentences after sklearn's stop words. Teal cells are words present. The right column counts the words kept per sentence: 4, 3, 4.](figures/bow-dtm.svg)

Look down the `party` column: a 1 in every row. A column that is the same for every document can't help tell them apart. That's why he added *party* to his stop words, and why [TF-IDF](#/t/tfidf) gives it a low weight.

## One-hot words add up to BoW

**One-hot encoding** a word gives a row as long as the vocabulary, with a single 1 at that word's position. A sentence is then a stack of one-hot rows, one per word.

**BoW = add those rows up.** You get one row per sentence, and the word order is gone.

![Top: S1 as six one-hot rows over the 15-word vocabulary, one per token. Bottom: summing the rows gives the BoW vector, where the appears twice so its cell is 2.](figures/bow-onehot.svg)

In the figure, the two `the` rows each have a 1 in the same column; in the BoW row they add up to 2.

## What BoW throws away

- **Word order.** *"the party shall pay the fee"* and *"the fee shall pay the party"* get the **same** row.
- **Meaning.** *terminate* and *cancel* are unrelated columns.
- **New words.** A word not seen in training is ignored later.

[N-grams](#/t/ngrams) bring back a little word order.

## Mostly zeros (sparse)

Each document uses only a few of the vocabulary words, so most cells are 0. On LEDGAR (59,933 × 95,559) over 99.9% of cells are zero. sklearn stores only the non-zero cells, a **sparse matrix**, which saves a huge amount of memory.

> **Key idea (from my notes):** zeros are normal. A 0 still means something: "this word isn't here".

> **Trap:** `.toarray()` turns a sparse matrix into a full one. Fine for 3 rows, but on LEDGAR it can crash the kernel.

## Code

```python
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

cv = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS))
X = cv.fit_transform(lemmas)        # fit = learn vocabulary; transform = count
print(cv.get_feature_names_out())   # ['agreement' 'fee' 'notice' 'party' 'pay' 'shall' 'terminate' 'write']
print(X.toarray())                  # [[0 1 0 1 1 1 0 0] [1 0 0 1 0 0 1 0] [0 0 1 1 0 1 0 1]]

new = cv.transform(["the party shall arbitrate"])   # 'arbitrate' is unknown, ignored
print(new.toarray())                                # [[0 0 0 1 0 1 0 0]]
```

`fit` learns the vocabulary, so use it on **training data only**, then `transform` validation and test.

## How he used it

He used BoW **only to find stop words** (see [Stop words](#/t/stopwords)). The model's input is [TF-IDF](#/t/tfidf).

## Caveats

- **Order is ignored.** Wrong for language, but often good enough for topic-style tasks like clause type.
- **Long documents get bigger counts.** TF-IDF fixes this by normalising each row.
- **Common words get the biggest numbers** unless removed.

## Use cases

- A quick, strong starting model for text classification (spam, topic, clause type).
- Finding stop words.
- Input to Naive Bayes, which expects counts.

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Very simple; easy to explain | Loses word order and meaning |
| Fast, even on large corpora | Common words get the biggest numbers |
| Each column is a readable word | Very many columns |

## Quick reference

| term | one line |
|---|---|
| Bag of Words | word counts per document; order ignored |
| Vocabulary | different words kept = columns |
| Document row | e.g. S1 = (0,1,0,1,1,1,0,0) |
| One-hot word | 1 at the word's position; BoW = sum of a sentence's one-hot rows |
| Sparse matrix | stores only the non-zero cells |
| `fit_transform(train)` | learn vocabulary + count |
| `transform(new)` | count with the existing vocabulary |
