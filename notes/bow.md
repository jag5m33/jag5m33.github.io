# Bag of Words

The simplest way to turn text into numbers: count how often each vocabulary word appears in each document.

## Definition

**Bag of Words (BoW)** represents each document as a vector of **word counts**, with one position per vocabulary word. It's called a "bag" because the words are tipped into a bag: you keep *which* words and *how many*, and lose the order.

- One **row** per document (clause).
- One **column** per vocabulary word.
- Each **cell** = how many times that word appears in that document.

That table is the **document-term matrix** from [Corpus, documents & cleaning](#/t/nlpterms).

## Step 1: build the vocabulary (with frequencies)

Start from the cleaned, lemmatised sentences:

| id | lemmatised text |
|---|---|
| S1 | the party shall pay the fee |
| S2 | the party may terminate this agreement |
| S3 | notice shall be give to the party in write |

List every distinct word and count it across the whole corpus (15 words, 19 tokens):

| word | total count | in how many sentences |
|---|---|---|
| the | 4 | 3 |
| party | 3 | 3 |
| shall | 2 | 2 |
| agreement, be, fee, give, in, may, notice, pay, terminate, this, to, write | 1 each | 1 each |

The top of the table is *the*, *party*, *shall*: words that say nothing about what a clause is about. That's the BoW problem in one table.

Now remove stop words (sklearn's list here, so you can see *party* and *shall* survive). The vocabulary becomes 8 words, sorted alphabetically by sklearn:

$$
[\text{agreement},\ \text{fee},\ \text{notice},\ \text{party},\ \text{pay},\ \text{shall},\ \text{terminate},\ \text{write}]
$$

## Step 2: one vector per sentence

Go through the vocabulary in order and write the count for each word:

| | agreement | fee | notice | party | pay | shall | terminate | write |
|---|---|---|---|---|---|---|---|---|
| S1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| S2 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| S3 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |

So **S1 = (0, 1, 0, 1, 1, 1, 0, 0)**. Each sentence is now a point in 8-dimensional space.

![BoW document-term matrix for the three sentences after sklearn's stop words. Teal cells are words present. The right column counts the words kept per sentence: 4, 3, 4.](figures/bow-dtm.svg)

Look down the `party` column: a 1 in every row. A column that's the same for every document can't help tell them apart. That's why he added *party* to the stop words, and why [TF-IDF](#/t/tfidf) gives it a low weight.

## One-hot encoding of words vs BoW

**One-hot encoding** a word gives a vector the length of the vocabulary with a single 1 at that word's position (the same idea as one-hot encoding a category in [Scaling & encoding](#/t/scaling)). A sentence is then a *stack* of one-hot rows, one per word.

**BoW = add those rows up.** You get one row per sentence, and the order of the rows (the word order) is gone.

![Top: S1 as six one-hot rows over the 15-word vocabulary, one per token. Bottom: summing the rows gives the BoW vector, where the appears twice so its cell is 2.](figures/bow-onehot.svg)

In the figure, the two `the` rows each have a 1 in the same column; in the BoW row below they add to 2.

| | one-hot (per word) | BoW (per document) |
|---|---|---|
| Shape for S1 | 6 × 15 (one row per token) | 1 × 15 |
| Keeps order? | yes (row order) | no |
| Used for | input to sequence models, embeddings | classic ML (logistic regression, Naive Bayes) |

## What BoW throws away

- **Word order.** *"the party shall pay the fee"* and *"the fee shall pay the party"* get **identical** vectors.
- **Meaning and synonyms.** *terminate* and *cancel* are unrelated columns, just as distant as *terminate* and *fee*.
- **Context.** *interest* (money) and *interest* (ownership) share one column.
- **Unseen words.** A word not in the training vocabulary is silently ignored at transform time.

[N-grams](#/t/ngrams) bring back a little local order; embeddings bring back meaning.

## Sparse matrices

Each document uses only a few words from the vocabulary, so most cells are 0.

- Three sentences, 15-word vocabulary: 20 non-zero cells out of 45, so **56% zeros**.
- LEDGAR: 59,933 × 95,559 (README; his earlier notes, before the stop-word fix, had 95,613) with about 73 non-zeros per clause, so **over 99.9% zeros**.

A **sparse matrix** stores only the non-zero values and their positions. sklearn returns a SciPy **CSR** (compressed sparse row) matrix. Printing S1 from a `CountVectorizer` *without* stop-word removal (the 15-word vocabulary) shows only its 5 stored values:

```
Coords    Values
(0, 11)   2        <- column 11 = 'the'
(0, 7)    1        <- 'party'
(0, 9)    1        <- 'shall'
(0, 8)    1        <- 'pay'
(0, 2)    1        <- 'fee'
```

A dense LEDGAR matrix of 64-bit numbers would need about $59{,}933 \times 95{,}613 \times 8$ bytes ≈ 46 GB. Sparse, it's tens of MB.

> **Key idea (from my notes):** zeros are normal. A 0 still means something: "this word isn't here".

> **Trap:** calling `.toarray()` on a big sparse matrix converts it to dense and can crash the kernel. Only do it on tiny examples like this one.

## Code

```python
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

lemmas = ["the party shall pay the fee",
          "the party may terminate this agreement",
          "notice shall be give to the party in write"]

cv = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS))  # tokenise, lowercase, drop stop words
X = cv.fit_transform(lemmas)        # FIT: learn the vocabulary; TRANSFORM: count -> sparse matrix

print(cv.get_feature_names_out())   # the columns, alphabetical
# ['agreement' 'fee' 'notice' 'party' 'pay' 'shall' 'terminate' 'write']
print(X.toarray())                  # dense view (fine for 3 rows only)
# [[0 1 0 1 1 1 0 0]
#  [1 0 0 1 0 0 1 0]
#  [0 0 1 1 0 1 0 1]]
print(cv.vocabulary_)               # dict word -> column index

new = cv.transform(["the party shall arbitrate"])  # TRANSFORM only: 'arbitrate' is unknown, ignored
print(new.toarray())                # [[0 0 0 1 0 1 0 0]]
```

`fit` learns the vocabulary, so it belongs on **training data only**; `transform` validation and test with the same vocabulary.

## How he used it

He used BoW **only to find stop words** (document frequency with `binary=True`, see [Stop words](#/t/stopwords)), not as the model's input. The model uses [TF-IDF](#/t/tfidf).

## Assumptions & caveats

- **Order doesn't matter** (the bag assumption). False for language, but often good enough for *topic*-style tasks like clause type.
- **Each word is an independent feature.** Synonyms aren't linked.
- **Length bias.** Long clauses have bigger counts; normalise (L2, see [TF-IDF](#/t/tfidf)) before comparing.
- **Common words dominate** the counts unless removed or down-weighted.
- **Fixed vocabulary** from training; new words at test time are dropped.

## Use cases

- A quick, strong baseline for text classification (spam, topic, clause type).
- Finding stop words and checking vocabulary size.
- Input to Multinomial Naive Bayes, which expects counts.

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Very simple; easy to explain | Loses word order and meaning |
| Fast; sparse matrices scale to large corpora | Common words get the biggest numbers |
| Features are readable (each column is a word) | Huge, sparse vocabulary |
| Works well with linear models | Synonyms are unrelated columns |

## Quick reference

| term | one line |
|---|---|
| Bag of Words | word counts per document; order ignored |
| Vocabulary | distinct words kept = columns |
| Document vector | one row of counts, e.g. S1 = (0,1,0,1,1,1,0,0) |
| One-hot word | 1 at the word's position, 0 elsewhere; BoW = sum of a document's one-hot rows |
| Sparse matrix | stores only non-zeros (CSR) |
| `CountVectorizer().fit_transform(train)` | learn vocabulary + count |
| `.transform(new)` | count with the existing vocabulary |
| `get_feature_names_out()` | column names |
