# Corpus, documents & cleaning

The basic words of NLP, the steps that turn text into numbers, and the first step: cleaning text with regex.

## The running example

Every NLP chapter uses the same three sentences from his LEDGAR notes.

| id | sentence |
|---|---|
| S1 | The Party shall pay the Fees. |
| S2 | The parties may terminate this Agreement. |
| S3 | Notices shall be given to the party in writing. |

## Terms first

| term | plain meaning | in the example | in LEDGAR |
|---|---|---|---|
| **Corpus** | the whole collection of texts | all three sentences | ~80,000 clauses |
| **Document** | one text you make a prediction for (one row) | S1 | one contract clause |
| **Token** | one piece the text is cut into, usually a word | `the`, `party`, `fees` | |
| **Vocabulary** | the list of *different* tokens kept | 16 words after cleaning | 95,559 terms |
| **Feature** | one column the model learns from; here, one vocabulary word | the `fees` column | |
| **Document-term matrix** | a table: one row per document, one column per word, a number in each cell | 3 × 16 | 59,933 × 95,559 |

![Corpus to document-term matrix. The corpus holds three documents; each is cut into tokens (gold); the unique tokens form the vocabulary; each vocabulary word becomes a column of the matrix, and each cell counts how often that word is in that sentence.](figures/nlpterms-hierarchy.svg)

In the figure, find `the` in S1: it appears twice, so its cell is 2. Notice `parties` and `party` are still separate columns; later chapters merge them.

> **Key idea:** a computer only works with numbers. NLP preprocessing is just deciding *which* columns the table gets and *what number* goes in each cell.

## The pipeline

**text → clean → tokens → lemmas → remove stop words → numbers (TF-IDF) → model**

![S1 at each stage of the pipeline: raw, cleaned, tokenised, lemmatised (fees → fee), stop words removed (only pay and fee remain), then TF-IDF numbers, then the model.](figures/nlpterms-pipeline.svg)

Follow S1 down the figure: 6 tokens become 2 useful words, `pay` and `fee`. That is what the model actually sees.

| stage | chapter | learns from data? |
|---|---|---|
| Clean | this one | no, fixed rules |
| Tokenise, lemmatise | [Tokenise, stem & lemmatise](#/t/tokenise) | no |
| Stop words | [Stop words](#/t/stopwords) | the list is chosen using train data |
| Numbers | [Bag of Words](#/t/bow), [N-grams](#/t/ngrams), [TF-IDF](#/t/tfidf) | **yes**: fit on train only |
| Model | [Multiclass (softmax)](#/t/softmax) | yes |

> **Say it like this (from my notes):** fixed rules (cleaning, lemmatising) → apply to all splits. Anything that learns from data (vectoriser, model) → fit on train only.

## Cleaning with regex

**Regex** (regular expression) is a short pattern that describes text, like "a £ sign followed by digits". `re.sub(pattern, replacement, text)` finds every match and replaces it.

Cleaning has two jobs:

1. **Reduce noise.** `$5,000` and `$12,500` both just mean "an amount of money". Each different number would otherwise become its own useless column.
2. **Make the same thing look the same.** `Fees` and `fees`; `Agreement.` and `agreement`.

### His function

From [py_legal_data/src/clean.py](https://github.com/jag5m33/py_legal_data/blob/main/src/clean.py):

```python
import re
def clean_text(text):
    text = re.sub(r"\S+@\S+", " emailtoken ", text)                # emails
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)     # websites
    text = re.sub(r"[$£€]\s?\d[\d,]*(\.\d+)?", " moneytoken ", text) # money
    text = re.sub(r"\(\s*([a-z]|[ivx]+|\d+)\s*\)", " ", text, flags=re.I)  # (a) (ii) (3)
    text = re.sub(r"\d+([.,]\d+)*", " numtoken ", text)             # any other number
    text = text.lower()                                             # lowercase
    text = re.sub(r"[^a-z\s]", " ", text)                           # drop punctuation
    return re.sub(r"\s+", " ", text).strip()                        # tidy spaces
```

Each line swaps a messy thing for a simple **placeholder** word (`moneytoken`, `numtoken`, `emailtoken`). The model keeps the fact that "there was an amount here" but not the exact value.

### Worked example

| input | output |
|---|---|
| The Party shall pay the Fees. | the party shall pay the fees |
| The parties may terminate this Agreement. | the parties may terminate this agreement |
| Notices shall be given to the party in writing. | notices shall be given to the party in writing |
| Notices shall be sent to legal@acme.com within 30 days; pay \$5,000 per Section 2(a)(ii). | notices shall be sent to emailtoken within numtoken days pay moneytoken per section numtoken |

### Order matters

**Money before numbers.** If the number rule ran first, `$5,000` would lose its digits and the money rule would never match:

| order | `Pay $5,000 within 30 days` becomes |
|---|---|
| money, then numbers (his order) | `pay moneytoken within numtoken days` |
| numbers, then money | `pay numtoken within numtoken days` (the money is lost) |

> **Trap:** removing punctuation turns `can't` into `can t`. His EDA checked LEDGAR had no contractions first. In other text, expand contractions (`can't` → `cannot`) before cleaning.

## Code: clean a whole DataFrame

```python
import pandas as pd
from src.clean import clean_text     # his function, so every split is cleaned the same way

df = pd.DataFrame({"text": [
    "The Party shall pay the Fees.",
    "The parties may terminate this Agreement.",
    "Notices shall be given to the party in writing.",
]})
df["clean_text"] = df["text"].apply(clean_text)   # run it on every row
print(df["clean_text"].tolist())
# ['the party shall pay the fees', 'the parties may terminate this agreement',
#  'notices shall be given to the party in writing']
```

## Advantages / disadvantages of regex cleaning

| advantages | disadvantages |
|---|---|
| Fast, clear, gives the same result every time | Each rule is a judgement call; easy to delete meaning |
| Shrinks the vocabulary (all money → one token) | Wrong rule order causes silent bugs |
| No training data needed | Doesn't understand context |

## Quick reference

| term | one line |
|---|---|
| Corpus | all the documents |
| Document | one row; here one clause |
| Token | one piece of cut-up text |
| Vocabulary | the different tokens kept = the columns |
| Document-term matrix | documents × words, numbers in cells |
| `re.sub(p, r, t)` | replace every match of `p` in `t` with `r` |
| Placeholder | `moneytoken`, `numtoken`: keep the type, drop the value |
