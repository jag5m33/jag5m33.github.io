# Corpus, documents & cleaning

The vocabulary of NLP, the pipeline that turns text into numbers, and the first step of that pipeline: regex cleaning.

## The running example

Every NLP chapter uses the same three sentences, taken from his LEDGAR notes. Every matrix is produced by running spaCy and scikit-learn on them.

| id | sentence |
|---|---|
| S1 | The Party shall pay the Fees. |
| S2 | The parties may terminate this Agreement. |
| S3 | Notices shall be given to the party in writing. |

## Terms first

| term | plain meaning | in the example | in LEDGAR |
|---|---|---|---|
| **Corpus** | the whole collection of texts | all three sentences | ~80,000 clauses |
| **Document** | one unit you make a prediction for (one row) | S1 | one contract clause |
| **Sentence** | a grammatical sentence; may or may not equal a document | here each document is one sentence | a clause can have several |
| **Word** | a unit of language as a person reads it | "Fees" | |
| **Token** | one piece the tokeniser cut the text into: usually a word, sometimes punctuation or part of a word | `the`, `party`, `fees` | |
| **Vocabulary** | the set of *distinct* tokens kept across the corpus | 16 words after cleaning | 95,559 terms |
| **Feature** | one column the model learns from; in BoW / TF-IDF, one vocabulary entry | the `fees` column | |
| **Document-term matrix (DTM)** | table with one row per document, one column per vocabulary entry, and a number in each cell | 3 × 16 | 59,933 × 95,559 |

![Corpus to document-term matrix. The corpus holds three documents; each is cut into tokens (gold); the unique tokens form the vocabulary; each vocabulary word becomes a column of the matrix, and each cell counts how often that word is in that sentence.](figures/nlpterms-hierarchy.svg)

In the figure, find `the` in S1: it appears twice in the gold tokens, so its cell in the matrix (dark teal) is 2. Every other cell is 0 or 1. Notice `parties` and `party` are still separate columns, and so are `notices`; later chapters merge them.

> **Key idea:** a computer only works with numbers. All of NLP preprocessing is about deciding *which* columns the matrix gets and *what number* goes in each cell.

## The pipeline

**text → clean → tokens → lemmas → remove stop words → numbers (TF-IDF) → model**

![S1 at each stage of the pipeline: raw, cleaned, tokenised, lemmatised (fees → fee), stop words removed (only pay and fee remain), then TF-IDF numbers, then the model.](figures/nlpterms-pipeline.svg)

Follow S1 down the figure: 6 tokens become 2 useful words, `pay` and `fee`, each with weight 0.71 (with his full stop-word list, see [Stop words](#/t/stopwords)). That is what the model actually sees.

| stage | chapter | tool | learns from data? |
|---|---|---|---|
| Clean | this one | `re` (his `clean.py`) | no, fixed rules |
| Tokenise, lemmatise | [Tokenise, stem & lemmatise](#/t/tokenise) | spaCy | no (pre-trained) |
| Stop words | [Stop words](#/t/stopwords) | a word list | no, but *choosing* the list uses train data |
| Numbers | [Bag of Words](#/t/bow), [N-grams](#/t/ngrams), [TF-IDF](#/t/tfidf) | `TfidfVectorizer` | **yes**: fit on train only |
| Model | [Multiclass (softmax)](#/t/softmax) | `LogisticRegression` | yes |

> **Say it like this (from my notes):** fixed rules (cleaning, lemmatising) → apply to all splits. Anything that learns from data (vectoriser, model) → fit on train only.

## Cleaning with regex

**Regex** (regular expression) is a mini-language for describing text patterns. `re.sub(pattern, replacement, text)` finds every match of the pattern and replaces it.

Cleaning has two jobs:

1. **Reduce noise.** `$5,000` and `$12,500` mean the same thing for clause *type*: "an amount of money". Different numbers would each become a separate, useless column.
2. **Make the same thing look the same.** `Fees` and `fees`; `Agreement.` and `agreement`.

### His function

From [py_legal_data/src/clean.py](https://github.com/jag5m33/py_legal_data/blob/main/src/clean.py):

```python
import re
def clean_text(text):
    text = re.sub(r"\S+@\S+", " emailtoken ", text) # emails
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text) # websites
    text = re.sub(r"[$£€]\s?\d[\d,]*(\.\d+)?", " moneytoken ", text) # money
    text = re.sub(r"\(\s*([a-z]|[ivx]+|\d+)\s*\)", " ", text, flags=re.I) # list markers: (a) (ii) (3)
    text = re.sub(r"\d+([.,]\d+)*", " numtoken ", text) # any other number
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text) # punctuation and symbols
    return re.sub(r"\s+", " ", text).strip()
```

### Line by line

| line | pattern in words | before → after |
|---|---|---|
| emails | `\S+` = one or more non-spaces, then `@`, then more non-spaces | `legal@acme.com` → `emailtoken` |
| websites | `http` or `https`, `://`, then non-spaces; or `www.` and non-spaces | `www.sec.gov` → `urltoken` |
| money | a currency sign, optional space, a digit, then digits/commas, optional decimals | `$5,000` → `moneytoken` |
| list markers | `(`, one letter **or** roman numerals i/v/x **or** digits, `)`; `re.I` = ignore case | `(a)`, `(ii)`, `(3)` → removed |
| numbers | digits, optionally followed by `.` or `,` and more digits | `30`, `2.5` → `numtoken` |
| lowercase | | `Notices` → `notices` |
| punctuation | `[^a-z\s]` = anything that is **not** a letter or whitespace | `fees.` → `fees` |
| spaces | `\s+` = one or more whitespace characters, replaced by one space; `.strip()` trims the ends | |

The placeholders are padded with spaces (`" moneytoken "`) so they never glue onto a neighbouring word.

### Worked example (run through his function)

| input | output |
|---|---|
| The Party shall pay the Fees. | the party shall pay the fees |
| The parties may terminate this Agreement. | the parties may terminate this agreement |
| Notices shall be given to the party in writing. | notices shall be given to the party in writing |
| Notices shall be sent to legal@acme.com within 30 days; pay \$5,000 per Section 2(a)(ii). | notices shall be sent to emailtoken within numtoken days pay moneytoken per section numtoken |

### Order matters

**Money before numbers.** If the number rule runs first, `$5,000` loses its digits before the money rule can see them:

| order | `Pay $5,000 within 30 days` becomes |
|---|---|
| money, then numbers (his order) | `pay moneytoken within numtoken days` |
| numbers, then money | `Pay $ numtoken within numtoken days`; the money rule never matches, and the `$` is later deleted as punctuation |

The same logic applies elsewhere: **placeholders before removing punctuation** (the email rule needs the `@` and `.`), and **list markers before numbers** (otherwise `(3)` becomes `( numtoken )`).

> **Trap:** contractions. Removing punctuation turns `can't` into `can t`, and spaCy then lemmatises `won t` to `win t`, which flips the meaning. His EDA checked LEDGAR for contractions (none found) before relying on this. A safer general rule is to expand contractions before cleaning.

> **Trap:** `U.S.` becomes `u s`, two meaningless one-letter tokens. Harmless here, but in other corpora you'd protect known abbreviations first.

## Code: clean a whole DataFrame

```python
import pandas as pd
from src.clean import clean_text          # his function, imported so every split is cleaned identically

df = pd.DataFrame({"text": [
    "The Party shall pay the Fees.",
    "The parties may terminate this Agreement.",
    "Notices shall be given to the party in writing.",
]})

df["clean_text"] = df["text"].apply(clean_text)   # run the function on every row
print(df["clean_text"].tolist())
# ['the party shall pay the fees',
#  'the parties may terminate this agreement',
#  'notices shall be given to the party in writing']
```

Cleaning is a fixed rule, so it is safe to apply to train, validation and test in the same way, before the split or after.

## Advantages / disadvantages of regex cleaning

| advantages | disadvantages |
|---|---|
| Fast, transparent, reproducible | Every rule is a judgement call; easy to delete meaning |
| Shrinks the vocabulary (all money → one token) | Rule order causes silent bugs |
| No training data needed | Can't understand context (`(a)` as a list marker vs a real word) |
| Placeholders keep the *fact* that a number was there | Loses the values: `\$5` and `\$5m` look the same |

## Quick reference

| term | one line |
|---|---|
| Corpus | all the documents |
| Document | one row; here one clause |
| Token | one piece produced by the tokeniser |
| Vocabulary | distinct tokens kept = the matrix's columns |
| Feature | one column |
| Document-term matrix | documents × vocabulary, numbers in cells |
| `re.sub(p, r, t)` | replace every match of pattern `p` in `t` with `r` |
| `\S` / `\s` / `\d` | non-space / whitespace / digit |
| `[^a-z\s]` | anything not a lowercase letter or space |
| Placeholder token | `moneytoken`, `numtoken`, `emailtoken`: keep the type, drop the value |
| Order | emails → URLs → money → list markers → numbers → lowercase → punctuation → spaces |
