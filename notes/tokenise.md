# Tokenise, stem & lemmatise

Cutting text into tokens, then reducing each token to a base form, so that "parties" and "party" end up in the same column.

## 1. Tokenisation

**Tokenisation** is splitting text into **tokens**, the units that become columns. "the parties agreed" → `the` · `parties` · `agreed`.

You don't code it yourself: spaCy and `TfidfVectorizer` both do it. But each tokeniser makes its own decisions, and those decisions change your vocabulary.

### What a tokeniser has to decide

Same sentence, three tokenisers (all run for real):

**Input:** `The Party can't pay U.S. $5,000.`

| tokeniser | tokens |
|---|---|
| spaCy (`nlp(text)`) | The · Party · ca · n't · pay · U.S. · \$ · 5,000 · . |
| sklearn default (`token_pattern`) | The · Party · can · pay · 000 |
| his `clean_text` then split on spaces | the · party · can · t · pay · u · s · moneytoken |

| decision | spaCy | sklearn default |
|---|---|---|
| Punctuation | its own token (`.`, `$`) | dropped |
| `can't` | split into `ca` + `n't` (so the negation survives as a token) | `can`, and `t` is dropped (too short) |
| `U.S.` | kept whole (it knows the abbreviation) | `U` and `S` dropped |
| `5,000` | one token | `5` dropped, `000` kept |
| One-letter words | kept | dropped: the default pattern `(?u)\b\w\w+\b` needs 2+ characters |

> **Trap:** sklearn's default `token_pattern` silently drops every one-character token. That's usually fine for English, but it drops `a` in `(a)` and single digits. See [TF-IDF parameters](#/t/tfidf).

## 2. POS tagging

**Part-of-speech (POS) tagging** labels each token with its grammatical role: noun, verb, adjective, determiner, and so on. spaCy's `tagger` does this with a small trained neural network, using the surrounding words as context.

spaCy on the running example (`token.text`, `token.pos_`, `token.lemma_`):

| S1 | The | Party | shall | pay | the | Fees | . |
|---|---|---|---|---|---|---|---|
| POS | DET | PROPN | AUX | VERB | DET | NOUN | PUNCT |
| lemma | the | **Party** | shall | pay | the | **fee** | . |

| S3 | Notices | shall | be | given | to | the | party | in | writing |
|---|---|---|---|---|---|---|---|---|---|
| POS | NOUN | AUX | AUX | VERB | ADP | DET | NOUN | ADP | VERB |
| lemma | notice | shall | be | **give** | to | the | party | in | **write** |

Notice capitalised `Party` was tagged PROPN (proper noun, a name) and kept as `Party`. On lowercased text it becomes a NOUN and lemma `party`. That's one reason his pipeline lowercases (in `clean_text`) *before* lemmatising.

## 3. Stemming vs lemmatisation

Both reduce a word to a base form, so that different forms count as the same feature.

- **Stemming:** chops endings off using fixed rules (e.g. Porter's rules: remove `-ing`, turn `-ies` into `-i`). It never looks the word up. The result is a **stem**, which may not be a real word.
- **Lemmatisation:** finds the dictionary form, the **lemma**, using a vocabulary and the word's POS. The result is always a real word.

### Real output

Porter stemmer (NLTK `PorterStemmer`) vs spaCy `en_core_web_sm`, word by word:

| word | Porter stem | spaCy lemma |
|---|---|---|
| parties | parti | party |
| terminating | termin | terminate |
| termination | termin | termination |
| notices | notic | notice |
| given | given | give |
| paid | paid | pay |
| writing | write | write |
| indemnification | indemnif | indemnification |
| governing | govern | govern |
| was | wa | be |
| better | better | well |

Read down the columns:

- Stemming merges `terminating` and `termination` (both `termin`); lemmatisation keeps the verb and the noun apart. Whether that's good depends on the task.
- Stemming misses irregular forms (`given`, `paid`, `was`); the lemmatiser knows them.
- Stems like `parti`, `notic`, `wa` are not words, so they're hard to read when you inspect top features later.
- Lemmatisation isn't perfect: single-word lookups can give odd lemmas (`better` → `well`, because without a sentence it guesses an adverb).

| | stemming | lemmatisation |
|---|---|---|
| How | chops endings off by rule | looks up the real dictionary word |
| parties | parti | party |
| terminating | termin | terminate |
| Speed | fast | slower (needs a tagger) |
| Result | sometimes not a real word | always a real word |
| Needs POS? | no | yes |
| Tool | NLTK `PorterStemmer` | spaCy `token.lemma_`, NLTK `WordNetLemmatizer` |

**He used lemmatisation (spaCy)** because legal words need to stay meaningful, and readable features make the error analysis possible.

### Why lemmatising needs the POS

The same spelling can be a different word. spaCy output:

| sentence | token | POS | lemma |
|---|---|---|---|
| We had a **meeting**. | meeting | NOUN | meeting |
| They are **meeting** the client. | meeting | VERB | meet |
| Notices given in **writing** | writing | NOUN | writing |
| …to the party in **writing**. | writing | VERB | write |

Without the tag, the lemmatiser can't know whether `meeting` is a thing or an action. This is why spaCy runs the tagger *before* the lemmatiser, and why you shouldn't switch the tagger off (spaCy will let you, with a warning, but the lemmas get worse).

> **Trap:** the last two rows show the tagger can disagree with itself: "in writing" in S3 was tagged VERB, giving `write`. Taggers are statistical and make mistakes, especially on short, odd or lowercased text.

## His Lemmatizer class

From [py_legal_data/src/lemmetizer.py](https://github.com/jag5m33/py_legal_data/blob/main/src/lemmetizer.py) (comments shortened):

```python
import spacy

class Lemmatizer:
    def __init__(self, model="en_core_web_sm", batch_size=500):
        # Load spaCy once, with parser and ner switched off (faster).
        self.nlp = spacy.load(model, disable=["parser", "ner"])
        self.batch_size = batch_size

    def lemmatize_clauses(self, df):
        results = []
        # nlp.pipe feeds cleaned clauses through the whole chain, 500 at a time
        for doc in self.nlp.pipe(df["clean_text"], batch_size=self.batch_size):
            words = []
            for token in doc:                 # tokenising and lemmatising already happened
                words.append(token.lemma_)    # e.g. "parties" -> "party"
            results.append(" ".join(words))   # back to one string per clause
        df["lemmatized_text"] = results
        return df
```

### How `nlp.pipe` works

`en_core_web_sm` is a chain of components. After `disable=["parser", "ner"]`, the remaining chain (checked with `nlp.pipe_names`) is:

**tokenizer → tok2vec → tagger → attribute_ruler → lemmatizer**

| component | what it does |
|---|---|
| tokenizer | splits the text into tokens (rules + exceptions like `U.S.`) |
| tok2vec | turns each token into a vector the next steps use |
| tagger | POS tag for each token |
| attribute_ruler | rule-based fixes to tags |
| lemmatizer | uses word + tag to look up the lemma |
| ~~parser~~ | sentence grammar tree: not needed for lemmas, switched off |
| ~~ner~~ | named entities (people, organisations): not needed, switched off |

`nlp(text)` processes one text. `nlp.pipe(texts, batch_size=500)` takes a whole column and processes it in batches, which is much faster than calling `nlp()` 60,000 times. It **yields** one `Doc` at a time, in the same order as the input, so `for doc in ...` lines up with the DataFrame rows.

Loading the model in `__init__` means it's loaded **once** per `Lemmatizer` object, not once per clause.

## Heatmap: columns merging

Each step changes the document-term matrix. Below is the count matrix for the three sentences at three stages (counted with `CountVectorizer`, one-letter tokens allowed).

![Document-term matrix of the three sentences at three stages. Raw: 18 columns, with The/the and Party/party/parties split. Lowercased: 16 columns. Lemmatised: 15 columns, and party now has a 1 in every row.](figures/tokenise-merge.svg)

What to look at, top to bottom:

1. **Raw (18 columns):** `The` and `the` are separate; so are `Party`, `party` and `parties`. Three columns for one idea.
2. **Lowercased (16):** `The`+`the` merge (S1 now shows `the` = 2), and `Party`+`party` merge. `parties` is still separate.
3. **Lemmatised (15):** `parties` → `party`, so the `party` column now has a 1 in every sentence. `fees` → `fee`, `notices` → `notice`, `given` → `give`, `writing` → `write`.

> **Key idea:** fewer, denser columns. Each merge means the model sees more examples of the same feature, so it learns its weight more reliably. With 60,000 clauses the savings are much larger than 18 → 15.

## Code

```python
import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])   # tagger + lemmatizer stay on
ps = PorterStemmer()

text = "the parties may terminate this agreement"
doc = nlp(text)                                    # tokenise, tag, lemmatise in one call

for token in doc:
    print(token.text, token.pos_, token.lemma_, ps.stem(token.text))
# parties NOUN party parti
# terminate VERB terminate termin
# ...

clauses = ["the party shall pay the fees", "notices shall be given to the party in writing"]
lemmas = [" ".join(t.lemma_ for t in doc) for doc in nlp.pipe(clauses, batch_size=500)]
print(lemmas)
# ['the party shall pay the fee', 'notice shall be give to the party in write']
```

## Advantages / disadvantages

| | advantages | disadvantages |
|---|---|---|
| Tokenisation (spaCy) | handles abbreviations, contractions, punctuation sensibly | slower than a regex split |
| Stemming | very fast, no model, merges more aggressively | non-words; over-merges (`termination`/`terminating`); misses irregulars |
| Lemmatisation | real words; handles irregulars; POS-aware | slower; needs a model; tagger errors become lemma errors |

## Limitations

- Lemmatising doesn't merge **synonyms** (`terminate` / `cancel`, `Governing Laws` / `Applicable Laws`). Only embeddings or hand-made rules do.
- Contractions after punctuation removal can go wrong (`won t` → `win t`). His EDA checked: no contractions in LEDGAR.
- On already-cleaned, lowercased text with no punctuation, the tagger has less context than on raw text, so it makes a few more mistakes.

## Quick reference

| term | one line |
|---|---|
| Token | one piece the tokeniser produced |
| Tokeniser | the rules that decide where tokens start and end |
| POS tag | grammatical role: NOUN, VERB, DET… (`token.pos_`) |
| Stem | ending chopped by rule; may not be a word (`parti`) |
| Lemma | dictionary form (`party`) via `token.lemma_` |
| Porter stemmer | NLTK's classic rule-based stemmer |
| `nlp.pipe(texts, batch_size)` | process many texts in batches; yields Docs in order |
| `disable=["parser","ner"]` | skip unneeded components for speed |
| Why POS matters | `meeting` (noun) vs `meet` (verb) |
