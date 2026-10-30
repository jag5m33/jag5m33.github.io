# Naive Bayes

Naive Bayes picks the class that most likely produced the features it sees, using Bayes' theorem plus one simplifying ("naive") assumption. Every variant shares the same core and differs in just one place: how it works out P(feature | class).

## Bayes' theorem in one example

$$
P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}
$$

| Part | Name | Plain meaning |
|---|---|---|
| $P(A)$ | **Prior** | how common A is before looking at anything |
| $P(B \mid A)$ | **Likelihood** | how typical the evidence B is when A is true |
| $P(B)$ | **Marginal probability** | how common the evidence B is overall |
| $P(A \mid B)$ | **Posterior** | how likely A is **after** seeing B: what we want |

In a classifier, A is the **class** and B is the **features**.

### Where it comes from: two marbles

A bag has **3 red and 2 black** marbles. Take two out, one after the other, without putting the first back. The second draw **depends** on the first.

- Event **A** = first marble is black: $P(A) = 2/5$.
- Event **B** = second marble is black, given A happened: only 1 black left out of 4, so $P(B \mid A) = 1/4$.
- Both black: $P(A \cap B) = P(A) \times P(B \mid A) = 2/5 \times 1/4 = 1/10$.

$A \cap B$ ("A and B") is the same event as $B \cap A$, so the two ways of writing it must be equal:

$$
P(B \mid A) \times P(A) = P(A \mid B) \times P(B)
$$

Divide both sides by $P(B)$ and you have **Bayes' theorem**. That's the whole derivation.

### A medical test

A disease affects 1% of people. A test catches 90% of people who have it, and wrongly flags 5% of healthy people. You test positive. How likely is it that you're ill?

![Of 10,000 people, 100 are ill and 90 of them test positive; of 9,900 healthy people, 495 test positive. Only 90 of the 585 positives are really ill.](figures/naivebayes-medical.svg)

Only **90 of 585** positives are really ill: **15.4%**. The healthy group is so big that its 5% false alarms swamp the real cases.

> **Say it like this:** "The posterior is the prior updated by the evidence. A rare class needs strong evidence before it becomes the most likely answer."

## The core (every variant)

With features $x_1 \dots x_n$ and class $y$, Bayes' theorem plus the naive assumption gives:

$$
P(y \mid x_1, \dots, x_n) = \frac{P(y) \times P(x_1 \mid y) \times \dots \times P(x_n \mid y)}{P(x_1) \times \dots \times P(x_n)}
$$

The bottom line is the **same for every class** (it only depends on the record), so we can drop it. The answer is then "directly proportional" ($\propto$) to the top:

$$
P(y \mid x_1, \dots, x_n) \propto P(y) \times \prod_{i} P(x_i \mid y)
$$

$\prod$ means "multiply them all". **Prediction = argmax**: work this out for every class and pick the biggest. If it comes to P(yes) = 0.7 vs P(no) = 0.3, predict yes.

```
P(class | features)  ∝  P(class)  ×  P(feature 1 | class) × P(feature 2 | class) × …
     posterior           prior                    likelihoods
```

| Part | Meaning |
|---|---|
| **prior** P(class) | how common the class is |
| **likelihood** P(feature \| class) | how typical this feature is for the class |
| **"naive"** | features treated as independent within a class, so likelihoods are just multiplied |
| **logs** | multiplying many tiny numbers underflows to 0, so in practice logs are **added** ([Logs & exponents](#/t/logs)) |
| **predict** | the class with the highest result |

- "Naive" does **not** mean every feature counts equally: each one gets its own learned likelihood.
- "Multinomial" does **not** mean multi-class: **all** variants handle many classes.

## Overview: from most like basic Naive Bayes to least

| # | Variant (scikit-learn) | Similarity to basic NB | Each feature is… | P(feature \| class) comes from… | On TF-IDF? |
|---|---|---|---|---|---|
| 1 | **CategoricalNB** | ●●●●● the textbook version | a category (sunny / rainy) | share of the class's rows with that category | ✗ |
| 2 | **MultinomialNB** | ●●●●○ counting, on word weights | a count or TF-IDF | share of the class's total TF-IDF on this word | ✓ |
| 3 | **BernoulliNB** | ●●●○○ counting, yes/no only | present / absent (1 / 0) | share of the class's clauses containing it | ~ (loses strength) |
| 4 | **ComplementNB** | ●●○○○ Multinomial, flipped | a count or TF-IDF | Multinomial's formula on **every other class** | ✓✓ |
| 5 | **GaussianNB** | ●○○○○ no counting | a continuous number | a **bell curve** (mean, spread) | ✗ |

## The example dataset (TF-IDF, like LEDGAR's `x_train`)

All the examples below use the **same small TF-IDF dataset**: 5 clauses × 4 n-gram columns. Each cell is a TF-IDF score from 0 to 1, and each row is scaled to length 1 (L2). See [TF-IDF & its parameters](#/t/tfidf).

```
            notice   waive   govern   law      label
clause 1     0.99    0.00    0.00     0.16     Notices
clause 2     0.97    0.00    0.00     0.24     Notices
clause 3     0.12    0.98    0.00     0.12     Waivers
clause 4     0.00    0.00    0.78     0.62     Governing Laws
clause 5     0.00    0.00    0.62     0.78     Governing Laws
```

**New clause to classify** ("…waive… law…"):

```
            notice   waive   govern   law
new          0.00    0.71    0.00     0.71
```

## 1. CategoricalNB: the textbook version

> **What you need to know**
> - features are **categories**; likelihood = share of the class's rows with that category
> - it's the classic "weather / play tennis" Naive Bayes, worked below
> - **can't use TF-IDF** (numbers, not categories), so it's not for LEDGAR
>
> **Skip:** memorising the tables; know the steps (count, divide, multiply, compare)

### Worked example: should we play tennis today?

14 past days: **Yes 9, No 5**, so the priors are P(Yes) = 9/14 and P(No) = 5/14. Two features: outlook and temperature.

**Outlook** (as in Jag's notes):

| Outlook | Yes | No | P(x \| Yes) | P(x \| No) |
|---|---|---|---|---|
| Sunny | 2 | 3 | 2/9 | 3/5 |
| Overcast | 3 | 2 | 3/9 | 2/5 |
| Rainy | 4 | 0 | 4/9 | 0/5 |

> **Check:** in the standard play-tennis dataset the counts are Overcast 4 Yes / 0 No and Rainy 3 Yes / 2 No, so the Overcast and Rainy rows look swapped. It doesn't change the calculation below, which only uses Sunny.

Notice P(Rainy | No) = 0/5. A zero likelihood would wipe out "No" for every rainy day, whatever the other features say. That's what smoothing (`alpha`, below) prevents.

**Temperature:**

| Temperature | Yes | No | P(x \| Yes) | P(x \| No) |
|---|---|---|---|---|
| Hot | 2 | 2 | 2/9 | 2/5 |
| Mild | 4 | 2 | 4/9 | 2/5 |
| Cool | 3 | 1 | 3/9 | 1/5 |

**Today = (Sunny, Hot).** Multiply prior × likelihoods for each class:

$$
P(\text{Yes} \mid \text{today}) \propto \tfrac{2}{9} \times \tfrac{2}{9} \times \tfrac{9}{14} = \tfrac{2}{63} = 0.0317
$$

$$
P(\text{No} \mid \text{today}) \propto \tfrac{3}{5} \times \tfrac{2}{5} \times \tfrac{5}{14} = 0.0857
$$

These don't add to 1 (we dropped the bottom line), so **normalise**: divide each by their total.

- P(Yes) = 0.0317 / (0.0317 + 0.0857) = **0.27**
- P(No) = 0.0857 / (0.0317 + 0.0857) = **0.73**

**"Today play tennis = NO (0.73)."**

**Use when:** features are labels from a small set (survey answers, colour, region).

## 2. MultinomialNB: on TF-IDF

> **What you need to know**
> - **what it learns:** for each class, how much of that class's text goes to each word
> - **how it predicts:** prior × word likelihoods; the class the clause fits **best** wins
> - **works on TF-IDF** (designed for counts, fine with fractions; no negatives, and TF-IDF has none)
> - **`alpha`:** smoothing, so an unseen word doesn't give a probability of zero
> - **weakness:** small classes are learned from little text and get a small prior, so predictions get **pulled towards big classes**
>
> **Skip:** the exact formula, the ln() sums, doing it by hand: `fit()` / `predict()` do it

**Step 1: total each class's TF-IDF per word**

| Class | notice | waive | govern | law | total | prior |
|---|---|---|---|---|---|---|
| Notices | 1.96 | 0.00 | 0.00 | 0.40 | 2.36 | 2/5 |
| Waivers | 0.12 | 0.98 | 0.00 | 0.12 | 1.22 ← smallest | 1/5 |
| Governing Laws | 0.00 | 0.00 | 1.40 | 1.40 | 2.80 | 2/5 |

**Step 2: probabilities** with `alpha = 1` and 4 words: (total for word + 1) ÷ (class total + 4)

| Class | notice | waive | govern | law |
|---|---|---|---|---|
| Notices | 0.465 | 0.157 | 0.157 | 0.220 |
| Waivers | 0.215 | **0.379** | 0.192 | 0.215 |
| Governing Laws | 0.147 | 0.147 | 0.353 | **0.353** |

**Step 3: score the new clause** = ln(prior) + Σ TF-IDF × ln(probability). **Highest wins.**

| Class | ln(prior) | 0.71 × ln P(waive) | 0.71 × ln P(law) | Score |
|---|---|---|---|---|
| Notices | −0.92 | −1.31 | −1.07 | −3.30 |
| Waivers | −1.61 | −0.69 | −1.09 | −3.39 |
| Governing Laws | −0.92 | −1.36 | −0.74 | **−3.02 ← highest** |

**Prediction: Governing Laws** ✗ wrong. The clause contains "waive", but:

- Waivers has the **smallest prior** (1/5 vs 2/5): a −0.69 penalty before any words count
- Waivers' probabilities come from only **1.22** of TF-IDF, so smoothing spreads them thin

That's Multinomial's **imbalance problem**, in miniature.

**Use when:** text with word counts or TF-IDF, roughly balanced classes.

## 3. BernoulliNB: TF-IDF turned into yes/no

> **What you need to know**
> - features are **present / absent (1 / 0)**; how often or how strongly doesn't matter
> - **absent words count as evidence too** (Multinomial ignores them)
> - on TF-IDF it **throws away the strength** of each score (0.99 and 0.12 both become 1)
> - **use for:** short texts or genuine yes/no features; weaker for long clauses
>
> **Skip:** the probability calculations

Any TF-IDF above 0 becomes 1 (BernoulliNB does this itself with `binarize=0.0`):

```
            notice   waive   govern   law      label
clause 1       1       0       0       1       Notices
clause 2       1       0       0       1       Notices
clause 3       1       1       0       1       Waivers
clause 4       0       0       1       1       Governing Laws
clause 5       0       0       1       1       Governing Laws
new            0       1       0       1
```

**Likelihood:** P("waive" present | Waivers) = share of Waivers clauses containing it.
**Absence counts too:** the new clause *lacks* "notice", which is evidence against Notices.

**Use when:** short texts (tweets, subject lines) or genuinely yes/no features. LEDGAR clauses are long (median 86 words) and TF-IDF strength matters, so it's weaker there.

## 4. ComplementNB: Multinomial flipped for imbalance

> **What you need to know**
> - **what it learns:** for each class, word frequencies from **all the other classes'** text
> - **why that helps:** every class, even tiny ones, is estimated from a big pool, so small classes aren't disadvantaged (the **imbalance fix**)
> - **how it predicts:** the class whose "everything except me" profile the clause fits **worst**
> - **ignores the prior:** rare classes aren't penalised just for being rare
> - **`alpha`:** smoothing, as in Multinomial
> - same input as Multinomial, so it **works on TF-IDF**
>
> **Skip:** the adding, dividing and ln() sums below; they're there so the idea clicks once

**Step 1: total each class**: same as Multinomial step 1.

**Step 2: build the complement table**: for each class, add up **every other** class.

| Complement of… | notice | waive | govern | law | total |
|---|---|---|---|---|---|
| not-Notices (Waivers + Gov Laws) | 0.12 | 0.98 | 1.40 | 1.52 | 4.02 |
| not-Waivers (Notices + Gov Laws) | 1.96 | 0.00 | 1.40 | 1.80 | **5.16** |
| not-Gov Laws (Notices + Waivers) | 2.08 | 0.98 | 0.00 | 0.52 | 3.58 |

Waivers alone: **1.22**. Its complement: **5.16**, now the *biggest* pool, not the smallest.

![For LEDGAR's largest class (Governing Laws) and smallest (Books, 23 training clauses), how many clauses each method learns from. The complement of Books still has nearly 60,000.](figures/naivebayes-complement.svg)

**Step 3: probabilities** with `alpha = 1`: (total for word + 1) ÷ (complement total + 4)

| Complement of… | notice | waive | govern | law |
|---|---|---|---|---|
| not-Notices | 0.140 | 0.247 | 0.299 | 0.314 |
| not-Waivers | 0.323 | **0.109** | 0.262 | 0.306 |
| not-Gov Laws | 0.406 | 0.261 | 0.132 | 0.201 |

**Step 4: score the new clause** = Σ TF-IDF × ln(probability). **No prior. Lowest wins.**

| Complement of… | 0.71 × ln P(waive) | 0.71 × ln P(law) | Score (fit to "not this class") |
|---|---|---|---|
| not-Notices | −0.99 | −0.82 | −1.82 |
| not-Waivers | −1.57 | −0.84 | **−2.41 ← fits worst** |
| not-Gov Laws | −0.95 | −1.14 | −2.09 |

**Step 5: predict** the class whose complement fits worst → **Waivers** ✓

"waive" barely appears outside Waivers, so the clause looks very *unlike* everything that isn't Waivers.

> **Say it in one breath:** "Multinomial learns each class from its own clauses, so small classes get poor estimates and get pulled towards big ones. Complement learns each class from all the *other* classes' clauses, so every class gets a big pool, then picks the class the clause looks *least* like the rest of. That suits LEDGAR's 138 : 1 imbalance."

### Same data, same clause, two answers

| | MultinomialNB | ComplementNB |
|---|---|---|
| learns from | each class's **own** clauses | **all other** classes' clauses |
| Waivers estimated from | 1.22 of TF-IDF | 5.16 of TF-IDF |
| uses prior? | yes (Waivers penalised for being rare) | no |
| picks | class it fits **best** | class whose complement it fits **worst** |
| prediction | Governing Laws ✗ | **Waivers ✓** |

Checked with scikit-learn: `MultinomialNB(alpha=1)` → Governing Laws, `ComplementNB(alpha=1)` → Waivers.

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB

X = np.array([[.99, 0, 0, .16], [.97, 0, 0, .24], [.12, .98, 0, .12],
              [0, 0, .78, .62], [0, 0, .62, .78]])          # the TF-IDF table above
y = ["Notices", "Notices", "Waivers", "Gov Laws", "Gov Laws"]
new = np.array([[0, .71, 0, .71]])                         # "...waive... law..."

print(MultinomialNB(alpha=1).fit(X, y).predict(new))       # ['Gov Laws']
print(ComplementNB(alpha=1).fit(X, y).predict(new))        # ['Waivers']
```

**Use when:** text with **imbalanced** classes.

## 5. GaussianNB: bell curves (why it fails on TF-IDF)

> **What you need to know**
> - features are **continuous** numbers; each class/feature gets a **bell curve** (mean + spread)
> - **use for:** bell-shaped measurements (sensors, lab values, sizes)
> - **wrong for TF-IDF:** columns are ~99.9% zeros, nothing like a bell curve; and it can't take sparse matrices (a dense LEDGAR `x_train` ≈ 100 GB)
>
> **Skip:** the bell-curve (normal distribution) formula

```
            height   weight   age
person 1     172      68      34
person 2     158      55      29
```

**Likelihood:** for each class and feature, learn a **mean** and **spread**, then ask how likely the value is on that bell curve ([Distributions](#/t/distributions)).

**On TF-IDF:** the "waive" column for Notices is `0.00, 0.00`: mean 0, spread 0. A bell curve with zero spread is a spike, so any non-zero "waive" looks "impossible" for Notices.

## Quick chooser

| Your features look like… | Use |
|---|---|
| categories | CategoricalNB |
| word counts / TF-IDF, balanced classes | MultinomialNB |
| word counts / TF-IDF, **imbalanced classes** | **ComplementNB** |
| yes/no, short texts | BernoulliNB |
| continuous, bell-shaped measurements | GaussianNB |

## In my LEDGAR project

In [Legal Clause NLP](#/p/legal) I've added a ComplementNB model next to the softmax logistic regression, on the same saved TF-IDF features so the comparison is fair ([src/complement_nb.py](https://github.com/jag5m33/py_legal_data/blob/main/src/complement_nb.py)):

```python
class ComplementNaiveBayes:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, alpha=1.0, norm=False):
        ...
        self.model = ComplementNB(alpha=alpha, norm=norm)

    def train(self):
        self.model.fit(self.x_train, self.y_train)
```

**Plan for `model_nb.ipynb`:**

1. **MultinomialNB**: baseline, on the saved TF-IDF
2. **ComplementNB**: expected winner (imbalance)
3. tune `alpha` (0.01, 0.1, 1) on **validation macro F1** ([Classification metrics](#/t/clsmetrics))
4. compare the best with logistic regression; reuse `ErrorAnalysis` unchanged
5. write-up: Gaussian and Bernoulli "considered and ruled out because…" (reasons above)

**Results: coming soon.**

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Extremely fast: it's just counting | Independence assumption is almost never true |
| Works with little data; few settings | Its probabilities are too extreme: trust the ranking, not the number |
| Handles huge sparse text well | Overlapping n-grams ("governing", "governing law") get double-counted |
| Easy to explain | Usually less accurate than tuned logistic regression on big data |
| ComplementNB copes with imbalance | Multinomial / Complement need non-negative inputs (counts or TF-IDF) |

## Interview one-liners

- **Why Complement, not Gaussian?** TF-IDF is sparse (~99.9% zeros), nothing like a bell curve; Complement is built for count-like features and corrects Multinomial's bias towards big classes. LEDGAR has about 138 : 1 imbalance.
- **Why Complement over Multinomial?** Multinomial learns small classes from little text and penalises them with a small prior; Complement learns every class from all the *other* classes, so each gets a large pool.
- **When Bernoulli?** Yes/no features, e.g. whether a word appears at all; suits short texts.
- **What does `alpha` do?** Smoothing: stops an unseen word giving a class probability of zero.

## Quick reference

| Term | Meaning |
|---|---|
| Bayes' theorem | $P(A \mid B) = P(B \mid A)\,P(A)\,/\,P(B)$ |
| Prior / likelihood / posterior | how common the class is / how typical the features are for it / the answer after seeing the features |
| Marginal probability | P(B): how common the evidence is overall; same for every class, so dropped |
| Predict | argmax over classes of prior × likelihoods; normalise to get probabilities |
| Naive assumption | features independent within a class, so likelihoods multiply |
| Logs | add log-probabilities instead of multiplying tiny numbers |
| `alpha` | smoothing so unseen words don't give zero |
| MultinomialNB | learns each class from its own text; picks best fit |
| ComplementNB | learns each class from all other classes; picks worst fit to "not this class"; no prior |
| BernoulliNB / GaussianNB / CategoricalNB | yes/no features / bell curves / categories |
