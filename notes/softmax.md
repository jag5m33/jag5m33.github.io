# Multiclass (softmax)

Softmax regression extends logistic regression from two classes to many by giving every class its own score and turning those scores into probabilities that add up to 1.

## From two classes to many

[Logistic regression](#/t/logreg) answers "class 1 or not?". When there are more than two classes (and each example has exactly one, see [What supervised means](#/t/supintro)), there are two common ways to extend it.

| Approach | How it works |
|---|---|
| **One-vs-rest (OvR)** | Train one yes/no model per class ("this class vs everything else"). Pick the class with the highest probability. |
| **Softmax (multinomial)** | One model with a score for every class, trained together. Probabilities add up to 1. |

sklearn's `LogisticRegression` uses softmax by default when the target has more than two classes.

## The softmax formula, step by step

**Step 1:** every class $j$ gets its own straight-line score, with its own weights $w_j$ and bias $b_j$:

$$
z_j = w_j \cdot x + b_j
$$

**Step 2:** turn each score into a probability:

$$
a_j = \frac{e^{z_j}}{e^{z_1} + e^{z_2} + \dots + e^{z_K}}
$$

$K$ is the number of classes and $a_j$ is the probability of class $j$. In words: take $e$ to the power of each score (this makes every value positive), then divide each by the total so they add up to 1.

The predicted class is the one with the biggest probability.

## Worked example: 3 classes

Let's say for one clause the three class scores are $z = [2.0, 1.0, 0.1]$.

| Class | z | e^z | e^z / 11.213 |
|---|---|---|---|
| A | 2.0 | 7.389 | **0.659** |
| B | 1.0 | 2.718 | 0.242 |
| C | 0.1 | 1.105 | 0.099 |
| Sum | | 11.213 | 1.000 |

![The three steps of softmax: raw scores, then e to the score, then divided by the total.](figures/softmax-steps.svg)

Read the figure left to right: raw scores (blue) become positive numbers (gold), then probabilities (teal) that add up to 1. Class A's score was only 1 more than B's, but its probability is almost three times bigger, because $e^{1} \approx 2.72$. The exponential exaggerates gaps, so the top class stands out.

## The cost: cross-entropy

For one example, the loss is minus the log of the probability the model gave to the **true** class:

$$
L = -\log a_{\text{true}}
$$

Using the example above:

| If the true class is | Loss = −log(a_true) |
|---|---|
| A (a = 0.659) | 0.417 |
| B (a = 0.242) | 1.417 |
| C (a = 0.099) | 2.317 |

Same idea as binary log loss: confident and wrong costs a lot. The cost $J$ is the average loss over all examples.

Two small facts worth knowing (no maths needed):

- With exactly 2 classes, softmax gives the same answer as the sigmoid. Binary logistic regression is a special case.
- Computers subtract the biggest score from every score before using $e$, so huge numbers don't overflow. The answer is unchanged; libraries do this for you.

## His project: LEDGAR, 100 clause types

In [py_legal_data](https://github.com/jag5m33/py_legal_data) he classifies contract clauses from SEC filings into 100 clause types (Governing Laws, Indemnifications, Confidentiality and so on). The pipeline is regex cleaning, spaCy lemmatisation, custom stop words, TF-IDF (unigrams to trigrams, 95,559 features, fitted on train only), then multinomial logistic regression with `C=1.0` and `class_weight="balanced"`.

His model class wraps sklearn so training and scoring live in one place ([multi_class_lr_softmax.py](https://github.com/jag5m33/py_legal_data/blob/main/src/multi_class_lr_softmax.py)):

```python
class MultiClassLogisticRegression:

    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test,
                 C=1.0, class_weight=None):
        ...
        self.model = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                        random_state=42, class_weight=class_weight)

    def _score(self, x, y, name):
        preds = self.model.predict(x)
        scores = {
            "accuracy": accuracy_score(y, preds),
            "macro_f1": f1_score(y, preds, average="macro"),   # main score: every class counts equally
            "micro_f1": f1_score(y, preds, average="micro"),   # equals accuracy here
        }
        ...
```

### Results

| Split | Accuracy | Macro F1 | Micro F1 |
|---|---|---|---|
| Train | 0.899 | 0.887 | 0.899 |
| Validation | 0.839 | 0.780 | 0.839 |

![LEDGAR train vs validation scores: accuracy and micro F1 drop 6 points, macro F1 drops nearly 11.](figures/softmax-ledgar.svg)

In the figure, look at the middle pair: macro F1 falls the most from train to validation. Macro F1 counts every clause type equally, so the overfitting is mostly in the **rare** classes. Accuracy and micro F1 are identical, as they always are when each example has one label (see [Classification metrics](#/t/clsmetrics)).

His error analysis found most mistakes are between near-synonym labels: Governing Laws predicted as Applicable Laws (73 times), Amendments as Modifications (25), Indemnifications as Indemnity (18). The weakest classes were Applicable Laws (F1 0.30), Jurisdictions (0.35) and Venues (0.36). The EDA had already found duplicate clauses with conflicting labels in exactly these pairs, so much of the remaining error is label noise rather than model error. The test set has not been scored yet.

## Code: sklearn multiclass

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X, y = load_iris(return_X_y=True)                 # 3 flower species, 4 features
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000)        # softmax by default for 3+ classes
model.fit(X_train, y_train)

print(model.predict_proba(X_val[:2]).round(3))   # each row adds up to 1
print(f1_score(y_val, model.predict(X_val), average="macro"))
```

## Assumptions & caveats

- **One label per example.** Probabilities must add up to 1, so it cannot say "both". For multi-label problems use one sigmoid per label.
- **Rare classes.** Each class needs its own weights. A class with 23 examples (LEDGAR's "Books") against 95,559 features will overfit, which is what the macro-F1 gap shows.
- **Class imbalance.** Without weighting, big classes dominate. `class_weight="balanced"` makes errors on rare classes count more. His next step is to compare it with no weighting on validation.
- **Near-synonym or noisy labels.** If two labels mean the same thing (Governing Laws / Applicable Laws), no model can separate them reliably.
- **Leakage.** Duplicate clauses split across train and validation would inflate scores; his EDA checked that no test clause appears in train.

## Use cases

- Text classification with many categories: clause types, support-ticket routing, topic labels.
- A fast, explainable baseline before trying bigger models.
- The last layer of a neural network classifier uses the same softmax.

## Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| One model gives a probability for every class | Straight-line boundaries only |
| Fast, handles large sparse TF-IDF data | Rare classes get poor weights |
| Top words per class are easy to inspect | Cannot do multi-label |
| Reliable training | Struggles with near-synonym labels |

## Limitations

- Bag-of-words features ignore word order, so "not liable" and "liable" can look similar.
- Performance is capped by label quality.
- With 100 classes the top probability can be low even when correct, so don't use a fixed 0.5 rule.

> **Say it like this:** "Softmax gives each class its own linear score, turns the scores into probabilities that add up to 1, and trains with cross-entropy: minus log of the true class's probability. On LEDGAR I used it with TF-IDF and balanced class weights, and reported macro F1 so rare clause types count."

## Quick reference

| Item | Formula / meaning |
|---|---|
| Class score | z_j = w_j·x + b_j |
| Softmax | a_j = e^(z_j) / sum of all e^(z_k) |
| Prediction | class with the biggest a_j |
| Loss | −log(a_true) |
| 2 classes | softmax = sigmoid |
| OvR | one yes/no model per class |
