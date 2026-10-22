# Multiclass (softmax)

Softmax regression extends logistic regression from two classes to K classes by giving every class its own linear score and turning the K scores into K probabilities that add up to 1.

## From binary to K classes

[Logistic regression](#/t/logreg) answers "class 1 or not?". With $K > 2$ classes (exactly one per example, see [What supervised means](#/t/supintro)) there are two standard ways to extend it.

| Approach | How it works | Notes |
|---|---|---|
| **One-vs-rest (OvR)** | Train K separate binary models: "class k vs everything else". Predict the class whose model gives the highest probability. | K independent fits; probabilities don't naturally sum to 1 |
| **Multinomial (softmax)** | One model with K score functions, trained together with one cost. | Probabilities sum to 1; usually better when classes compete |

sklearn's `LogisticRegression` uses the multinomial (softmax) form by default for multiclass targets with the `lbfgs` solver. One-vs-rest is available through `sklearn.multiclass.OneVsRestClassifier`.

## The softmax formula

**Step 1**, one linear score per class. Each class $j$ has its own weights $w_j$ and bias $b_j$:

$$
z_j = w_j \cdot x + b_j \qquad j = 1, \dots, K
$$

**Step 2**, exponentiate each score so every value is positive:

$$
e^{z_1}, \; e^{z_2}, \; \dots, \; e^{z_K}
$$

**Step 3**, divide by the total so they sum to 1:

$$
a_j = P(y = j \mid x) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}
$$

The predicted class is the one with the largest $a_j$ (equivalently the largest $z_j$).

Why $e^{z}$ and not just $z$? Scores can be negative, and probabilities cannot. The exponential makes every score positive, keeps the order (bigger score, bigger probability) and exaggerates gaps, so the top class stands out.

## Worked example: 3 classes

Let's say for one clause the three class scores are $z = [2.0, 1.0, 0.1]$.

| Class | z | e^z | e^z / 11.213 |
|---|---|---|---|
| A | 2.0 | 7.389 | **0.659** |
| B | 1.0 | 2.718 | 0.242 |
| C | 0.1 | 1.105 | 0.099 |
| Sum | | 11.213 | 1.000 |

![The three steps of softmax: raw scores, then e to the score, then divided by the total.](figures/softmax-steps.svg)

Read the figure left to right: the raw scores (blue) become all-positive numbers (gold), which are then rescaled to probabilities (teal) that sum to 1. Class A's score was only 1 more than B's, but its probability is almost three times bigger, because $e^{1} \approx 2.72$.

## The cost: cross-entropy

For one example whose true class is $y$, the loss is minus the log of the probability given to the **true** class:

$$
L = -\log a_y
$$

All the other classes' probabilities don't appear directly, but they are in the denominator, so pushing $a_y$ up pushes the others down.

Using the example above:

| If the true class is | Loss = −log(a_true) |
|---|---|
| A (a = 0.659) | 0.417 |
| B (a = 0.242) | 1.417 |
| C (a = 0.099) | 2.317 |

The cost $J$ is the average of $L$ over all training examples, plus a regularisation term. This is the same "confident and wrong → huge cost" shape as binary log loss.

## Numerical stability: subtract the max

$e^{1000}$ overflows to infinity in a computer, and infinity divided by infinity is `nan`. The fix: subtract the largest score from every score before exponentiating.

$$
\frac{e^{z_j - c}}{\sum_k e^{z_k - c}} = \frac{e^{-c}e^{z_j}}{e^{-c}\sum_k e^{z_k}} = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

The $e^{-c}$ cancels, so the answer is unchanged, but now the largest exponent is $e^{0} = 1$ and nothing overflows. Libraries do this for you.

```python
import numpy as np

def softmax(z):
    z = z - z.max()          # shift so the biggest score is 0 (answer unchanged)
    e = np.exp(z)            # all values now between 0 and 1: no overflow
    return e / e.sum()       # divide by the total so they sum to 1

print(softmax(np.array([2.0, 1.0, 0.1])))        # [0.659 0.242 0.099]
print(softmax(np.array([1000., 999., 998.])))    # [0.665 0.245 0.090], no nan
```

## With 2 classes, softmax is the sigmoid

Take $K = 2$ with scores $z_1$ and $z_2$. Divide top and bottom by $e^{z_1}$:

$$
a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = g(z_1 - z_2)
$$

That is the sigmoid of the difference in scores. Check: $z = [1.5, 0]$ gives softmax $a_1 = 0.818$, and $g(1.5) = 0.818$. So binary logistic regression is a special case of softmax.

## His project: LEDGAR, 100 clause types

In [py_legal_data](https://github.com/jag5m33/py_legal_data) he classifies contract clauses from SEC filings into 100 clause types (Governing Laws, Indemnifications, Confidentiality and so on). The pipeline is regex cleaning, spaCy lemmatisation, custom stop words, TF-IDF (unigrams to trigrams, 95,559 features, fitted on train only), then multinomial logistic regression with `C=1.0` and `class_weight="balanced"`.

His model class wraps sklearn so training and scoring live in one place ([multi_class_lr_softmax.py](https://github.com/jag5m33/py_legal_data/blob/main/src/multi_class_lr_softmax.py)):

```python
class MultiClassLogisticRegression:

    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test,
                 C=1.0, class_weight=None):
        ...
        self.model = LogisticRegression(C=C,
                                        solver="lbfgs",
                                        max_iter=1000,
                                        random_state=42,
                                        class_weight=class_weight,
                                        verbose= 2)

    def _score(self, x, y, name):
        preds = self.model.predict(x)
        scores = {
            "accuracy": accuracy_score(y, preds),
            "macro_f1": f1_score(y, preds, average="macro"),   # main score: every class counts equally
            "micro_f1": f1_score(y, preds, average="micro"),   # equals accuracy for single-label multiclass
        }
        ...
```

With 100 classes and 95,559 features, the model learns a $100 \times 95{,}559$ weight matrix: one row of weights per clause type.

### Results

| Split | Accuracy | Macro F1 | Micro F1 |
|---|---|---|---|
| Train | 0.899 | 0.887 | 0.899 |
| Validation | 0.839 | 0.780 | 0.839 |

![LEDGAR train vs validation scores: accuracy and micro F1 drop 6 points, macro F1 drops nearly 11.](figures/softmax-ledgar.svg)

In the figure, look at the middle pair: macro F1 falls the most from train to validation. Because macro F1 weights every clause type equally, this says the overfitting is concentrated in the **rare** classes. Accuracy and micro F1 are identical, as they always are for single-label multiclass (see [Classification metrics](#/t/clsmetrics)).

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

model = LogisticRegression(max_iter=1000)        # multinomial softmax by default with lbfgs
model.fit(X_train, y_train)

print(model.coef_.shape)                         # (3, 4): one weight row per class
print(model.predict_proba(X_val[:2]).round(3))   # each row sums to 1
print(f1_score(y_val, model.predict(X_val), average="macro"))
```

## Assumptions & caveats

- **Linear in the log-odds, per class.** Each class score is a straight-line function of the features. With TF-IDF this is usually fine; with tabular data, curved effects need engineered features.
- **Exactly one label per example.** Softmax forces probabilities to sum to 1, so it cannot say "both". For multi-label problems use one sigmoid per label instead.
- **Many classes with few examples.** Each class needs its own weight vector. With 23 examples (LEDGAR's "Books") against 95,559 features, those weights overfit, which is what the macro-F1 gap shows. Stronger regularisation (smaller `C`), merging classes or collecting more data help.
- **Class imbalance.** Without weighting, the big classes dominate the cost. `class_weight="balanced"` weights each class by $\frac{n_{\text{samples}}}{K \cdot n_{\text{class}}}$, so a rare class's errors count more. It can lower precision on rare classes and distort the probabilities. His next step is to compare it with no weighting on validation.
- **Near-synonym or noisy labels.** If two labels mean the same thing (Governing Laws / Applicable Laws), no model can separate them reliably. Check for conflicting duplicates in the EDA, and consider merging labels to measure how much error is label noise.
- **Multicollinearity.** As in logistic regression, correlated features make individual weights unstable. With TF-IDF n-grams this is everywhere ("governing", "governing law"), so read the weights as a rough guide, not exact effects. L2 regularisation keeps it stable.
- **Independence and leakage.** Duplicate clauses split across train and validation would inflate scores; his EDA checked that no test clause appears in train.
- **Scaling.** TF-IDF rows are already L2-normalised; for other numeric features, scale before regularising.

## Use cases

- Text classification with many categories: clause types, support-ticket routing, topic labelling.
- A fast, explainable baseline before trying transformers or boosted trees.
- Any problem with mutually exclusive classes where you want a probability for each.
- As the last layer of a neural network classifier (the same softmax and cross-entropy).

## Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| One model gives a coherent probability for every class | Linear boundaries between each pair of classes |
| Fast and scales to large sparse TF-IDF matrices | K × n weights: large memory with many classes and features |
| Weights per class are inspectable (top words per clause type) | Rare classes get poorly estimated weights |
| Convex cost, so reliable training | Cannot do multi-label |
| Reduces to the sigmoid for 2 classes | Struggles to separate near-synonym labels |

## Limitations

- Bag-of-words features ignore word order and context, so "not liable" and "liable" may look similar to the model.
- Performance is capped by label quality; on LEDGAR much of the error is inconsistent labels.
- Probabilities are spread over all K classes, so with 100 classes the top probability can be low even when the prediction is correct; don't apply a fixed 0.5 rule.
- Macro F1 on classes with 0 to 3 validation examples is very noisy (Books had none; Assigns had three).

> **Say it like this:** "Softmax gives each class its own linear score, exponentiates and normalises them into probabilities, and trains with cross-entropy, minus log of the true class's probability. On LEDGAR I used it with TF-IDF and balanced class weights, and reported macro F1 so rare clause types count."

## Quick reference

| Item | Formula / meaning |
|---|---|
| Class score | z_j = w_j·x + b_j |
| Softmax | a_j = e^(z_j) / Σ e^(z_k) |
| Prediction | class with the largest a_j |
| Loss | −log(a_true) |
| Stability | subtract max(z) before exp |
| K = 2 | softmax = sigmoid(z₁ − z₂) |
| OvR | K binary models |
| Multinomial | one joint model (sklearn default) |
| class_weight="balanced" | n / (K × n_class) |
