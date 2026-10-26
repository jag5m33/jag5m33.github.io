# Logistic regression

Logistic regression is a classification model: it computes a straight-line score, squashes it into a probability with the sigmoid, and predicts a class by comparing that probability with a threshold.

## Regression or classification?

**Classification.** The name is historical. The model outputs a **probability** between 0 and 1, and a **threshold** (normally 0.5) turns that probability into a class, 0 or 1.

## Why not just use linear regression?

Let's say we have 8 tumours, sizes 1 to 9, with the small ones benign ($y=0$) and the large ones malignant ($y=1$). Fit a straight line and call anything above 0.5 malignant.

![Left: a line through 8 tumours crosses 0.5 at size 5, which separates the classes. Right: adding one very large malignant tumour flattens the line and pushes the cut-off to 6.4, so the size-6 tumour is now called benign.](figures/logreg-whynotlinear.svg)

In the figure, the blue dashed line is where the fitted line crosses 0.5.

- **Problem 1: one extreme point moves the cut-off.** Adding a size-40 tumour (obviously malignant) tilts the line and moves the cut-off from 5.0 to 6.4. The size-6 malignant tumour is now wrong.
- **Problem 2: outputs are not probabilities.** The line goes above 1 and below 0. "Probability 1.3" means nothing.

Logistic regression fixes both by bending the output into an S-shape that stays between 0 and 1.

## The sigmoid function

**Step 1:** compute the straight-line score, as in [linear regression](#/t/linreg):

$$
z = w \cdot x + b
$$

$w$ are the weights (one per feature), $x$ the features, $b$ the bias (intercept).

**Step 2:** squash it with the **sigmoid**:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

$e$ is the number 2.718…; $g(z)$ is the predicted probability that $y = 1$.

![The sigmoid S-curve: 0.5 at z = 0, near 1 for large positive z, near 0 for large negative z.](figures/logreg-sigmoid.svg)

Look at three points on the curve:

- $z = 0$: $e^{0} = 1$, so $g = 1/2 = 0.5$ (the gold dot).
- $z = 2$: $g = 1/(1 + 0.135) = 0.88$. Big positive score, probability near 1.
- $z = -2$: $g = 1/(1 + 7.39) = 0.12$. Big negative score, probability near 0.

## Decision boundary

We predict class 1 when $g(z) \ge 0.5$. Since $g(z) = 0.5$ exactly at $z = 0$, that is the same as saying **predict 1 when $z \ge 0$**.

The **decision boundary** is the line where $z = 0$: the model is exactly 50/50 there.

### Worked example: w = [1, 1], b = −3

$z = x_1 + x_2 - 3$, so the boundary is the line $x_1 + x_2 = 3$.

| Point (x₁, x₂) | z = x₁ + x₂ − 3 | g(z) | Prediction |
|---|---|---|---|
| (1, 1) | −1 | 0.269 | 0 |
| (1, 2) | 0 | 0.500 | on the boundary |
| (2, 2) | 1 | 0.731 | 1 |

![Left: w = [1, 1], b = −3 gives the straight boundary x₁ + x₂ = 3. Right: squared features give a circular boundary.](figures/logreg-boundaries.svg)

In the left panel, everything above the gold line is predicted 1. The right panel shows a trick: if you add squared features ($x_1^2$, $x_2^2$), the boundary can become a circle. The model is still "linear", but in the features you give it.

## The cost function: log loss

We need a way to score how wrong the predicted probabilities are. Squared error (from linear regression) works badly here: with the sigmoid inside, its cost surface is bumpy, so gradient descent can get stuck. Log loss gives one smooth bowl instead.

Take it case by case. $f$ is the predicted probability that $y = 1$.

**If the true label is $y = 1$:** loss $= -\log(f)$.
- $f = 0.99$: loss $= 0.01$. Confident and right, almost no cost.
- $f = 0.5$: loss $= 0.693$.
- $f = 0.01$: loss $= 4.61$. Confident and wrong, big cost.

**If the true label is $y = 0$:** loss $= -\log(1 - f)$. Mirror image: now $f$ near 1 is the disaster.

![Left: when y = 1 the loss −log(f) explodes as f goes to 0. Right: when y = 0 the loss −log(1 − f) explodes as f goes to 1.](figures/logreg-logloss.svg)

In both panels: confident and right costs almost nothing; confident and wrong costs a lot.

### The two cases in one line

$$
L(f, y) = -\,y\log(f) \;-\; (1-y)\log(1-f)
$$

$y$ is only ever 0 or 1, so it works like an on/off switch:

- $y = 1$: the second term is multiplied by 0 and disappears, leaving $-\log(f)$.
- $y = 0$: the first term disappears, leaving $-\log(1-f)$.

The **cost** $J$ is just the average loss over all examples. It is also called **binary cross-entropy**.

### Worked example: 4 predictions

| Example | True y | Predicted f | Which term is on | Loss |
|---|---|---|---|---|
| 1 | 1 | 0.9 | −log(0.9) | 0.105 |
| 2 | 1 | 0.3 | −log(0.3) | 1.204 |
| 3 | 0 | 0.2 | −log(0.8) | 0.223 |
| 4 | 0 | 0.8 | −log(0.2) | 1.609 |
| | | | **Mean (the cost J)** | **0.785** |

Examples 2 and 4 are on the wrong side and make up most of the cost. (`sklearn.metrics.log_loss([1,1,0,0], [0.9,0.3,0.2,0.8])` gives 0.7855.)

Training uses **gradient descent**, exactly as in linear regression: nudge $w$ and $b$ a little at a time in the direction that lowers $J$.

## Regularisation: L1 and L2

To stop overfitting, add a penalty for big weights to the cost:

- **L2 (ridge):** adds $\lambda \sum w_j^2$. Shrinks all weights towards zero. This is sklearn's default.
- **L1 (lasso):** adds $\lambda \sum \lvert w_j \rvert$. Pushes some weights to exactly zero, so it also drops features.

$\lambda$ (lambda) controls how strong the penalty is. Full explanation and the worked example are in [Regularisation](#/t/regularisation).

> **Key idea:** in sklearn, `C` is the **inverse** of the penalty strength, $C = 1/\lambda$. Small `C` = strong regularisation (simpler model). Large `C` = weak regularisation (fits the training data more closely). His own note in the LEDGAR model file says the same: a high C risks overfitting, a low C keeps coefficients small and generalises better.

## Threshold and predict_proba

- `predict_proba(X)[:, 1]` gives the probability of class 1.
- `predict(X)` applies a 0.5 threshold for you.
- Lower the threshold to catch more positives (more false alarms); raise it for fewer false alarms (more misses). Choose it on the **validation** set; see [Classification metrics](#/t/clsmetrics).

His accidents project notes exactly this as a next step: both models use 0.5, and the plan is to pick a threshold based on the cost of a missed severe accident against a false alarm.

## Code: sklearn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

model = make_pipeline(
    StandardScaler(),                 # scale features so the penalty treats them equally
    LogisticRegression(
        C=1.0,                        # smaller C = stronger regularisation
        class_weight="balanced",      # give the rarer class more weight
        max_iter=1000,                # raise if you see a convergence warning
    ),
)
model.fit(X_train, y_train)                    # learn from the training set only
p_val = model.predict_proba(X_val)[:, 1]       # probability of class 1
pred = (p_val >= 0.3).astype(int)              # custom threshold of 0.3
print("F1:", f1_score(y_val, pred))
```

## Assumptions & caveats

- **Straight-line relationship.** The score $z$ is a straight-line combination of the features. Curved effects need extra features (squares, bins).
- **Independent rows.** Each row should be a separate case. Near-duplicates make the model over-confident and leak across splits (his LEDGAR EDA removed 64 duplicate clauses for this reason).
- **Multicollinearity.** If two features carry nearly the same information (temperature and wind chill), the weights become unstable and can flip sign. VIF is a number that flags this; above about 5 to 10 is a warning. He dropped wind chill for this reason.
- **Scaling.** Needed when regularising, so the penalty treats features fairly.
- **Class imbalance.** With few positives, use `class_weight="balanced"`, tune the threshold, and judge with precision/recall, not accuracy.

## Use cases

- A fast, strong **baseline** for any yes/no problem.
- When you need to **explain** each feature's effect (medicine, credit risk).
- When you need **probabilities**, not just labels (risk scores).
- **Text with TF-IDF**, where linear models do very well (his LEDGAR model, extended to [softmax](#/t/softmax)).

## Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| Outputs probabilities | Only a straight boundary unless you add features |
| Fast to train, even on millions of rows | Misses interactions unless you add them |
| Weights are easy to interpret | Weights unstable with multicollinearity |
| Smooth cost, reliable training | Needs scaling when regularised |
| Works well on sparse text data | Usually beaten by tree models on complex tabular data |

## Limitations

- It cannot learn curves or interactions on its own; you must add them as features.
- 0.5 is rarely the best threshold on imbalanced data.
- Multiclass needs an extension ([softmax](#/t/softmax)).
- It shows association, not cause and effect.

> **Say it like this:** "Logistic regression is a linear classifier. It computes $w\cdot x + b$, turns it into a probability with the sigmoid, and is trained by minimising log loss with gradient descent. I'd regularise with C and choose the threshold on the validation set."

## Quick reference

| Item | Formula / meaning |
|---|---|
| Linear score | z = w·x + b |
| Sigmoid | g(z) = 1 / (1 + e^−z), g(0) = 0.5 |
| Decision boundary | z = 0 |
| Loss, y = 1 | −log(f) |
| Loss, y = 0 | −log(1 − f) |
| Cost J | average loss over all examples |
| L2 / L1 penalty | λ Σ w² / λ Σ abs(w) |
| sklearn C | 1 / λ (small C = strong penalty) |
