# Logistic regression

Logistic regression is a classification model: it computes a straight-line score, squashes it into a probability with the sigmoid, and predicts a class by comparing that probability with a threshold.

## Is it regression or classification?

**Classification.** The name is historical. What it "regresses" is the **log-odds** of the positive class, a number that can be anything from $-\infty$ to $+\infty$:

$$
\log\frac{p}{1-p} = w \cdot x + b
$$

That number is turned into a probability $p$ between 0 and 1, and then a threshold (normally 0.5) turns the probability into a class, 0 or 1. So the output you use is a category.

- **Odds**: $p/(1-p)$. A probability of 0.8 is odds of 4 (4 to 1 on).
- **Log-odds (logit)**: the natural log of the odds. See [Logs & exponents](#/t/logs).

## Why not just use linear regression?

Let's say we have 8 tumours, sizes 1 to 9, with the small ones benign ($y=0$) and the large ones malignant ($y=1$). Fit a straight line and call anything above 0.5 malignant.

![Left: a line through 8 tumours crosses 0.5 at size 5, which separates the classes. Right: adding one very large malignant tumour flattens the line and pushes the cut-off to 6.4, so the size-6 tumour is now called benign.](figures/logreg-whynotlinear.svg)

In the figure, the blue dashed line is where the fitted line crosses 0.5.

- **Problem 1: one extreme point moves the threshold.** Adding a size-40 tumour, which is obviously malignant and should change nothing, tilts the line and moves the cut-off from 5.0 to 6.4. The size-6 malignant tumour is now misclassified.
- **Problem 2: outputs are not probabilities.** The line goes above 1 and below 0, and "probability 1.3" means nothing.

Logistic regression fixes both by bending the output into an S-shape that flattens out at 0 and 1.

## The sigmoid function

**Step 1**, compute the linear score (as in [linear regression](#/t/linreg)):

$$
z = w \cdot x + b = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

**Step 2**, squash it with the **sigmoid** (also called the logistic function):

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

So the model is $f_{w,b}(x) = g(w \cdot x + b)$, read as "the probability that $y = 1$ given $x$".

![The sigmoid S-curve: 0.5 at z = 0, near 1 for large positive z, near 0 for large negative z.](figures/logreg-sigmoid.svg)

Look at three points on the curve:

- $z = 0$: $e^{0} = 1$, so $g = 1/2 = 0.5$ (the gold dot).
- $z$ large and positive: $e^{-z} \to 0$, so $g \to 1$.
- $z$ large and negative: $e^{-z}$ is huge, so $g \to 0$.

**Where it comes from:** solve the log-odds equation $\log\frac{p}{1-p} = z$ for $p$. Exponentiate: $\frac{p}{1-p} = e^{z}$, so $p = e^{z}(1-p)$, so $p(1+e^z) = e^z$, so $p = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}$. The sigmoid is just the inverse of the log-odds.

## Decision boundary

We predict $\hat{y} = 1$ when $f(x) \ge 0.5$. Because $g(z) \ge 0.5$ exactly when $z \ge 0$, this is the same as:

$$
\hat{y} = 1 \iff w \cdot x + b \ge 0
$$

The **decision boundary** is the set of points where $z = 0$: the model is exactly 50/50 there.

### Worked example: w = [1, 1], b = −3

$z = x_1 + x_2 - 3$, so the boundary is the line $x_1 + x_2 = 3$.

| Point (x₁, x₂) | z = x₁ + x₂ − 3 | g(z) | Prediction |
|---|---|---|---|
| (1, 1) | −1 | 0.269 | 0 |
| (1, 2) | 0 | 0.500 | on the boundary |
| (2, 2) | 1 | 0.731 | 1 |
| (3, 1) | 1 | 0.731 | 1 |

### Non-linear boundaries from new features

Logistic regression is linear in its **features**, but you choose the features. Add $x_1^2$ and $x_2^2$ as features, with $w = [1, 1]$ on them and $b = -1$:

$$
z = x_1^2 + x_2^2 - 1 = 0 \iff x_1^2 + x_2^2 = 1
$$

which is a circle of radius 1.

![Left: w = [1, 1], b = −3 gives the straight boundary x₁ + x₂ = 3. Right: squared features give a circular boundary.](figures/logreg-boundaries.svg)

In the left panel, everything above the gold line has $z \ge 0$ and is predicted 1. In the right panel, everything outside the circle is predicted 1. Higher-order terms give ellipses and wigglier shapes, at the cost of more overfitting risk.

## The cost function

### Why not squared error?

For linear regression we minimised the mean squared error. If we plug the sigmoid into that same formula:

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(g(w\cdot x^{(i)}+b) - y^{(i)}\right)^2
$$

the cost surface is **not convex**: it has flat plateaus and can have local minima, so gradient descent can stall or get stuck.

![Left: squared error with a sigmoid, flat and wavy. Right: log loss on the same data, a single smooth bowl.](figures/logreg-convex.svg)

Both panels use the same 6 data points and only vary the weight $w$. In the left panel, look at the flat shelves on either side: the slope there is nearly zero, so gradient descent barely moves. The right panel is the log-loss cost: one bowl, one minimum.

### The loss for one example

Take it case by case, the way you would by hand. $f$ is the predicted probability that $y = 1$.

**If the true label is $y = 1$:** loss $= -\log(f)$.
- Model says $f = 0.99$: loss $= -\log(0.99) = 0.01$. Confident and right, almost no cost.
- Model says $f = 0.5$: loss $= 0.693$.
- Model says $f = 0.01$: loss $= 4.61$. Confident and wrong, big cost.

**If the true label is $y = 0$:** loss $= -\log(1 - f)$. Mirror image: now $f$ near 1 is the disaster.

![Left: when y = 1 the loss −log(f) explodes as f goes to 0. Right: when y = 0 the loss −log(1 − f) explodes as f goes to 1.](figures/logreg-logloss.svg)

In both panels the arrows mark the same idea: confident and right costs almost nothing; confident and wrong costs a lot, heading to infinity. The log is what gives this steep wall.

### The combined formula, and why it is written that way

$$
L(f, y) = -\,y\log(f) \;-\; (1-y)\log(1-f)
$$

The $y$ and $1-y$ act as on/off switches, because $y$ is only ever 0 or 1:

- Plug in $y = 1$: $L = -1\cdot\log(f) - 0\cdot\log(1-f) = -\log(f)$. The second term switches off.
- Plug in $y = 0$: $L = -0\cdot\log(f) - 1\cdot\log(1-f) = -\log(1-f)$. The first term switches off.

So it is just the two cases written as one line, which makes it easy to vectorise and differentiate. The **cost** is the average loss over all $m$ examples:

$$
J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log f^{(i)} + (1-y^{(i)})\log(1-f^{(i)})\right]
$$

This is also called **log loss** or **binary cross-entropy**.

### Worked example: 4 predictions

| Example | True y | Predicted f | Which term is on | Loss |
|---|---|---|---|---|
| 1 | 1 | 0.9 | −log(0.9) | 0.105 |
| 2 | 1 | 0.3 | −log(0.3) | 1.204 |
| 3 | 0 | 0.2 | −log(1 − 0.2) = −log(0.8) | 0.223 |
| 4 | 0 | 0.8 | −log(1 − 0.8) = −log(0.2) | 1.609 |
| | | | **Mean (the cost J)** | **0.785** |

Examples 2 and 4 are the wrong-side predictions and contribute most of the cost. (Checked with `sklearn.metrics.log_loss([1,1,0,0], [0.9,0.3,0.2,0.8])`, which gives 0.7855.)

### Where it comes from: maximum likelihood

If the model says $P(y=1) = f$, then the probability it assigns to the label actually observed is $f^{y}(1-f)^{1-y}$ (which is $f$ when $y=1$ and $1-f$ when $y=0$). Multiplying this over all examples gives the **likelihood**; taking the log turns the product into a sum; putting a minus sign in front turns "maximise likelihood" into "minimise cost". The result is exactly $J$ above. So log loss is not an arbitrary choice: it is the cost that makes the parameters the most likely ones given the data.

## Gradient descent

Differentiating $J$ gives (the derivation uses $g'(z) = g(z)(1-g(z))$, which cancels neatly):

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)}) - y^{(i)}\right)x_j^{(i)}
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)}) - y^{(i)}\right)
$$

Update each step, all parameters at the same time, with learning rate $\alpha$:

$$
w_j := w_j - \alpha\frac{\partial J}{\partial w_j} \qquad b := b - \alpha\frac{\partial J}{\partial b}
$$

> **Trap:** these look identical to the linear regression gradients, but they are not the same algorithm, because $f$ is now $g(w\cdot x + b)$ rather than $w\cdot x + b$.

## L2 regularisation (ridge)

Add a penalty on the size of the weights to the cost:

$$
J(w,b) = \text{log loss} + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2
$$

Large weights are expensive, so all weights shrink towards zero but rarely reach it. This smooths the decision boundary and reduces overfitting. The gradient gains an extra $\frac{\lambda}{m}w_j$ term. $b$ is not penalised. This is sklearn's default.

## L1 regularisation (lasso)

Penalise absolute values instead:

$$
J(w,b) = \text{log loss} + \frac{\lambda}{m}\sum_{j=1}^{n} \lvert w_j \rvert
$$

L1 pushes some weights to **exactly zero**, so it also selects features. Useful with many features, such as TF-IDF vocabularies. Needs a solver that supports it (`liblinear` or `saga`).

For why these shapes behave differently, and how to choose $\lambda$, see [Regularisation](#/t/regularisation).

> **Key idea:** in sklearn, `C` is the **inverse** of the regularisation strength, $C = 1/\lambda$. Small `C` means strong regularisation (simpler model); large `C` means weak regularisation (fits training data more closely). His own note in the LEDGAR model file says the same: a high C risks overfitting, a low C keeps coefficients small and generalises better.

## Threshold and predict_proba

- `predict_proba(X)[:, 1]` gives $f(x)$, the probability of class 1.
- `predict(X)` applies a threshold of 0.5 for you.
- You can pick your own threshold: lower it to catch more positives (higher recall, more false alarms), raise it for fewer false alarms (higher precision). Choose it on the **validation** set from the precision-recall curve; see [Classification metrics](#/t/clsmetrics).

His accidents project notes exactly this as a next step: both models use 0.5, and the plan is to pick a threshold based on the cost of a missed severe accident against a false alarm.

## Code: sklearn

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
import numpy as np

# toy data with ~30% positives, split 60/20/20 as in "What supervised means"
X, y = make_classification(n_samples=1000, n_features=5, weights=[0.7, 0.3], random_state=42)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

model = make_pipeline(
    StandardScaler(),                 # scale features so the penalty treats them equally
    LogisticRegression(
        C=1.0,                        # C = 1/lambda: smaller C = stronger regularisation
        solver="lbfgs",               # default optimiser; supports L2 only
        class_weight="balanced",      # up-weight the rarer class
        max_iter=1000,                # raise if you see a convergence warning
    ),
)
model.fit(X_train, y_train)          # scaler and model learn from train only

p_val = model.predict_proba(X_val)[:, 1]          # P(y = 1) for each validation row
pred_05 = (p_val >= 0.5).astype(int)               # the default threshold
pred_03 = (p_val >= 0.3).astype(int)               # a lower threshold: more recall
print("F1 at 0.5:", f1_score(y_val, pred_05))
print("F1 at 0.3:", f1_score(y_val, pred_03))
print("log loss :", log_loss(y_val, p_val))

coefs = model[-1].coef_[0]                         # one weight per (scaled) feature
print("odds ratios:", np.exp(coefs).round(2))      # e^w, see below
```

For L1 in older sklearn: `LogisticRegression(penalty="l1", solver="liblinear", C=0.5)`. From sklearn 1.8 the `penalty` argument is deprecated in favour of `l1_ratio` (`l1_ratio=1` for L1, `0` for L2, with `solver="liblinear"` or `"saga"` for L1), so check your installed version.

## Interpreting coefficients: odds ratios

Each weight $w_j$ is the change in **log-odds** for a one-unit increase in $x_j$, holding the other features fixed. Exponentiate it to get an **odds ratio**:

- $w_j = 0.7$: $e^{0.7} = 2.01$, so each extra unit roughly doubles the odds of $y = 1$.
- $w_j = -0.5$: $e^{-0.5} = 0.61$, so each extra unit cuts the odds by about 39%.
- $w_j = 0$: $e^0 = 1$, no effect.

> **Trap:** it multiplies the **odds**, not the probability. Doubling odds of 1 (p = 0.5) gives odds 2 (p = 0.67), not p = 1.0. And after scaling, "one unit" means one standard deviation.

## Assumptions & caveats

- **Linear in the log-odds.** The log-odds must be a straight-line function of the features. If the true effect is curved (risk rises then falls with age), add squared terms, splines or bins.
- **Independent observations.** Each row should be a separate case. Repeated measurements of the same patient, or near-duplicate rows, make the model over-confident and leak across splits (his LEDGAR EDA removed 64 duplicate clauses for this reason).
- **Little multicollinearity.** If two features carry nearly the same information (temperature and wind chill), the model cannot tell which deserves the weight. Predictions may be fine, but coefficients become unstable, can flip sign and have huge standard errors, so odds ratios become meaningless. Check correlations or the **variance inflation factor**, $\text{VIF}_j = 1/(1-R_j^2)$ where $R_j^2$ comes from regressing feature $j$ on the others; above roughly 5 to 10 is a warning. Drop one of the pair (he dropped wind chill for this reason) or use L2 regularisation.
- **No perfect separation.** If one feature splits the classes perfectly, the best weight is infinite: the optimiser keeps increasing it and never converges. Regularisation (the default in sklearn) stops this.
- **Enough data per feature.** With few examples and many features the weights overfit. A common rule of thumb is at least 10 to 20 examples of the rarer class per feature for interpretable models.
- **Scaling.** Not needed for the unregularised fit, but needed with regularisation, because the penalty compares raw weight sizes, and for faster convergence.
- **Class imbalance.** With 1% positives the model can push all probabilities low. Use `class_weight="balanced"`, choose a threshold on validation, and judge with precision/recall/PR-AUC, not accuracy.
- **Calibration.** Plain logistic regression is usually well calibrated (a predicted 0.7 happens about 70% of the time). `class_weight` and strong regularisation distort this, so the probabilities may need recalibrating (`CalibratedClassifierCV`) if you use them as real risks.
- **Outliers in X.** Less damaging than in linear regression (the sigmoid caps the output), but extreme feature values can still dominate a weight.

## Use cases

- A strong, fast **baseline** for any binary classification problem.
- When you need to **explain** the effect of each feature (medicine, credit risk, policy), via odds ratios.
- When you need **probabilities**, not just labels (risk scores, ranking leads).
- High-dimensional sparse data such as **text with TF-IDF**, where linear models do very well (his LEDGAR model, extended to [softmax](#/t/softmax)).

## Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| Outputs probabilities, usually well calibrated | Only a linear boundary unless you engineer features |
| Fast to train and to predict, even on millions of rows | Misses interactions unless you add them |
| Coefficients are interpretable as odds ratios | Coefficients unstable under multicollinearity |
| Convex cost: one global minimum, reliable training | Needs scaling when regularised |
| Regularisation (L1/L2) built in | Sensitive to imbalance without weighting or threshold tuning |
| Works well on sparse, high-dimensional text | Usually beaten by trees/boosting on tabular data with complex interactions |

## Limitations

- It cannot learn curved or interacting effects on its own; you must supply them as features, and you have to know which ones.
- Odds-ratio interpretation assumes the other features are held fixed, which is unrealistic when features are correlated.
- A good AUC does not mean a good default threshold: 0.5 is rarely the best cut-off on imbalanced data.
- Multiclass needs an extension (one-vs-rest or [softmax](#/t/softmax)).
- It describes association, not causation.

> **Say it like this:** "Logistic regression is a linear classifier. It models the log-odds as $w\cdot x + b$, turns that into a probability with the sigmoid, and is trained by minimising log loss, which comes from maximum likelihood and is convex. I'd threshold on validation, regularise with C, and read $e^{w}$ as an odds ratio, checking VIF first."

## Quick reference

| Item | Formula / meaning |
|---|---|
| Linear score | z = w·x + b |
| Sigmoid | g(z) = 1 / (1 + e^−z), g(0) = 0.5 |
| Model | f(x) = g(w·x + b) = P(y = 1 given x) |
| Log-odds | log(p / (1 − p)) = w·x + b |
| Decision boundary | w·x + b = 0 |
| Loss, y = 1 | −log(f) |
| Loss, y = 0 | −log(1 − f) |
| Cost | mean of −[y log f + (1 − y) log(1 − f)] |
| Gradient | (1/m) Σ (f − y) x_j |
| L2 / L1 penalty | (λ/2m) Σ w² / (λ/m) Σ abs(w) |
| sklearn C | 1 / λ |
| Odds ratio | e^w per one-unit increase |
| VIF | 1 / (1 − R²_j), worry above 5 to 10 |
