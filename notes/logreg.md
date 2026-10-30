# Logistic regression

Logistic regression is a classification model: it computes a straight-line score, squashes it into a probability between 0 and 1 with the sigmoid, and predicts a class by comparing that probability with a threshold.

## The running example

8 tumours, sizes 1 to 9. The small ones are benign ($y = 0$), the large ones malignant ($y = 1$).

```
size x   1   2   3   4   6   7   8   9
y        0   0   0   0   1   1   1   1
```

Question: a new tumour is size 6. Malignant or not, and how sure are we?

## 1. Why not linear regression?

Fit a straight line and call anything above 0.5 malignant:

![Left: a line through 8 tumours crosses 0.5 at size 5, which separates the classes. Right: adding one very large malignant tumour flattens the line and pushes the cut-off to 6.4, so the size-6 tumour is now called benign.](figures/logreg-whynotlinear.svg)

In the figure, the blue dashed line is where the fitted line crosses 0.5.

- **One extreme point moves the cut-off.** A size-40 tumour tilts the line and moves the cut-off from 5.0 to 6.4: the size-6 tumour is now called benign.
- **Not probabilities.** The line goes above 1 and below 0. "Probability 1.3" means nothing.

> **Key idea:** despite the name, logistic regression is a **classification** model. It outputs a probability, then a threshold turns it into a class, 0 or 1.

## 2. The sigmoid

> **What you need to know**
> - straight-line score, then the **sigmoid** squashes it to between 0 and 1
> - $g(0) = 0.5$; big positive $z$ → near 1; big negative $z$ → near 0
>
> **Skip:** why $e$ is in it

**Step 1:** the straight-line score, as in [linear regression](#/t/linreg):

$$
z = \theta_0 + \theta_1 x
$$

**Notation:** Krish Naik writes $\theta$, Andrew Ng writes $w$ and $b$: $\theta_0 = b$ (intercept), $\theta_1 = w$ (slope). With more features, $z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots$

**Step 2:** squash it with the **sigmoid** $g$:

$$
h_\theta(x) = g(z) = \frac{1}{1 + e^{-z}}
$$

$e$ = 2.718…; $h_\theta(x)$ = the **hypothesis**: the predicted probability that $y = 1$.

![The sigmoid S-curve: 0.5 at z = 0, near 1 for large positive z, near 0 for large negative z.](figures/logreg-sigmoid.svg)

In the figure, the gold dot is $z = 0$, where the curve is exactly 0.5.

**Worked example.** Let's say training gave $\theta_0 = -5$, $\theta_1 = 1$, so $z = x - 5$:

| Size x | z = x − 5 | hθ(x) | Prediction |
|---|---|---|---|
| 2 | −3 | 0.047 | benign |
| 4 | −1 | 0.269 | benign |
| 5 | 0 | 0.500 | on the boundary |
| **6** | **1** | **0.731** | **malignant** |
| 8 | 3 | 0.953 | malignant |

So the size-6 tumour: **73% likely malignant**. (sklearn's own fit on these 8 tumours gives $\theta_0 = -5.68$, $\theta_1 = 1.14$: the same boundary at size 5.)

## 3. Decision boundary

> **What you need to know**
> - predict 1 when $h_\theta(x) \ge 0.5$, which is the same as **$z \ge 0$**
> - the **decision boundary** is where $z = 0$: the model is 50/50 there
> - it's a **straight line**; add squared features and it can curve

In the running example $z = x - 5 = 0$ at **size 5**: bigger is malignant, smaller benign.

With two features, $\theta_0 = -3$, $\theta_1 = \theta_2 = 1$ gives $z = x_1 + x_2 - 3$, so the boundary is the line $x_1 + x_2 = 3$:

![Left: θ0 = −3, θ1 = θ2 = 1 gives the straight boundary x₁ + x₂ = 3. Right: squared features give a circular boundary.](figures/logreg-boundaries.svg)

Left: everything above the gold line is predicted 1. Right: squared features ($x_1^2$, $x_2^2$) bend it into a circle.

## 4. Cost function: log loss

> **What you need to know**
> - **confident and right** costs almost nothing; **confident and wrong** costs a lot
> - cost $J$ = average loss over all examples (also called **binary cross-entropy**)
> - not squared error: with the sigmoid inside it gives a **bumpy** cost where gradient descent gets stuck; log loss is a smooth bowl
>
> **Skip:** where log loss comes from

**Case $y = 1$:** loss $= -\log(h_\theta(x))$

- $h_\theta = 0.99$ → 0.01 (confident and right)
- $h_\theta = 0.5$ → 0.69
- $h_\theta = 0.01$ → 4.61 (confident and wrong)

**Case $y = 0$:** loss $= -\log(1 - h_\theta(x))$. The mirror image: now $h_\theta$ near 1 is the disaster.

![Left: when y = 1 the loss −log(f) explodes as f goes to 0. Right: when y = 0 the loss −log(1 − f) explodes as f goes to 1.](figures/logreg-logloss.svg)

In both panels, f is $h_\theta(x)$; the loss shoots up at the "confident and wrong" end.

**Both cases in one line**, averaged over $m$ examples:

$$
J(\theta_0, \theta_1) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^i \log h_\theta(x^i) + (1 - y^i) \log\big(1 - h_\theta(x^i)\big) \Big]
$$

$y$ is only 0 or 1, so it works like an on/off switch: when $y = 1$ the second term vanishes; when $y = 0$ the first does.

**Worked example** with $\theta_0 = -5$, $\theta_1 = 1$, on 4 of the tumours:

| Size | True y | hθ(x) | Term that's "on" | Loss |
|---|---|---|---|---|
| 2 | 0 | 0.047 | −log(1 − 0.047) | 0.049 |
| 4 | 0 | 0.269 | −log(1 − 0.269) | 0.313 |
| 6 | 1 | 0.731 | −log(0.731) | 0.313 |
| 8 | 1 | 0.953 | −log(0.953) | 0.049 |
| | | | **Cost J (mean)** | **0.181** |

Sizes 4 and 6 sit near the boundary, so they make up most of the cost. (`sklearn.metrics.log_loss` agrees: 0.181.)

## 5. Gradient descent

> **What you need to know**
> - **the same update rule shape as linear regression**; only $h_\theta$ has changed (now the sigmoid)
> - $\alpha$ = learning rate (step size); repeat until the cost stops falling
>
> **Skip:** the derivative working

$$
\theta_j := \theta_j - \alpha \, \frac{1}{m} \sum_{i=1}^{m} \big(h_\theta(x^i) - y^i\big)\, x_j^i
$$

$h_\theta(x^i) - y^i$ = how wrong example $i$ is; for $\theta_0$ there's no $x$ in the bracket ($x_0^i = 1$), just as in [linear regression](#/t/linreg).

**One step by hand.** Start at $\theta_0 = 0$, $\theta_1 = 0$: every $h_\theta = 0.5$ and the cost is 0.693. With $\alpha = 0.1$ on all 8 tumours:

- $\theta_0$: the errors are +0.5 four times and −0.5 four times; average 0, so $\theta_0$ stays 0.
- $\theta_1$: average of (error × size) = −1.25, so $\theta_1 := 0 - 0.1 \times (-1.25) = 0.125$.
- New cost: **0.598**, down from 0.693.

## 6. Regularisation

> **What you need to know**
> - add a penalty to the log loss, exactly as in [Regularisation](#/t/regularisation): $+\lambda(\text{slope})^2$ (L2, ridge) or $+\lambda\lvert\text{slope}\rvert$ (L1, lasso)
> - sklearn uses **C = 1/λ**: **small C = strong regularisation**, large C = weak
> - L2 is sklearn's default

![Left: the sigmoid with θ0 = −5, θ1 = 1 over the 8 tumours; a 0.5 threshold cuts at size 5 and a 0.3 threshold at size 4.2. Right: sklearn fits with C = 0.1, 1 and 100; smaller C gives a flatter, less confident curve.](figures/logreg-tumours.svg)

Right panel: all three cross 0.5 at size 5, but confidence differs. For the size-6 tumour:

| C | Penalty | θ1 | P(malignant), size 6 |
|---|---|---|---|
| 0.1 | strong | 0.44 | 0.61 |
| 1 (default) | medium | 1.14 | 0.76 |
| 100 | weak | 3.94 | 0.98 |

My LEDGAR model file notes the same: high C risks overfitting; low C keeps coefficients small and generalises better. Pick C on the **validation** set.

## 7. Threshold and predict_proba

> **What you need to know**
> - `predict_proba(X)[:, 1]` = probability of class 1; `predict(X)` uses 0.5 for you
> - **lower the threshold** → catch more positives (higher recall), more false alarms (lower precision); raise it → the opposite
> - choose it on the validation set ([Classification metrics](#/t/clsmetrics))

In the left panel of the figure above, a 0.3 threshold moves the cut-off from size 5 to **size 4.2**: sensible for tumours, where a miss is worse than a false alarm.

## 8. More than two classes: my LEDGAR project

For many classes use **softmax**: one score per class, turned into probabilities that add to 1 ([Softmax](#/t/softmax)). In [Legal Clause NLP](#/p/legal) I use it on TF-IDF features to sort contract clauses into 100 types: validation macro F1 0.780, accuracy 0.839 ([src/multi_class_lr_softmax.py](https://github.com/jag5m33/py_legal_data/blob/main/src/multi_class_lr_softmax.py)).

## Code: sklearn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),                   # scale: penalty treats features fairly
    LogisticRegression(C=1.0,           # smaller C = stronger penalty
                       class_weight="balanced",  # upweight the rare class
                       max_iter=1000),  # more steps if it warns
)
model.fit(X_train, y_train)                # training set only
p_val = model.predict_proba(X_val)[:, 1]   # probability of class 1
pred = (p_val >= 0.3).astype(int)          # my own threshold: 0.3
```

## Assumptions & caveats

- **Straight line in the score.** $z$ is a straight-line combination of features (statisticians say "linear in the log-odds"). Curves need extra features (squares, bins).
- **Multicollinearity.** Two features carrying the same information (temperature and wind chill) make the weights unstable; they can even flip sign. VIF flags it; above about 5 to 10 is a warning.
- **Scaling.** Needed when regularising, so the penalty treats all features fairly.
- **Class imbalance.** Accuracy misleads when positives are rare (see the one-liners for the fixes).
- **Perfectly separated classes.** With no penalty the weights keep growing for ever (like my 8 tumours); regularisation stops this.

## Use cases

- A fast, strong **baseline** for any yes/no problem.
- When you need to **explain** each feature's effect, or need **probabilities** (medicine, credit risk).
- **Text with TF-IDF** (my LEDGAR model).

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Outputs probabilities | Straight boundary unless you add features |
| Fast to train, even on millions of rows | Misses interactions unless you add them |
| Weights are easy to interpret | Weights unstable with multicollinearity |
| Smooth cost, reliable training | Needs scaling when regularised |
| Works well on sparse text | Usually beaten by tree models on complex tables |

## Interview one-liners

- **Is it regression or classification?** Classification: it outputs a probability, and a threshold makes it a class.
- **In one breath?** Straight-line score → sigmoid → probability; trained by minimising log loss with gradient descent; regularise with C; pick the threshold on validation.
- **Why log loss, not squared error?** Squared error with a sigmoid gives a bumpy cost; log loss is a smooth bowl and punishes confident mistakes hard.
- **What's C?** 1/λ. Small C = strong regularisation.
- **Imbalanced classes?** `class_weight="balanced"`, tune the threshold, judge with precision/recall.

## Quick reference

| Item | Formula / meaning |
|---|---|
| Score | z = θ0 + θ1x (θ0 = b, θ1 = w) |
| Sigmoid | hθ(x) = 1 / (1 + e^−z), g(0) = 0.5 |
| Decision boundary | z = 0 |
| Loss, y = 1 / y = 0 | −log(hθ) / −log(1 − hθ) |
| Cost J | average loss (binary cross-entropy) |
| Update | θj := θj − α · 1/m Σ (hθ(xⁱ) − yⁱ) xⱼⁱ |
| L2 / L1 penalty | λ(slope)² / λ·abs(slope) |
| sklearn C | 1/λ (small C = strong penalty) |
| Multiclass | softmax |
