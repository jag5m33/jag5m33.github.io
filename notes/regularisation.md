# Regularisation

Regularisation adds a charge for large weights to the cost function, so the model gives up a little training accuracy in exchange for a simpler, steadier fit that does better on new data.

## 1. The problem it solves

- **Overfitting (high variance):** the model fits the training data, noise included, so well that it does badly on new data. See [Bias & variance](#/t/biasvar).
- Overfit models usually have **large weights**: big slopes let a curve bend sharply to hit every point.
- **Regularisation** = add a term to the cost that grows with the size of the weights. Now the model has to balance two goals: fit the data, **and** keep the weights small.

## 2. Two points and a line (Krish Naik's walkthrough)

This is the standard way Krish Naik introduces ridge: shrink the problem down until you can do it by hand.

**Let's say we have 2 training points:** (1, 2) and (2, 4).

The line $\hat{y} = 2x$ passes **exactly** through both. Training error:

$$
\sum (y - \hat{y})^2 = (2 - 2)^2 + (4 - 4)^2 = 0
$$

A cost of zero looks perfect. But two points can't tell you the true pattern; any line through both of them fits "perfectly". The model has learned these two points, not the relationship. On new data it does badly:

![Two training points (black). The red line ŷ = 2x goes exactly through both, so its training cost is 0. The orange circles are new test points; the pink bars show how far the red line misses them. The flatter teal line ŷ = 1.2x + 1.2 misses the training points slightly but sits much closer to the test points.](figures/regularisation-twopoints.svg)

**Read the figure:** the red line is steep, so it shoots away from the test points on the right. Test squared error (SSE, sum of squared errors): **5.45** for the red line, **0.33** for the flatter teal line. Zero training error, worse test error: that's **overfitting / high variance**.

### Add the ridge penalty

Ridge changes the cost to:

$$
\text{cost} = \underbrace{\sum (y - \hat{y})^2}_{\text{fit the data}} + \underbrace{\lambda \cdot (\text{slope})^2}_{\text{keep the slope small}}
$$

- $\lambda$ (lambda): how much you charge for a big slope. Take $\lambda = 1$.

Now score some candidate lines (computed in Python):

| Line | Training error | Penalty λ·w² | **Total cost** |
|---|---|---|---|
| ŷ = 2.00x + 0.00 (perfect fit) | 0.00 | 4.00 | **4.00** |
| ŷ = 1.60x + 0.60 | 0.08 | 2.56 | **2.64** |
| ŷ = 1.20x + 1.20 | 0.32 | 1.44 | **1.76** |
| ŷ = 0.67x + 2.00 (ridge minimum) | 0.89 | 0.44 | **1.33** |

**The perfect-fit line now has the highest total cost.** The flatter lines miss the training points a little (training error goes up from 0 to 0.32), but their penalty drops by much more (4.00 → 1.44). So ridge **prefers the flatter line**.

![The ridge cost broken into its parts as the slope changes (the intercept is re-fitted for each slope). Blue dashed: training error, zero at slope 2. Gold dashed: the penalty, growing with slope². Black: the total, whose lowest point is at slope 0.67, not at 2.](figures/regularisation-costparts.svg)

**Read the figure:** the training error (blue) wants slope 2. The penalty (gold) wants slope 0. The total (black) is a compromise; its minimum sits between them. Increase λ and the gold curve gets steeper, pushing the minimum further left (flatter).

> **Key idea:** Regularisation deliberately accepts a **slightly worse fit on training data** (a bit more bias) in exchange for a **much steadier model** (a lot less variance).

With only 2 points, $\lambda = 1$ is a heavy penalty, which is why the minimum (0.67) overshoots to quite a flat line. In practice you choose λ with a validation set (section 7).

## 3. L2 regularisation (Ridge)

For a model with $n$ weights:

$$
J(\mathbf{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2
$$

- First term: the usual squared-error cost from [Linear regression](#/t/linreg).
- Second term: the **L2 penalty**, the sum of squared weights. "L2" because $\sqrt{\sum w_j^2}$ is the L2 norm (length) of the weight vector.
- $\frac{\lambda}{2m}$: dividing by $2m$ keeps the penalty on the same scale as the error term (Andrew Ng's convention), so a good λ doesn't change just because you add rows. The 2 again cancels on differentiating.

### Weight decay: what it does to gradient descent

Differentiate the penalty: $\frac{\partial}{\partial w_j}\frac{\lambda}{2m}w_j^2 = \frac{\lambda}{m}w_j$. Add that to the usual gradient:

$$
w_j := w_j - \alpha\left[\frac{1}{m}\sum_{i=1}^{m}\left(f(\mathbf{x}^{(i)}) - y^{(i)}\right)x_j^{(i)} + \frac{\lambda}{m}w_j\right]
$$

Collect the $w_j$ terms:

$$
w_j := w_j\left(1 - \alpha\frac{\lambda}{m}\right) - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(f(\mathbf{x}^{(i)}) - y^{(i)}\right)x_j^{(i)}
$$

- The second part is the **normal** gradient step, unchanged.
- The first part multiplies $w_j$ by a number **slightly less than 1** every iteration. With $\alpha = 0.01$, $\lambda = 1$, $m = 50$: $1 - 0.01 \times 1/50 = 0.9998$.
- So on every step, each weight is shrunk by a tiny percentage **before** the normal update. That's why L2 is also called **weight decay**.

**Behaviour:**

- Shrinks **all** weights towards zero, big ones most (the penalty is squared).
- Weights get small but **never exactly zero**. Ridge keeps every feature.
- With correlated features, Ridge **shares** the weight between them instead of picking one; this is what stabilises coefficients under multicollinearity.

## 4. L1 regularisation (Lasso)

$$
J(\mathbf{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{m}\sum_{j=1}^{n} \lvert w_j \rvert
$$

(Libraries scale this term slightly differently; the idea is the same: λ times the sum of absolute weights.)

- **L1 penalty:** the sum of **absolute** weights.
- **Lasso** = Least Absolute Shrinkage and Selection Operator.
- **Behaviour:** drives some weights to **exactly zero**. Those features are switched off entirely, so Lasso does **automatic feature selection**.

**Why exactly zero?** The penalty's pull on a weight is constant ($\lambda$, whatever the size of $w_j$) for L1, but proportional to $w_j$ for L2. Under L2, as a weight gets small the pull gets small too, so it never quite reaches 0. Under L1 the pull stays full strength all the way down, so a weight with a weak link to $y$ is pushed all the way to 0 and kept there.

### The geometry: circle vs diamond

Regularisation is equivalent to saying "find the best weights, but they must stay inside a fixed budget region". For two weights:

- L2 budget: $w_1^2 + w_2^2 \le t$, a **circle**.
- L1 budget: $\lvert w_1 \rvert + \lvert w_2 \rvert \le t$, a **diamond** with corners on the axes.

![Left (Ridge): oval cost contours centred on the unpenalised best weights, and a teal circle for the L2 budget. The smallest contour that touches the circle touches it at a point where both weights are non-zero. Right (Lasso): the same contours and a red diamond for the L1 budget. The contour first touches the diamond at its corner on the w1 axis, where w2 is exactly 0.](figures/regularisation-geometry.svg)

**Read the figure:**

- The black dot is the best (w1, w2) with no penalty. The grey ovals are cost contours around it (as in the [Linear regression](#/t/linreg) contour plot).
- The regularised answer is where the **expanding ovals first touch the shaded region** (gold star).
- The **circle** is smooth, so the first contact is usually somewhere on its curve: both weights shrunk, neither zero (here 0.85 and 0.53).
- The **diamond** has sharp **corners sitting on the axes**. The ovals very often hit a corner first, and on a corner one weight is exactly 0 (here w2 = 0). In many dimensions the diamond has many corners and edges, so many weights land on 0.

## 5. Elastic Net

$$
\text{penalty} = \lambda\left[r\sum_j \lvert w_j \rvert + \frac{1 - r}{2}\sum_j w_j^2\right]
$$

- A mix of L1 and L2. $r$ (sklearn `l1_ratio`) sets the mix: $r = 1$ is Lasso, $r = 0$ is Ridge.
- Use it when you want Lasso's feature selection but have **groups of correlated features**: the L2 part makes it keep correlated features together instead of picking one at random.

## 6. Integration into the cost functions

The pattern is always **the model's usual cost + a penalty on the weights**.

### Linear regression (squared error)

$$
J(\mathbf{w}, b) = \underbrace{\frac{1}{2m}\sum_{i=1}^{m}\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}\right)^2}_{\text{how badly we fit}} + \underbrace{\frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2}_{\text{how big the weights are}}
$$

### Logistic regression (log loss)

$$
J(\mathbf{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log f^{(i)} + (1 - y^{(i)})\log\left(1 - f^{(i)}\right)\right] + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2
$$

Each term:

- $f^{(i)} = \sigma(\mathbf{w}\cdot\mathbf{x}^{(i)} + b)$: the predicted probability that example $i$ is class 1 ($\sigma$ is the sigmoid; see [Logistic regression](#/t/logreg)).
- $y^{(i)}\log f^{(i)}$: active only when the true label is 1. If the model gives a low probability, $\log f$ is very negative, so the loss is big.
- $(1 - y^{(i)})\log(1 - f^{(i)})$: active only when the true label is 0. Punishes a high probability for class 1.
- The minus sign and $\frac{1}{m}$: turn it into a positive average loss.
- $\frac{\lambda}{2m}\sum w_j^2$: the **same** L2 penalty as before. It stops the weights growing huge, which in logistic regression means it stops the model becoming **overconfident** (probabilities pushed to 0.0000 and 1.0000) and stops the decision boundary twisting to fit individual points.

The gradient-descent update for logistic regression has **exactly the same** weight-decay form as for linear regression; only $f$ changes.

### Why b isn't penalised

- $b$ just shifts the whole prediction up or down. It doesn't make the model wigglier or more sensitive to any feature, so it can't cause overfitting.
- Penalising it would pull predictions towards 0 for no reason (e.g. biasing every house price towards £0).
- So the sum runs over $j = 1 \ldots n$ (the weights), and $b$ is left out. sklearn does the same.

## 7. λ in plain words

> **Say it like this:** "λ is the price you charge the model for big weights."

| λ | What happens | Bias / variance |
|---|---|---|
| **0** | No charge. Plain linear / logistic regression | Can overfit (high variance) |
| **Small** | Big weights cost a little; the wildest ones are trimmed | Usually the sweet spot |
| **Large** | Big weights are expensive; all weights squeezed towards 0 | Underfits (high bias) |
| **Huge** | Every weight ≈ 0; predicts roughly a constant ($b$) | Severe underfitting |

![Coefficient paths for eight features. Left, Ridge: as λ grows (left to right), every weight ends up curving smoothly towards zero but none reaches it. Right, Lasso: weights drop and hit exactly zero one at a time; near the right-hand end only the strongest features remain.](figures/regularisation-paths.svg)

**Read the figure:** follow one coloured line from left (weak penalty) to right (strong). On the **Ridge** side it bends gently and flattens near zero. (The red line w2 first *rises*: w2's feature is a near-copy of w1's, and as Ridge shrinks w1 it hands some of that shared weight to w2. That's the weight-sharing from section 3.) On the **Lasso** side lines **snap onto the zero line** and stay there; the order they drop out is roughly the order of usefulness. (The x-axes differ because sklearn's Ridge and Lasso scale `alpha` differently.)

![Training and validation error against λ for a degree-12 polynomial fitted to 15 points. With λ tiny (left), training error is very low but validation error is high: overfitting. With λ large (right), both errors are high: underfitting. Validation error is lowest in between, marked by the gold dashed line.](figures/regularisation-lambda.svg)

**Read the figure:** this is the [Bias & variance](#/t/biasvar) U-curve again, but now the x-axis is λ and it runs the **opposite** way: moving right means a **simpler** model. Pick λ at the bottom of the **validation** curve. Never pick it from training error: training error is always lowest at λ = 0.

## 8. sklearn: names and the C trap

| Model | Regularisation strength | Penalty type |
|---|---|---|
| `Ridge(alpha=...)` | `alpha` = λ. **Bigger = stronger** | L2 |
| `Lasso(alpha=...)` | `alpha` = λ. **Bigger = stronger** | L1 |
| `ElasticNet(alpha=..., l1_ratio=...)` | `alpha` = λ; `l1_ratio` = mix | L1 + L2 |
| `LogisticRegression(C=...)` | `C` = **1/λ**. **Bigger = WEAKER** | L2 by default |

> **Trap:** `C` is the **inverse** of λ. Small C = strong regularisation (simpler model). Large C = weak regularisation (risk of overfitting). From Jag's LEDGAR notes: "If variance is high: more data, more regularisation (**lower C**)."

**L1 in LogisticRegression.** Not every solver supports L1. Use `solver="liblinear"` (small/medium data, binary or one-vs-rest) or `solver="saga"` (large data, multinomial). The default `lbfgs` supports only L2.

> **Version note:** In sklearn 1.8 the `penalty` argument of `LogisticRegression` was deprecated (to be removed in 1.10). The new spelling is `l1_ratio`: `l1_ratio=0` for L2 (the default), `l1_ratio=1` for L1, a value in between for elastic net. On older versions write `penalty="l1"` or `penalty="l2"`. Check `sklearn.__version__`.

## 9. Code

### The two-point example

```python
import numpy as np

x = np.array([1.0, 2.0])          # two training points
y = np.array([2.0, 4.0])
lam = 1.0                         # lambda: the price per unit of slope²

def ridge_cost(w, b):
    sse = ((y - (w * x + b)) ** 2).sum()   # training error
    penalty = lam * w ** 2                 # ridge penalty on the slope only
    return sse, penalty, sse + penalty

for w, b in [(2.0, 0.0), (1.6, 0.6), (1.2, 1.2), (2/3, 2.0)]:
    sse, pen, total = ridge_cost(w, b)
    print(f"w={w:.2f} b={b:.2f}  error={sse:.2f}  penalty={pen:.2f}  total={total:.2f}")
```

Prints the table in section 2: totals 4.00, 2.64, 1.76, 1.33.

### Ridge, Lasso and Elastic Net on an easy-to-overfit dataset

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import root_mean_squared_error

# 60 rows, 40 features, only 5 of them useful: easy to overfit
X, y = make_regression(n_samples=60, n_features=40, n_informative=5,
                       noise=15, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "OLS":        LinearRegression(),
    "Ridge":      Ridge(alpha=1.0),                       # alpha = lambda
    "Lasso":      Lasso(alpha=1.0, max_iter=10_000),
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.9, max_iter=10_000),
}
for name, reg in models.items():
    pipe = make_pipeline(StandardScaler(), reg)            # scale INSIDE the pipeline
    pipe.fit(X_tr, y_tr)                                    # scaler + model fit on train only
    tr = root_mean_squared_error(y_tr, pipe.predict(X_tr))
    va = root_mean_squared_error(y_va, pipe.predict(X_va))
    n_zero = np.sum(pipe[-1].coef_ == 0)                    # weights set exactly to 0
    print(f"{name:10s} train RMSE {tr:6.1f}  val RMSE {va:6.1f}  zero weights {n_zero}")
```

| Model | Train RMSE | Val RMSE | Zero weights (of 40) |
|---|---|---|---|
| OLS (no penalty) | 2.2 | 44.9 | 0 |
| Ridge | 5.4 | 37.0 | 0 |
| Lasso | 7.5 | 16.4 | 17 |
| Elastic Net | 9.7 | 31.6 | 10 |

Plain least squares nearly memorises the 42 training rows (train 2.2) and fails on validation (44.9): textbook high variance. Every penalty raises training error and lowers validation error. Lasso wins **here** because the truth is sparse (only 5 useful features); when most features matter a little, Ridge usually wins.

### Choosing λ by cross-validation

```python
# continues from the block above (same imports, X and y)
from sklearn.linear_model import RidgeCV, LassoCV

alphas = np.logspace(-3, 3, 50)                           # candidate lambdas
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(X, y)
lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=5, max_iter=10_000)).fit(X, y)
print("best ridge alpha:", ridge[-1].alpha_)
print("best lasso alpha:", round(lasso[-1].alpha_, 3), "| features kept:", np.sum(lasso[-1].coef_ != 0))
```

### Logistic regression

```python
# continues from the blocks above (make_pipeline, StandardScaler already imported)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=300, n_features=20, n_informative=4, random_state=0)

# L2 (default). C = 1/lambda, so SMALL C = STRONG regularisation
l2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1))
# L1 (lasso-style). sklearn >= 1.8: l1_ratio=1 means pure L1
l1 = make_pipeline(StandardScaler(),
                   LogisticRegression(C=0.1, l1_ratio=1.0, solver="liblinear"))
# Older sklearn (< 1.8) spelling of the same model:
# LogisticRegression(C=0.1, penalty="l1", solver="liblinear")

for name, m in [("L2", l2), ("L1", l1)]:
    m.fit(X, y)
    print(name, "zero weights:", (m[-1].coef_ == 0).sum(), "of", m[-1].coef_.size)
```

Output: L2 keeps all 20 weights; L1 sets 15 of 20 to exactly zero.

Jag's LEDGAR classifier ([multi_class_lr_softmax.py](https://github.com/jag5m33/py_legal_data/blob/main/src/multi_class_lr_softmax.py)) uses L2 with `C=1.0`; C is the first lever to try against its 11-point train/validation gap.

## 10. Scale features first

The penalty $\sum w_j^2$ treats every weight the same, but a weight's size depends on its feature's **units**:

- Floor area in m² (values ~100) needs a small weight, e.g. 0.02 per m².
- Floor area in km² (values ~0.0001) needs a huge weight for the **same** effect.

Without scaling, the penalty punishes the km² feature far more, purely because of its units. **Standardise** (mean 0, standard deviation 1) first so every weight is on a fair footing. Fit the scaler on training data only, inside a pipeline, to avoid leakage. See [Scaling](#/t/scaling).

## 11. Assumptions & caveats

- **Features must be scaled** (section 10). This is the most common mistake.
- **λ must be tuned** on validation data or by cross-validation. The default (`alpha=1.0`, `C=1.0`) is a starting point, not an answer.
- **Lasso picks arbitrarily among correlated features.** If size and rooms carry the same signal, Lasso may keep one and zero the other, and which one can flip with a small change in the data. Don't read "Lasso dropped rooms" as "rooms doesn't matter".
- **Ridge keeps them all** and shares weight between correlated features. More stable, but no feature selection.
- **Coefficients are biased** towards zero on purpose. Don't use regularised coefficients (or their standard errors) for classical inference / p-values.
- **The intercept is not penalised** (section 6), so centre-or-scale thinking applies to features, not to $b$.
- **Different libraries scale λ differently** (with or without $1/m$, $1/2$). The same `alpha` doesn't mean the same strength in Ridge and Lasso.

## 12. Use cases

- **More features than you can trust**: text (TF-IDF with thousands of words), genomics (thousands of genes, few samples). Jag's LEDGAR model is regularised logistic regression on TF-IDF.
- **Multicollinearity**: Ridge stabilises coefficients that would otherwise flip sign.
- **Feature selection**: Lasso to find a short list of useful predictors.
- **Any model showing a big train–validation gap**: first-line fix before collecting more data.
- **Default for linear and logistic models** in production: a little L2 rarely hurts.

## 13. Advantages / Disadvantages

| | Advantages | Disadvantages |
|---|---|---|
| **Ridge (L2)** | Reduces variance; stabilises correlated features; smooth, closed-form solution; keeps every feature | No feature selection; coefficients biased; needs scaling and λ tuning |
| **Lasso (L1)** | Feature selection built in; sparse, interpretable models | Unstable choice among correlated features; at most m features selected when n > m; no closed form |
| **Elastic Net** | Selection plus stability with correlated groups | Two settings to tune (alpha and l1_ratio) |
| **All** | Simple to add; works with linear, logistic, neural nets | Too much λ underfits; not a cure for bias |

## 14. Limitations

- Regularisation fixes **variance**, not **bias**. If the model is too simple, penalising it makes things worse.
- It can't create information that isn't in the features. More data or better features may still be needed.
- It shrinks weights uniformly by **size**, not by **meaning**: a genuinely important feature with a large effect is shrunk too.
- Lasso's selected feature set is not a reliable statement about which features "truly" matter when features are correlated.

## Quick reference

| Term | Formula / meaning |
|---|---|
| Regularisation | Cost = usual loss + penalty on weight size |
| L2 / Ridge | Penalty $\frac{\lambda}{2m}\sum w_j^2$; shrinks, never zero |
| L1 / Lasso | Penalty proportional to $\lambda\sum \lvert w_j \rvert$; some weights exactly zero |
| Elastic Net | Mix of L1 and L2; `l1_ratio` sets the mix |
| Weight decay | $w_j := w_j(1 - \alpha\lambda/m) - \alpha \cdot \text{gradient}$ |
| Regularised log loss | Log loss + $\frac{\lambda}{2m}\sum w_j^2$ |
| λ | Price charged for big weights; 0 = none, large = underfit |
| sklearn Ridge / Lasso | `alpha` = λ (bigger = stronger) |
| sklearn LogisticRegression | `C` = 1/λ (bigger = weaker); L1 needs liblinear or saga |
| b | Not penalised |
| Before fitting | Scale features; tune λ on validation / CV |
