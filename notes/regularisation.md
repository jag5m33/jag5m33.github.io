# Regularisation

Regularisation adds a charge for large weights to the cost function, so the model gives up a little training accuracy in exchange for a simpler, steadier fit that does better on new data.

## 1. The problem it solves

Jag's notes: **regularisation aims to reduce overfitting** (reduce model complexity).

- **Overfitting (high variance):** the model "struggles with new unseen data"; it "memorises training data"; its training cost is almost zero, $J(\theta) \approx 0$. See [Bias & variance](#/t/biasvar).
- Overfit models usually have **large weights**: big slopes let a line or curve bend sharply to hit every point.
- **Regularisation** adds a **penalty** to the cost that grows with the size of the weights. Now the model has two goals: fit the data, **and** keep the weights small.

## 2. Two points and a line (Krish Naik's walkthrough)

This is the standard way Krish Naik introduces ridge: shrink the problem until you can do it by hand.

**Let's say we have 2 training points:** (1, 2) and (2, 4).

The line $\hat{y} = 2x$ goes **exactly** through both, so its training error is 0:

$$
\sum (y - \hat{y})^2 = (2 - 2)^2 + (4 - 4)^2 = 0
$$

That looks perfect. But two points can't show the true pattern. The model has learned these two points, not the relationship, so it does badly on new data:

![Two training points (black). The red line ŷ = 2x goes exactly through both, so its training cost is 0. The orange circles are new test points; the pink bars show how far the red line misses them. The flatter teal line ŷ = 1.2x + 1.2 misses the training points slightly but sits much closer to the test points.](figures/regularisation-twopoints.svg)

In the figure, the steep red line shoots away from the test points. Its test error (sum of squared misses) is **5.45**; the flatter teal line's is only **0.33**. Zero training error but worse test error: that's **overfitting**.

### Add the ridge penalty

Ridge adds a penalty to the cost from [Linear regression](#/t/linreg):

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda\,(\text{slope})^2
$$

- The first part says "fit the data". The second says "keep the slope small". ($\theta_1$ = slope = $w$.)
- In Jag's words: "if the first part is 0 due to overfitting, add λ(slope)² to ensure it counteracts overfitting."
- $\lambda$ (lambda): the **regularisation parameter**, the value multiplied by slope² to lessen or increase the effect. It "controls how extreme you counteract overfitting".
- For the hand example we drop the $\frac{1}{2m}$ and just add up the squared errors (Krish Naik's simple version), with $\lambda = 1$. The idea is identical; the $\frac{1}{2m}$ just makes the penalty count for relatively more, so the best line comes out flatter still.

Now score some lines:

| Line | Training error | Penalty λ·w² | **Total cost** |
|---|---|---|---|
| ŷ = 2.00x + 0.00 (perfect fit) | 0.00 | 4.00 | **4.00** |
| ŷ = 1.60x + 0.60 | 0.08 | 2.56 | **2.64** |
| ŷ = 1.20x + 1.20 | 0.32 | 1.44 | **1.76** |
| ŷ = 0.67x + 2.00 (ridge best) | 0.89 | 0.44 | **1.33** |

**The perfect-fit line now has the highest cost.** The flatter lines miss the training points a little, but their penalty drops by much more. So ridge **prefers the flatter line**.

![The ridge cost broken into its parts as the slope changes (the intercept is re-fitted for each slope). Blue dashed: training error, zero at slope 2. Gold dashed: the penalty, growing with slope². Black: the total, whose lowest point is at slope 0.67, not at 2.](figures/regularisation-costparts.svg)

In the figure, the training error (blue) wants slope 2 and the penalty (gold) wants slope 0. The total (black) is a compromise, with its lowest point in between. A bigger λ pushes that lowest point towards a flatter line.

The cost function is minimised as usual, but now the minimum sits on a flatter line → less overfitting → **lower validation error**.

> **Key idea:** regularisation accepts a **slightly worse fit on training data** (a bit more bias) in exchange for a **much steadier model** (a lot less variance).

## 3. L2 (Ridge) and L1 (Lasso)

Both add a penalty on the slopes (weights) to the usual cost. They differ in how they measure "big". With lots of features there are lots of slopes, one per feature, so the penalty adds them all up.

### L2 / Ridge: penalise squared slopes

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda \sum_{j} \theta_j^2
$$

- $\theta_j$: the slope (weight) of feature $j$. $\lambda$: the strength.
- **Example:** slopes 3 and −1 with $\lambda = 1$: penalty $= 3^2 + (-1)^2 = 10$.
- Shrinks **all** slopes towards zero, big ones most, but **never exactly to zero**. Every feature stays in.
- With two near-identical features, Ridge **shares** the weight between them, which calms down multicollinearity (see [Linear regression](#/t/linreg)).

### L1 / Lasso: penalise absolute slopes

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda \sum_{j} \lvert \theta_j \rvert
$$

- $\lvert \theta_j \rvert$: the **modulus** (absolute value) of each slope. No squaring, as there is in L2.
- In Jag's words: the "overfitting equation + λ × sum of (absolute) coefficients".
- **Example:** the same slopes 3 and −1: penalty $= 3 + 1 = 4$.
- Features with small coefficients have little effect on the output, so Lasso **neglects** them: it drives their slopes to **exactly zero**. That's **automatic feature selection**.

> **Say it like this (Jag's summary):** "L1 helps with feature selection / importance + reduces overfitting. L2 reduces overfitting."

## 4. λ in plain words

> **Say it like this:** "λ is the price you charge the model for big weights."

| λ | What happens | Result |
|---|---|---|
| **0** | No charge: plain linear / logistic regression | Can overfit |
| **Small** | The wildest weights get trimmed | Usually the sweet spot |
| **Large** | All weights squeezed towards 0 | Underfits |
| **Huge** | Every weight ≈ 0; predicts roughly a constant | Badly underfits |

The bias $b$ is **not** penalised: it only shifts predictions up or down and can't make the model wiggly.

![Coefficient paths for eight features. Left, Ridge: as λ grows (left to right), every weight ends up curving smoothly towards zero but none reaches it. Right, Lasso: weights drop and hit exactly zero one at a time; near the right-hand end only the strongest features remain.](figures/regularisation-paths.svg)

In the figure, follow one coloured line from left (weak penalty) to right (strong). In Ridge it bends gently towards zero. In Lasso the lines **snap onto zero** one by one; the last ones left are the most useful features.

![Training and validation error against λ for a degree-12 polynomial fitted to 15 points. With λ tiny (left), training error is very low but validation error is high: overfitting. With λ large (right), both errors are high: underfitting. Validation error is lowest in between, marked by the gold dashed line.](figures/regularisation-lambda.svg)

In the figure, pick λ at the bottom of the **validation** curve (gold dashed line). Never pick it from training error, which is always lowest at λ = 0.

### λ in sklearn: alpha and C

| Model | Setting | Bigger means |
|---|---|---|
| `Ridge(alpha=...)` | `alpha` = λ | **Stronger** penalty |
| `Lasso(alpha=...)` | `alpha` = λ | **Stronger** penalty |
| `LogisticRegression(C=...)` | `C` = 1/λ | **Weaker** penalty |

> **Trap:** `C` works **backwards**. Small C = strong regularisation (simpler model). Large C = weak regularisation (more risk of overfitting). From Jag's LEDGAR notes: "If variance is high: more data, more regularisation (**lower C**)."

## 5. Scale features first

The penalty treats every weight the same, but a weight's size depends on its feature's **units**. Floor area in km² needs a huge weight; in m² a tiny one, for the same effect. Without scaling, the penalty unfairly punishes some features. **Standardise** first (see [Scaling](#/t/scaling)), inside a pipeline fitted on training data only.

## 6. Code

### The two-point example

```python
import numpy as np

x = np.array([1.0, 2.0])          # two training points
y = np.array([2.0, 4.0])
lam = 1.0                         # lambda: the price per unit of slope squared

for w, b in [(2.0, 0.0), (1.6, 0.6), (1.2, 1.2), (2/3, 2.0)]:
    error = ((y - (w * x + b)) ** 2).sum()   # training error
    penalty = lam * w ** 2                   # ridge penalty on the slope
    print(f"w={w:.2f}  error={error:.2f}  penalty={penalty:.2f}  total={error + penalty:.2f}")
```

Prints the totals from the table: 4.00, 2.64, 1.76, 1.33.

### Ridge and Lasso on an easy-to-overfit dataset

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error

# 60 rows, 40 features, only 5 of them useful: easy to overfit
X, y = make_regression(n_samples=60, n_features=40, n_informative=5, noise=15, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0)

for model in [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=1.0, max_iter=10_000)]:
    pipe = make_pipeline(StandardScaler(), model).fit(X_tr, y_tr)   # scale, then fit, on train only
    val_rmse = root_mean_squared_error(y_va, pipe.predict(X_va))    # typical miss on new data
    print(model, round(val_rmse, 1), (pipe[-1].coef_ == 0).sum())   # val RMSE, weights at 0
```

| Model | Train RMSE | Val RMSE | Weights set to 0 (of 40) |
|---|---|---|---|
| No penalty | 2.2 | 44.9 | 0 |
| Ridge | 5.4 | 37.0 | 0 |
| Lasso | 7.5 | 16.4 | 17 |

With no penalty the model nearly memorises the training rows and fails on validation. Both penalties raise training error but lower validation error. Lasso wins **here** because only 5 features really matter.

Jag's LEDGAR classifier ([multi_class_lr_softmax.py](https://github.com/jag5m33/py_legal_data/blob/main/src/multi_class_lr_softmax.py)) uses L2 with `C=1.0`; lowering C is the first lever to try against its 11-point train/validation gap.

## 7. Caveats

- **Scale features first.** The most common mistake.
- **Tune λ** on validation data or with cross-validation (`RidgeCV`, `LassoCV`). The default is only a starting point.
- **Lasso picks one of a correlated pair at random.** If it drops "rooms" but keeps "size", that doesn't mean rooms is useless.
- **Weights are shrunk on purpose**, so don't read them as exact effect sizes.

## 8. Use cases

- **Lots of features, few rows:** text (TF-IDF), genomics. Jag's LEDGAR model is regularised logistic regression on TF-IDF.
- **Multicollinearity:** Ridge steadies jumpy weights.
- **Feature selection:** Lasso finds a short list of useful features.
- **Any big train–validation gap:** a first fix before collecting more data.

## 9. Advantages / Disadvantages

| | Advantages | Disadvantages |
|---|---|---|
| **Ridge (L2)** | Reduces overfitting; steadies correlated features; keeps every feature | No feature selection; needs scaling and λ tuning |
| **Lasso (L1)** | Built-in feature selection; simpler models | Unstable choice among correlated features |
| **Both** | Easy to add to linear and logistic models | Too much λ underfits; fixes variance, not bias |

## Quick reference

| Term | Meaning |
|---|---|
| Regularisation | Cost = usual error + penalty on weight size |
| Overfitting | Memorises training data, $J(\theta) \approx 0$, high validation error |
| L2 / Ridge | Cost + $\lambda\sum \theta_j^2$; shrinks, never exactly zero; reduces overfitting |
| L1 / Lasso | Cost + $\lambda\sum \lvert \theta_j \rvert$; small slopes go to exactly zero; feature selection + reduces overfitting |
| λ | Regularisation parameter: how hard you counteract overfitting; 0 = none, large = underfit |
| sklearn Ridge / Lasso | `alpha` = λ (bigger = stronger) |
| sklearn LogisticRegression | `C` = 1/λ (bigger = weaker) |
| b | Not penalised |
| Before fitting | Scale features; tune λ on validation |
