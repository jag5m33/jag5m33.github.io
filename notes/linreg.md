# Linear regression

Linear regression predicts a number by drawing the straight line (or flat plane) that makes the squared prediction errors as small as possible, and gradient descent is one way to find that line.

## The running example

Six houses. One feature, one target. Every number in this chapter comes from these six rows.

| House | Size x (100 m²) | Price y (£100k) |
|---|---|---|
| 1 | 0.8 | 1.9 |
| 2 | 1.2 | 2.4 |
| 3 | 1.5 | 3.3 |
| 4 | 2.0 | 3.8 |
| 5 | 2.4 | 4.9 |
| 6 | 3.0 | 5.6 |

- **Feature** $x$: the input you know (size).
- **Target** $y$: the number you want to predict (price).
- **Training example** $(x^{(i)}, y^{(i)})$: row $i$. The superscript is a row number, not a power.
- $m$: the number of training examples. Here $m = 6$.

## 1. The model: f = wx + b

$$
f_{w,b}(x) = wx + b
$$

- $f_{w,b}(x)$, also written $\hat{y}$ ("y-hat"): the **prediction**.
- $w$: the **weight**, which is the **slope**. How much the prediction rises when $x$ goes up by 1.
- $b$: the **bias** or **intercept**. The prediction when $x = 0$; where the line crosses the y-axis.
- $w$ and $b$ are the **parameters**: the two numbers the model learns. Training = choosing $w$ and $b$.

![The six houses (red dots) and the best straight line through them (teal). The gold dashed triangle shows the slope: go 1 unit right and the line rises by w = 1.74. The gold dot on the y-axis is the intercept b = 0.49.](figures/linreg-model.svg)

**Read the figure:** the slope triangle says "each extra 100 m² adds about 1.74 × £100k = £174k to the predicted price". The intercept (0.49, i.e. £49k for a house of size 0) is where the line hits the axis. It is needed to place the line, but it has no sensible real meaning here because no house has zero size. Intercepts often mean nothing on their own.

> **Say it like this:** "The model is a straight line. w is how steep it is, b is where it starts. Learning means finding the w and b that fit the data best."

## 2. Residuals: how wrong is one prediction?

Before we can find the "best" line we need a way to score a line. Start with one house.

$$
\text{residual}^{(i)} = \hat{y}^{(i)} - y^{(i)} = f_{w,b}(x^{(i)}) - y^{(i)}
$$

A **residual** (also called the **error**) is the vertical gap between the prediction and the truth. (Statisticians often write it the other way round, $y - \hat{y}$; the sign doesn't matter once we square it.)

Take a deliberately poor guess, $w = 1.0$, $b = 1.0$:

| x | y | ŷ = 1.0x + 1.0 | residual ŷ − y | residual² |
|---|---|---|---|---|
| 0.8 | 1.9 | 1.8 | −0.1 | 0.01 |
| 1.2 | 2.4 | 2.2 | −0.2 | 0.04 |
| 1.5 | 3.3 | 2.5 | −0.8 | 0.64 |
| 2.0 | 3.8 | 3.0 | −0.8 | 0.64 |
| 2.4 | 4.9 | 3.4 | −1.5 | 2.25 |
| 3.0 | 5.6 | 4.0 | −1.6 | 2.56 |
| | | | **sum** | **6.14** |

![The guess f = 1.0x + 1.0 (blue) with the red vertical lines showing each residual. Every dot is above the line, so every residual is negative: the guess is too flat and too low.](figures/linreg-residuals.svg)

**Read the figure:** the red lines get longer to the right. That pattern is a clue: the slope is too small. A good line has residuals that are short and scattered above and below it with no pattern.

> **Trap:** The residual is the **vertical** distance (measured in units of y), not the shortest perpendicular distance to the line. We only care how wrong the predicted price is.

## 3. Why squared error (and why ½m)

We need to combine six residuals into one score. Options:

| Idea | Problem |
|---|---|
| Add the residuals | +2 and −2 cancel to 0 and look perfect |
| Add absolute residuals $\lvert e \rvert$ | Works (that's MAE), but has a sharp corner at 0, so it's awkward to differentiate |
| **Add squared residuals** $e^2$ | Always positive, punishes big misses more, smooth everywhere, easy calculus |

So the **cost function** (one number that scores the whole line) is:

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2
$$

Where each piece comes from:

- $\sum (\ldots)^2$: add up the squared residuals (6.14 above).
- $\frac{1}{m}$: take the **average**, so the cost doesn't grow just because you have more rows. Now it's comparable across datasets.
- $\frac{1}{2}$: pure convenience. When you differentiate $(\ldots)^2$ you get $2(\ldots)$; the ½ **cancels that 2** and keeps the gradient formula clean. It doesn't move the minimum: halving a function doesn't change where its lowest point is.

For the guess $w=1, b=1$: $J = 6.14 / (2 \times 6) = 0.512$.

For the best line ($w = 1.741, b = 0.487$): $J = 0.014$. Much smaller. That's what "best" means here.

> **Key idea:** $J$ is the average squared error, halved. Without the ½ it is exactly the **MSE**, so minimising one minimises the other. See [Regression metrics](#/t/regmetrics).

## 4. The cost is a bowl

For every possible pair $(w, b)$ there is one value of $J$. Plotting $J$ against the parameters shows the "landscape" we're trying to get to the bottom of.

### J against w alone

Fix $b$ at its best value and slide $w$:

![Left: three candidate lines with different slopes (b held fixed). Right: the cost J for every slope. Each line on the left is a single dot on the parabola on the right, in the matching colour.](figures/linreg-cost-w.svg)

**Read the figure:** the blue line ($w = 0.9$) is too flat and sits high up the left side of the parabola; the orange line ($w = 2.6$) is too steep and sits up the right side; the teal line sits at the bottom. Because $J$ is a sum of squares of something linear in $w$, it is a **parabola** ($J$ is quadratic in $w$). A parabola has exactly one lowest point.

### J against w and b together: the contour plot

![Left: J(w, b) as a 3-D bowl. Right: the same bowl seen from directly above. Each ring is a contour, a set of (w, b) pairs with the same cost, like height lines on a map. The star at the centre is the minimum.](figures/linreg-contour.svg)

**Read the figure:**

- Each ring = all lines that are **equally bad**. Different $(w, b)$ pairs, same $J$.
- The rings are **long thin ellipses tilted downwards**. Along the long direction you can raise $w$ and lower $b$ together and the line barely changes (it pivots around the middle of the data), so the cost barely changes. Across the short direction the cost rises fast.
- One single centre (the star). For linear regression with squared error, $J$ is **convex**: one bowl, no false bottoms (local minima). Gradient descent cannot get stuck.

## 5. Deriving the gradient (step by step)

The **gradient** is the slope of the cost surface: how much $J$ changes if you nudge $w$ (or $b$) a tiny bit. To walk downhill we need $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$.

**Step 1. Name the inside.** Let $e^{(i)} = wx^{(i)} + b - y^{(i)}$ (the residual). Then

$$
J = \frac{1}{2m}\sum_{i=1}^{m} \left(e^{(i)}\right)^2
$$

**Step 2. Chain rule.** To differentiate "something squared", differentiate the square, then multiply by the derivative of the inside:

$$
\frac{\partial}{\partial w}\left(e^{(i)}\right)^2 = 2e^{(i)} \cdot \frac{\partial e^{(i)}}{\partial w}
$$

**Step 3. Derivative of the inside.** $e^{(i)} = wx^{(i)} + b - y^{(i)}$. Only the first term contains $w$, so $\frac{\partial e^{(i)}}{\partial w} = x^{(i)}$. Only the second contains $b$, so $\frac{\partial e^{(i)}}{\partial b} = 1$.

**Step 4. Put it together.** The ½ cancels the 2:

$$
\frac{\partial J}{\partial w} = \frac{1}{2m}\sum 2e^{(i)}x^{(i)} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)x^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
$$

In words: the $b$-gradient is the **average residual**; the $w$-gradient is the **average residual weighted by x**.

**With numbers** (guess $w = 1, b = 1$, residuals from the table above):

- $\frac{\partial J}{\partial b}$ = mean of (−0.1, −0.2, −0.8, −0.8, −1.5, −1.6) = **−0.833**
- $\frac{\partial J}{\partial w}$ = mean of residual × x = (−0.08 − 0.24 − 1.2 − 1.6 − 3.6 − 4.8) / 6 = **−1.92**

Both are negative: increasing $w$ or $b$ would **decrease** the cost. So we should increase them. That matches the residual figure: the line was too flat and too low.

## 6. Gradient descent

**Gradient descent** = repeatedly take a small step in the direction that lowers $J$ fastest, which is the opposite direction to the gradient.

$$
w := w - \alpha\frac{\partial J}{\partial w} \qquad b := b - \alpha\frac{\partial J}{\partial b}
$$

- $:=$ means "replace with" (assignment, like `=` in Python).
- $\alpha$ (alpha): the **learning rate**, the step size. A small positive number like 0.01 or 0.1.
- Minus sign: the gradient points uphill; we want downhill.

**One step with numbers** ($\alpha = 0.1$):

- $w := 1.0 - 0.1 \times (-1.92) = 1.192$
- $b := 1.0 - 0.1 \times (-0.833) = 1.083$
- Cost falls from 0.512 to **0.177** in one step.

### Simultaneous update

Compute **both** gradients using the **old** $w$ and $b$, then update both. If you update $w$ first and then compute the $b$-gradient with the new $w$, you are mixing two different points on the surface.

```python
# correct: both gradients come from the same (old) w, b
tmp_w = w - alpha * dj_dw(w, b)
tmp_b = b - alpha * dj_db(w, b)
w, b = tmp_w, tmp_b
```

### Watching it work

![Left: gradient descent starting at (w=0, b=−2), each dot one step, on the contour plot. Right: the cost after each iteration on a log scale. It falls fast, then flattens once the minimum is reached.](figures/linreg-descent.svg)

**Read the figure:**

- **Left:** each step goes in the **steepest downhill direction**, which cuts straight across the contour rings (exactly at right angles when both axes use the same scale). It shoots quickly down the steep short direction, then crawls along the long valley floor towards the star. Steps get shorter on their own as the slope flattens, even though $\alpha$ is fixed.
- **Right:** the **learning curve** of the optimiser. It must go down every iteration. When it flattens, you have **converged** (further steps change almost nothing). If it ever goes **up**, $\alpha$ is too big or there's a bug.

### Choosing the learning rate

![Gradient descent on J(w) with three learning rates, same start. Too small (left): tiny steps, nowhere near the bottom after 8 steps. About right (middle): reaches the bottom in a few steps. Too big (right): each step jumps past the minimum to a higher point on the other side, so the cost grows.](figures/linreg-alpha.svg)

**Read the figure:** on the right, the dots zig-zag across the bowl and climb the walls: that is **divergence**. Practical recipe (Andrew Ng's): try $\alpha$ = 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, …, plot cost vs iteration for each, and pick the largest one whose cost falls smoothly.

## 7. Multiple features and vector form

Real data has many features: size, rooms, age, distance to station.

$$
f_{\mathbf{w},b}(\mathbf{x}) = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = \mathbf{w}\cdot\mathbf{x} + b
$$

- $n$: number of features.
- $\mathbf{w} = [w_1, \ldots, w_n]$: one weight per feature. $w_j$ = change in prediction when feature $j$ rises by 1 **with all other features held fixed**.
- $\mathbf{w}\cdot\mathbf{x}$: the **dot product**: multiply matching entries and add. In numpy, `np.dot(w, x)` or `w @ x`.

The picture changes from a line to a flat **plane** (2 features) or **hyperplane** (more). The update rule is the same, done for every $j$ at once:

$$
w_j := w_j - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}\right)x_j^{(i)}
$$

**Vectorisation** means writing this with whole-array operations instead of a `for` loop over features. It's shorter and much faster because numpy runs it in optimised C.

> **Key idea:** With several features on different scales (m² in the hundreds, rooms under 10), the bowl becomes a very stretched ellipse and gradient descent zig-zags slowly. **Scale the features** first; see [Scaling](#/t/scaling).

## 8. The normal equation: solving it in one go

Because $J$ is a bowl, we can also find the bottom directly: set both gradients to zero and solve. Stack a column of 1s (for $b$) next to the features to make matrix $X$, and put all parameters in $\boldsymbol{\theta} = [b, w_1, \ldots, w_n]$. Setting the gradient to zero gives:

$$
\boldsymbol{\theta} = (X^{\top}X)^{-1}X^{\top}\mathbf{y}
$$

Where it comes from: the gradient in matrix form is $\frac{1}{m}X^{\top}(X\boldsymbol{\theta} - \mathbf{y})$. Set that to zero: $X^{\top}X\boldsymbol{\theta} = X^{\top}\mathbf{y}$. Multiply both sides by the inverse of $X^{\top}X$.

For the six houses this gives $b = 0.487$, $w = 1.741$, exactly what gradient descent converged to.

| | Gradient descent | Normal equation |
|---|---|---|
| Needs α and iterations | Yes | No |
| Feature scaling | Strongly helps | Not needed |
| Speed with many features | Fine | Slow: inverting an n × n matrix is roughly n³ work |
| Works for other models | Yes (logistic regression, neural nets) | Only linear regression |
| Breaks when | α badly chosen | $X^{\top}X$ not invertible (perfectly duplicated features) |

sklearn's `LinearRegression` solves the least-squares problem directly (a stable version of this idea), not by gradient descent.

## 9. Code

### From scratch with numpy

```python
import numpy as np

# --- data: 6 houses ---
x = np.array([0.8, 1.2, 1.5, 2.0, 2.4, 3.0])   # size (100 m^2)
y = np.array([1.9, 2.4, 3.3, 3.8, 4.9, 5.6])   # price (GBP 100k)
m = len(x)                                     # number of examples

def cost(w, b):
    f = w * x + b                              # predictions for every house
    return ((f - y) ** 2).sum() / (2 * m)      # squared error cost J(w, b)

def gradients(w, b):
    err = (w * x + b) - y                      # residuals f - y
    dj_dw = (err * x).mean()                   # (1/m) * sum(err * x)
    dj_db = err.mean()                         # (1/m) * sum(err)
    return dj_dw, dj_db

w, b, alpha = 0.0, 0.0, 0.1                    # start at zero, learning rate 0.1
for i in range(2000):
    dj_dw, dj_db = gradients(w, b)             # compute BOTH gradients first...
    w, b = w - alpha * dj_dw, b - alpha * dj_db  # ...then update both together
    if i % 500 == 0:
        print(f"iter {i:4d}  J = {cost(w, b):.4f}")
print(f"w = {w:.3f}, b = {b:.3f}")
```

Output: the cost drops from 2.0969 after the first step to 0.0140, and it prints `w = 1.741, b = 0.487`.

### sklearn

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[0.8], [1.2], [1.5], [2.0], [2.4], [3.0]])  # 2-D: rows = houses, cols = features
y = np.array([1.9, 2.4, 3.3, 3.8, 4.9, 5.6])

model = LinearRegression()          # uses least squares (no learning rate needed)
model.fit(X, y)                     # learns w and b
print(model.coef_, model.intercept_)  # w (one per feature), b
print(model.predict([[2.2]]))       # price for a 220 m^2 house
```

Output: `[1.741] 0.487` and a prediction of 4.317 (about £432k).

> **Trap:** sklearn wants `X` as a **2-D** array (rows × features), even with one feature. `X = df[["size"]]` (double brackets) not `df["size"]`.

## 10. Assumptions & caveats

Least squares will always draw *a* line. These assumptions decide whether the line, and especially the **coefficients and their p-values**, can be trusted. Prediction alone needs fewer of them than inference (explaining which feature matters, and by how much).

| Assumption | Plain words | What breaks if violated | How to check |
|---|---|---|---|
| **Linearity** | y really does change in a straight-line way with each x | Systematic errors; the line misses a curve | Residuals vs fitted plot shows a curve |
| **Independence of errors** | One row's error tells you nothing about the next | Standard errors too small, p-values too optimistic | Think about how data was collected; time series → check autocorrelation |
| **Homoscedasticity** | Errors have the same spread everywhere ("same scatter") | Coefficients still fine, but confidence intervals and p-values wrong | Residuals vs fitted: funnel shape = bad |
| **Normality of residuals** | Errors roughly bell-shaped | Only matters for **inference** (CIs, p-values) with small samples; prediction is fine | Histogram or Q-Q plot of residuals |
| **No / low multicollinearity** | Features are not near-copies of each other | Coefficients unstable and uninterpretable | Correlation matrix, VIF |

### Diagnostic plot: residuals vs fitted

![Residuals plotted against fitted values for three models. Left: a random horizontal band around zero, which is what you want. Middle: a U-shaped curve, so the relationship isn't linear. Right: a funnel that widens to the right, so the error spread grows with the prediction (heteroscedasticity).](figures/linreg-diagnostics.svg)

**Read the figure:** plot fitted values $\hat{y}$ on the x-axis and residuals on the y-axis. A **good** model leaves only noise: a shapeless band centred on 0 (left). A **curve** (middle) means the model is missing a pattern; try a transformed or polynomial feature. A **funnel** (right) means bigger predictions have bigger errors; try predicting $\log(y)$ (see [Logs & exponents](#/t/logs)).

### Multicollinearity in detail

**Multicollinearity** = two or more features carry almost the same information (size in m² and number of rooms; temperature in °C and °F).

Why it hurts: the model is asked "how much of the price is due to size, and how much to rooms?". If the two always move together, the data can't answer. Many combinations fit equally well: (3, 3), (6, 0), (0, 6), even (9, −3). The contour bowl becomes a long flat trough, and tiny changes in the data slide the answer along it.

**Symptoms:**

- Coefficients **change wildly** when you add/remove a few rows or a feature.
- Coefficients take the **wrong sign** (rooms gets a negative weight even though more rooms obviously means pricier).
- Large standard errors: features look "not significant" individually even though together they predict well.
- Predictions are often still fine: it's the **interpretation** that breaks.

![Fit the same model on 40 fresh samples each time; the true weights are both 3 (gold line). Left: when x1 and x2 are unrelated, the fitted weights cluster tightly around 3. Right: when x2 is almost a copy of x1, the weights scatter widely, and some go negative.](figures/linreg-multicollinearity.svg)

**Read the figure:** the right panel is the "sign-flipping, unstable coefficients" symptom made visible. Same model, same true relationship; only the overlap between the features changed.

**VIF (variance inflation factor).** For feature $j$, regress it on all the *other* features and get $R_j^2$ (how well the others predict it). Then

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

- $R_j^2 = 0$ (feature is independent of the rest) → VIF = 1, no inflation.
- $R_j^2 = 0.9$ → VIF = 10: the variance of $w_j$ is **10 times** larger than it would be with no overlap.
- Rules of thumb: VIF above **5** is worth a look, above **10** is a problem.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

rng = np.random.default_rng(0)
n = 200
size_m2 = rng.normal(100, 20, n)                    # floor area
rooms = size_m2 / 25 + rng.normal(0, 0.2, n)        # rooms: almost a copy of size
age = rng.normal(30, 10, n)                         # unrelated to size
df = pd.DataFrame({"size_m2": size_m2, "rooms": rooms, "age": age})

X = sm.add_constant(df)                             # VIF needs the intercept column
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns)
print(vif.drop("const").round(1))                   # ignore the constant's VIF
```

Output: `size_m2 14.7`, `rooms 14.7`, `age 1.0`. Size and rooms flag each other; age is clean.

**Fixes:** drop one of the pair; combine them (e.g. m² per room); use [Regularisation](#/t/regularisation) (Ridge shares weight between correlated features and stabilises them); or accept it if you only need predictions.

### Outliers and extrapolation

![Left: one extreme point (orange X) drags the whole least-squares line towards it, because its large residual is squared. Right: the data only covers x from 0 to 4; a straight line fits well there, but beyond it (shaded) the line and the true curve separate.](figures/linreg-outlier.svg)

- **Outliers.** Squaring makes one huge residual dominate the cost, so a single bad point can tilt the line (left panel). Check for data errors first; consider a robust loss (Huber) or a robust metric (MAE). See [Quartiles, IQR & outliers](#/t/quartiles).
- **Extrapolation.** The model is only checked where you have data. Predicting outside that range (right panel, shaded) assumes the straight line continues forever, which it rarely does.

## 11. Use cases

- **A first baseline** for any numeric target. If a fancy model can't beat it, the fancy model isn't earning its keep.
- **Explaining effects:** "each extra bedroom adds £X, holding size fixed", e.g. pricing, economics, A/B effect sizes.
- **Forecasting simple trends** (demand vs temperature, sales vs ad spend) where relationships are close to linear.
- **When interpretability is required** (regulated settings, stakeholder reports).
- **As a building block:** logistic regression and neural network layers are the same $\mathbf{w}\cdot\mathbf{x}+b$ with something added on top.

## 12. Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Simple, fast to train, even on big data | Only captures straight-line relationships unless you engineer features |
| Coefficients are directly interpretable | Sensitive to outliers (squared error) |
| Convex cost: one global minimum, no luck involved | Coefficients unreliable under multicollinearity |
| Exact closed-form solution exists | Inference relies on assumptions (independence, equal variance, normality) |
| Low variance, hard to overfit with few features | High bias on complex patterns (underfits) |
| Gives a solid baseline and diagnostic plots | Extrapolates blindly beyond the data |

## 13. Limitations

- **It can only draw flat shapes.** Curves, thresholds and interactions ("bedrooms matter only in London") must be added by hand as features (polynomials, logs, products).
- **Correlation, not causation.** A coefficient is an association with the other features held fixed in the data, not proof that changing $x$ causes $y$ to change.
- **Many features vs few rows** → overfitting; plain least squares has no brake. That's what [Regularisation](#/t/regularisation) adds.
- **Unbounded output.** It will happily predict negative prices or probabilities above 1, which is why classification uses [Logistic regression](#/t/logreg).
- **Scoring it** needs the right metric; see [Regression metrics](#/t/regmetrics).

## Quick reference

| Term | Formula / meaning |
|---|---|
| Model | $f_{w,b}(x) = wx + b$; vector form $\mathbf{w}\cdot\mathbf{x} + b$ |
| w (weight, slope) | Change in prediction per 1-unit rise in x (others held fixed) |
| b (bias, intercept) | Prediction when every x = 0 |
| Residual | $\hat{y} - y$, the vertical miss |
| Cost | $J = \frac{1}{2m}\sum(\hat{y} - y)^2$; ½ cancels the 2 from differentiating |
| Gradient w | $\frac{1}{m}\sum(\hat{y} - y)x$ |
| Gradient b | $\frac{1}{m}\sum(\hat{y} - y)$ |
| Update | $w := w - \alpha\,\partial J/\partial w$, simultaneously for b |
| Learning rate α | Step size; too small = slow, too big = diverges |
| Normal equation | $\boldsymbol{\theta} = (X^{\top}X)^{-1}X^{\top}\mathbf{y}$ |
| VIF | $1/(1 - R_j^2)$; above 5 look, above 10 act |
| Good residual plot | Shapeless band around 0 |
