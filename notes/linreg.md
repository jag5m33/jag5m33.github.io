# Linear regression

Linear regression predicts a number by drawing the straight line that makes the squared prediction errors as small as possible, and gradient descent is one way to find that line.

## The running example

Let's say we have six houses. One feature (size), one target (price).

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
- $m$: the number of rows. Here $m = 6$.

## 1. The model: f = wx + b

$$
f_{w,b}(x) = wx + b
$$

- $f_{w,b}(x)$, also written $\hat{y}$ ("y-hat"): the **prediction**.
- $w$: the **weight**, i.e. the **slope**. How much the prediction goes up when $x$ goes up by 1.
- $b$: the **bias** or **intercept**. Where the line crosses the y-axis (the prediction when $x = 0$).
- Training = finding the best $w$ and $b$.

![The six houses (red dots) and the best straight line through them (teal). The gold dashed triangle shows the slope: go 1 unit right and the line rises by w = 1.74. The gold dot on the y-axis is the intercept b = 0.49.](figures/linreg-model.svg)

**Example:** the best line is $w = 1.74$, $b = 0.49$. In the figure, the slope triangle says each extra 100 m² adds about £174k. A 220 m² house ($x = 2.2$) gets $1.74 \times 2.2 + 0.49 \approx 4.32$, about £432k.

> **Say it like this:** "The model is a straight line. w is how steep it is, b is where it starts. Learning means finding the w and b that fit the data best."

## 2. Residuals: how wrong is one prediction?

A **residual** (or **error**) is the vertical gap between the prediction and the truth: $\hat{y} - y$.

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

In the figure, the red lines get longer to the right. That tells you the slope is too small. A good line has short residuals scattered above and below it, with no pattern.

## 3. The cost: squared error

We need one number that scores the whole line. Adding the plain residuals doesn't work: +2 and −2 cancel to 0. So we **square** them (always positive, and big misses count more), then average.

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2
$$

- $J$: the **cost**, one number for how bad the line is. Lower is better.
- $\sum (\ldots)^2$: add up the squared residuals.
- $\frac{1}{m}$: take the average. The extra $\frac{1}{2}$ is just a convenience that makes the maths tidier later; it doesn't change which line is best.

**Example:** for the poor guess, $J = 6.14 / (2 \times 6) = 0.512$. For the best line ($w = 1.74$, $b = 0.49$), $J = 0.014$. Much smaller: that's what "best" means.

> **Key idea:** $J$ is just the average squared error, halved. See [Regression metrics](#/t/regmetrics) for MSE.

### The cost is a bowl

Every pair $(w, b)$ gives one value of $J$. Plot $J$ for all of them and you get a bowl.

![Left: J(w, b) as a 3-D bowl. Right: the same bowl seen from directly above. Each ring is a contour, a set of (w, b) pairs with the same cost, like height lines on a map. The star at the centre is the minimum.](figures/linreg-contour.svg)

In the figure, each ring is a set of lines that are **equally bad**. The star at the bottom is the best line. For linear regression the bowl has **one** bottom only, so you can't get stuck in the wrong place.

## 4. Gradient descent

**Gradient descent** finds the bottom of the bowl step by step:

1. Start with any $w$ and $b$ (for example 0 and 0).
2. Work out the **gradient**: the slope of the bowl where you are standing. It points uphill.
3. Take a small step the **opposite** way (downhill).
4. Repeat until the steps barely change anything. That's called **converging**.

The update rule, one line:

$$
w := w - \alpha\frac{\partial J}{\partial w}
$$

- $:=$ means "replace with".
- $\alpha$ (alpha): the **learning rate**, the step size, e.g. 0.1.
- $\frac{\partial J}{\partial w}$: the slope of the bowl in the $w$ direction. The minus sign turns uphill into downhill.
- $b$ is updated the same way, **at the same time** (work out both slopes first, then update both).

**Example:** from the poor guess ($w = 1$, $b = 1$), the slopes say "increase both". One step with $\alpha = 0.1$ moves $w$ to 1.19 and $b$ to 1.08, and the cost falls from 0.512 to 0.177.

![Left: gradient descent starting at (w=0, b=−2), each dot one step, on the contour plot. Right: the cost after each iteration on a log scale. It falls fast, then flattens once the minimum is reached.](figures/linreg-descent.svg)

In the figure, the dots walk downhill towards the star. On the right, the cost must go **down** every step. If it ever goes up, $\alpha$ is too big or there's a bug.

### Choosing the learning rate

![Gradient descent on J(w) with three learning rates, same start. Too small (left): tiny steps, nowhere near the bottom after 8 steps. About right (middle): reaches the bottom in a few steps. Too big (right): each step jumps past the minimum to a higher point on the other side, so the cost grows.](figures/linreg-alpha.svg)

- **Too small** (left): tiny steps, very slow.
- **About right** (middle): reaches the bottom in a few steps.
- **Too big** (right): jumps over the bottom and climbs the other wall. The cost grows instead of falling.

Andrew Ng's tip: try $\alpha$ = 0.001, 0.01, 0.1, …, plot cost against step for each, and pick the largest one where the cost falls smoothly.

### More than one feature

Real data has many features (size, rooms, age). The model just gets one weight per feature: $f = w_1x_1 + w_2x_2 + \dots + b$. Each $w_j$ means "how much the prediction changes when this feature goes up by 1, with the others held fixed". Gradient descent works the same way. **Scale the features** first, or it zig-zags slowly; see [Scaling](#/t/scaling).

## 5. Code (sklearn)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[0.8], [1.2], [1.5], [2.0], [2.4], [3.0]])  # 2-D: rows = houses, cols = features
y = np.array([1.9, 2.4, 3.3, 3.8, 4.9, 5.6])               # prices

model = LinearRegression()            # finds the best w and b directly
model.fit(X, y)                       # learn w and b from the data
print(model.coef_, model.intercept_)  # w = [1.741], b = 0.487
print(model.predict([[2.2]]))         # 4.317, about GBP 432k for 220 m^2
```

> **Trap:** sklearn wants `X` as a **2-D** table, even with one feature. Use `df[["size"]]` (double brackets), not `df["size"]`.

## 6. Assumptions & caveats

The line can always be drawn. These decide whether you can **trust** it, especially the coefficients.

- **Linearity:** y really changes in a straight-line way with x. If not, the line misses a curve.
- **Independent errors:** one row's error tells you nothing about the next. Often broken in time series.
- **Equal spread (homoscedasticity):** errors are about the same size everywhere, not fanning out.
- **Roughly normal residuals:** errors look bell-shaped. Only matters for p-values and confidence intervals, not for prediction.
- **Outliers:** squaring makes one huge miss dominate, so one bad point can tilt the whole line. See [Quartiles, IQR & outliers](#/t/quartiles).
- **Extrapolation:** don't trust predictions far outside the range of your data.

![Residuals plotted against fitted values for three models. Left: a random horizontal band around zero, which is what you want. Middle: a U-shaped curve, so the relationship isn't linear. Right: a funnel that widens to the right, so the error spread grows with the prediction (heteroscedasticity).](figures/linreg-diagnostics.svg)

To check, plot residuals against predictions. In the figure, the left panel is good (a shapeless band around 0). The middle curve means a straight line is the wrong shape. The right funnel means errors grow with the prediction; predicting $\log(y)$ often helps (see [Logs & exponents](#/t/logs)).

### Multicollinearity

**Multicollinearity** = two features carry almost the same information (size in m² and number of rooms).

- The model can't tell how much of the price is due to size and how much to rooms, so many weight combinations fit equally well.
- **Symptoms:** weights jump around when you add or remove a few rows; a weight gets the wrong sign (more rooms → cheaper?).
- Predictions are often still fine. It's the **interpretation** that breaks.
- **VIF** (variance inflation factor) is a number per feature that flags it: above about 5–10 is a warning.
- **Fixes:** drop one of the pair, combine them, or use [Regularisation](#/t/regularisation) (Ridge).

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = sm.add_constant(df[["size_m2", "rooms", "age"]])   # VIF needs an intercept column
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns)
print(vif.drop("const").round(1))   # e.g. size_m2 14.7, rooms 14.7, age 1.0
```

## 7. Use cases

- **A first baseline** for any number you want to predict.
- **Explaining effects:** "each extra bedroom adds £X, holding size fixed".
- **Simple trends** such as demand vs temperature.
- **When the model must be easy to explain** to stakeholders.

## 8. Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Simple and fast | Only straight-line patterns unless you add features |
| Weights are easy to interpret | Sensitive to outliers |
| One best answer (the bowl has one bottom) | Weights unreliable with multicollinearity |
| Hard to overfit with few features | Underfits complex patterns |
| Great baseline | Extrapolates blindly beyond the data |

## 9. Limitations

- A weight shows association, not causation.
- With many features and few rows it can overfit; [Regularisation](#/t/regularisation) adds a brake.
- It can predict impossible values (negative prices), which is why classification uses [Logistic regression](#/t/logreg).

## Quick reference

| Term | Meaning |
|---|---|
| Model | $f = wx + b$ |
| w (weight, slope) | Change in prediction per 1-unit rise in x |
| b (bias, intercept) | Where the line crosses the y-axis |
| Residual | $\hat{y} - y$, the vertical miss |
| Cost J | Average squared residual, halved; a bowl |
| Gradient descent | Step downhill on the bowl until it flattens |
| Update | $w := w - \alpha\,\partial J/\partial w$ (same for b, at the same time) |
| Learning rate α | Step size: too small = slow, too big = diverges |
| VIF | Flags multicollinearity; above ~5–10 is a warning |
| Good residual plot | Shapeless band around 0 |
