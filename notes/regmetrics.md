# Regression metrics

Regression metrics turn a list of prediction errors into one number, and each metric answers a slightly different question about how wrong the model is.

## The running example

Six houses, prices in £k. A model has made these predictions:

| House | Actual y | Predicted ŷ |
|---|---|---|
| 1 | 200 | 210 |
| 2 | 250 | 240 |
| 3 | 310 | 330 |
| 4 | 180 | 170 |
| 5 | 420 | 380 |
| 6 | 300 | 310 |

- **Error** (or **residual**) $e = y - \hat{y}$: how far off one prediction is. Positive = the model guessed too low.
- $n$: the number of predictions. Here $n = 6$.
- Every metric below is built from these six errors. All numbers were computed in Python (code at the end).

## 1. The step-by-step table

Do the calculation in columns, left to right. Every metric is just an average of one of these columns (or a ratio built from them).

| House | y | ŷ | e = y − ŷ | e² | abs(e) | abs(e) / y (%) |
|---|---|---|---|---|---|---|
| 1 | 200 | 210 | −10 | 100 | 10 | 5.00 |
| 2 | 250 | 240 | 10 | 100 | 10 | 4.00 |
| 3 | 310 | 330 | −20 | 400 | 20 | 6.45 |
| 4 | 180 | 170 | 10 | 100 | 10 | 5.56 |
| 5 | 420 | 380 | 40 | 1600 | 40 | 9.52 |
| 6 | 300 | 310 | −10 | 100 | 10 | 3.33 |
| **Sum** | | | **10** | **2400** | **100** | **33.86** |
| **Mean** | | | | **MSE = 400** | **MAE = 16.67** | **MAPE = 5.64%** |

Notice the plain errors sum to 10: positives and negatives nearly cancel. That's why no metric uses the raw average error to measure size.

## 2. MSE: mean squared error

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2
$$

- **In words:** the average of the squared errors.
- **Example:** 2400 / 6 = **400**.
- **Units:** squared units (£k²). "400 thousand-pounds-squared" means nothing to a human. MSE is for **optimising and comparing**, not for reporting.
- **Behaviour:** squaring makes big errors count far more. House 5 (error 40) contributes 1600 of the 2400, two-thirds of the total, from one house out of six.

> **Key idea:** The linear regression cost is $J = \frac{1}{2m}\sum(\hat{y}-y)^2$, which is **MSE / 2**. The ½ only exists to cancel the 2 when differentiating (see [Linear regression](#/t/linreg)). Halving doesn't move the minimum, so "minimise J" and "minimise MSE" give the same line. Report MSE (1/n); train on J (1/2m).

## 3. RMSE: root mean squared error

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2}
$$

- **In words:** the square root of MSE, which puts it back into the **original units**.
- **Example:** $\sqrt{400}$ = **20**, i.e. **£20k**.
- **Say it like this:** "A typical prediction is off by about £20k, with big misses weighted more heavily."
- RMSE is always at least as big as MAE. The further apart they are, the more your errors include a few big misses.

## 4. MAE: mean absolute error

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left\lvert y_i - \hat{y}_i\right\rvert
$$

- **In words:** the average size of a miss, ignoring direction.
- **Example:** 100 / 6 = **16.67**, i.e. **£16.7k**.
- **Say it like this:** "On average, predictions are off by about £16.7k."
- **Behaviour:** every pound of error counts the same, so one huge miss doesn't dominate. More **robust to outliers** than MSE/RMSE.

## 5. MSE vs MAE when an outlier appears

Change one prediction: house 5 is now predicted at 220 instead of 380 (an error of 200).

| | Original | House 5 badly wrong |
|---|---|---|
| MAE | 16.7 | 43.3 |
| RMSE | 20.0 | 82.5 |

MAE grew 2.6×. RMSE grew 4.1×. Squaring amplifies the one big miss.

![Each house's share of the total error. Left: original predictions; house 5 already dominates MSE (red bars) more than MAE (teal bars). Right: with house 5 badly wrong, it supplies 98% of the MSE but a smaller share of the MAE.](figures/regmetrics-outlier.svg)

**Read the figure:** compare the red bar for house 5 in the two panels. Under MSE one bad prediction becomes almost the **whole** score; the other five houses barely register. Under MAE (teal) the other houses still carry visible weight.

**Which to choose:**

- **Large errors are genuinely worse than several small ones** (under-forecasting peak electricity demand, a dosing error) → **RMSE / MSE**.
- **Every unit of error costs the same**, or the data has outliers you don't want to dominate → **MAE**.
- Report **both**: if RMSE is much bigger than MAE, look for a handful of large misses.

## 6. R²: coefficient of determination

R² compares your model to the laziest possible model: **always predict the mean** of y.

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- $\bar{y}$: the mean of the actual values. Here 276.67.
- $SS_{\text{res}}$ (residual sum of squares): total squared misses of **your model** = 2400.
- $SS_{\text{tot}}$ (total sum of squares): total squared misses of the **mean-only model** = 38,133. This is how much y varies in the first place.

**Step by step:**

| House | y | y − ȳ | (y − ȳ)² |
|---|---|---|---|
| 1 | 200 | −76.67 | 5877.8 |
| 2 | 250 | −26.67 | 711.1 |
| 3 | 310 | 33.33 | 1111.1 |
| 4 | 180 | −96.67 | 9344.4 |
| 5 | 420 | 143.33 | 20544.4 |
| 6 | 300 | 23.33 | 544.4 |
| **Sum** | | | **38133.3** |

$R^2 = 1 - 2400 / 38133 = 1 - 0.063$ = **0.937**.

![Left: the gold vertical lines are the misses of a model that always predicts the mean price; their squares add to SS_tot = 38,133. Right: the teal lines are the misses of your model; their squares add to SS_res = 2,400. R² is how much of the left-hand total the model removed.](figures/regmetrics-r2.svg)

**Read the figure:** the gold lines on the left are long, the teal lines on the right are short. R² = 0.937 says the model removed 93.7% of the squared miss that the mean-only guess leaves behind.

> **Say it like this:** "The model explains about 94% of the variation in house prices." (As in the [Correlation](#/t/correlation) chapter: it is the share of the variation in the **outcome** that the model explains, not "94% accurate".)

| R² | Meaning |
|---|---|
| 1 | Perfect predictions |
| 0 | No better than predicting the mean every time |
| Below 0 | **Worse** than predicting the mean |

## 7. R² traps

**Trap 1: R² always rises when you add features (on training data).** Any extra column, even random noise, can fit a scrap of the training data by luck, and plain R² counts that as explanation. It is a one-way ratchet.

**Trap 2: R² can be negative on test data.** On the data the model was trained on (with an intercept), R² can't go below 0. On **new** data it can, if the model does worse than predicting the mean. Example: a broken model predicting 400 for all six houses scores R² = **−2.39**. A negative test R² means "throw it away", not "a calculation error".

**Trap 3: R² depends on the spread of y.** The same RMSE gives a high R² when y varies a lot and a low R² when y varies little. Compare models on the **same** data only.

**Trap 4: a high R² doesn't mean the model is right.** A curved relationship fitted with a straight line can still score well; the residual plot (below) will show the curve.

## 8. Adjusted R²: the noise tax

$$
R^2_{\text{adj}} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}
$$

- $n$: number of rows. $p$: number of features (predictors).
- The fraction $\frac{n-1}{n-p-1}$ is always above 1 and **grows as p grows relative to n**. It inflates the "unexplained" part, $(1 - R^2)$, before subtracting.

From Jag's statistics notes:

> **Plain r² asks "how well did I fit?" Adjusted r² asks "how well did I fit, given how many chances I gave myself?"**

Adjusted R² **doesn't inspect the features**. It knows only two numbers: how many rows and how many predictors. Even a useless variable raises plain R² by a predictable amount purely by luck, so adjusted R² charges **every** feature the same **noise tax**:

- **A junk variable** raises plain R² by about the lucky amount → the tax cancels it → adjusted R² stays flat or dips.
- **A real variable** raises plain R² by far more than the lucky amount → the tax is small next to the gain → adjusted R² still rises.
- **The less data per variable, the heavier the tax.** With 50 rows and 50 variables, plain R² hits 1.0 on pure garbage.

**Worked example** (our six houses, R² = 0.937): if that R² came from

| Features p | Adjusted R² |
|---|---|
| 1 | 0.921 |
| 2 | 0.895 |
| 3 | 0.843 |

Same fit, more "chances" used, bigger tax.

![50 rows with one real feature, then pure-noise columns added one at a time. Plain R² (red) climbs steadily from 0.58 to 0.74 on nothing but noise. Adjusted R² (teal) wanders around its starting value and ends lower than it began.](figures/regmetrics-adjr2.svg)

**Read the figure:** the gap between the red and teal lines is the tax. Red rising is an illusion; teal staying flat is the honest verdict that nothing useful was added.

> **Key idea:** In ML we usually go one step further and score R² on a **validation set**. Noise features can't fool a held-out set, so validation R² does the same job as adjusted R², and works for any model.

## 9. MAPE: mean absolute percentage error

$$
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left\lvert\frac{y_i - \hat{y}_i}{y_i}\right\rvert
$$

- **In words:** the average miss as a percentage of the true value.
- **Example:** mean of (5.00, 4.00, 6.45, 5.56, 9.52, 3.33) = **5.64%**.
- **Say it like this:** "Predictions are off by about 5.6% on average." Very easy for stakeholders.

**Traps:**

- **Breaks when y is zero or near zero:** dividing by a tiny y explodes the percentage. Useless for targets that can be 0 (sales of a slow product, net flows).
- **Asymmetric:** over-forecasting can cost more than 100%, under-forecasting can't (the worst under-forecast is predicting 0, a 100% miss). It quietly favours models that predict low.
- sklearn's `mean_absolute_percentage_error` returns a **fraction** (0.0564), not 5.64. Multiply by 100 yourself.

## 10. The residual plot

A single number hides **where** the errors are. Always plot them.

![Simulated 80 houses. Left: predicted vs actual; a good model's points hug the dashed diagonal. Right: residuals against the prediction; a shapeless band around zero, mostly inside ±1 RMSE (shaded), is what you want.](figures/regmetrics-residuals.svg)

**Read the figure:** on the right, look for **patterns**. A curve means the model misses a non-linear shape; a funnel means errors grow with the prediction (heteroscedasticity); a few points far outside the band are outliers worth reading. Here there is no pattern, so the metric summarises the errors fairly. More examples of bad residual plots are in [Linear regression](#/t/linreg).

## 11. When to use each

| Metric | Units | Outliers | Use when |
|---|---|---|---|
| MSE | squared | very sensitive | Training / optimising; comparing models |
| RMSE | same as y | sensitive | Reporting when big errors are especially bad |
| MAE | same as y | robust | Reporting a typical miss; data has outliers |
| R² | none (0 to 1, can be negative on test) | sensitive | "How much better than the mean?"; same dataset only |
| Adjusted R² | none | sensitive | Comparing linear models with different numbers of features |
| MAPE | % | robust-ish | Business reporting; y never near zero |

## 12. Code

```python
import numpy as np
from sklearn.metrics import (mean_squared_error, root_mean_squared_error,
                             mean_absolute_error, r2_score,
                             mean_absolute_percentage_error)

y_true = np.array([200, 250, 310, 180, 420, 300])   # actual prices (GBP k)
y_pred = np.array([210, 240, 330, 170, 380, 310])   # model's predictions

mse  = mean_squared_error(y_true, y_pred)            # average squared error
rmse = root_mean_squared_error(y_true, y_pred)       # sklearn >= 1.4
# rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # works in any version
mae  = mean_absolute_error(y_true, y_pred)           # average absolute error
r2   = r2_score(y_true, y_pred)                      # share of variance explained
mape = mean_absolute_percentage_error(y_true, y_pred)  # a FRACTION, not a %

n, p = len(y_true), 1                                # rows, number of features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)        # no sklearn function for this

print(f"MSE {mse:.1f}  RMSE {rmse:.1f}  MAE {mae:.2f}")
print(f"R2 {r2:.3f}  adj R2 {adj_r2:.3f}  MAPE {mape*100:.2f}%")
```

Output: `MSE 400.0  RMSE 20.0  MAE 16.67` and `R2 0.937  adj R2 0.921  MAPE 5.64%`.

> **Trap:** Old tutorials write `mean_squared_error(y_true, y_pred, squared=False)` to get RMSE. That argument was deprecated in sklearn 1.4 and **removed** in 1.6; it now raises an error. Use `root_mean_squared_error` or `np.sqrt(mean_squared_error(...))`.

> **Trap:** Argument order is `(y_true, y_pred)`. For MSE, MAE and RMSE swapping them gives the same answer, but for R² and MAPE it doesn't.

## Quick reference

| Metric | Formula | One line |
|---|---|---|
| Error | $y - \hat{y}$ | How far off one prediction is |
| MSE | $\frac{1}{n}\sum (y-\hat{y})^2$ | Average squared miss; squared units |
| Cost J | $\frac{1}{2m}\sum (\hat{y}-y)^2$ | MSE / 2, for clean gradients |
| RMSE | $\sqrt{\text{MSE}}$ | Typical miss in y's units, big misses weighted more |
| MAE | $\frac{1}{n}\sum \lvert y-\hat{y} \rvert$ | Average miss size; robust to outliers |
| R² | $1 - SS_{\text{res}}/SS_{\text{tot}}$ | Share of variation explained vs predicting the mean |
| Adjusted R² | $1 - (1-R^2)\frac{n-1}{n-p-1}$ | R² minus a noise tax per feature |
| MAPE | $\frac{100\%}{n}\sum \lvert (y-\hat{y})/y \rvert$ | Average % miss; breaks near y = 0 |
