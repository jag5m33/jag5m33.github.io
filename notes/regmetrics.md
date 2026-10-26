# Regression metrics

Regression metrics turn a list of prediction errors into one number, and each metric answers a slightly different question about how wrong the model is.

## The running example

Let's say a model has predicted the prices of six houses (in £k):

| House | Actual y | Predicted ŷ | Error e = y − ŷ | e² | abs(e) |
|---|---|---|---|---|---|
| 1 | 200 | 210 | −10 | 100 | 10 |
| 2 | 250 | 240 | 10 | 100 | 10 |
| 3 | 310 | 330 | −20 | 400 | 20 |
| 4 | 180 | 170 | 10 | 100 | 10 |
| 5 | 420 | 380 | 40 | 1600 | 40 |
| 6 | 300 | 310 | −10 | 100 | 10 |
| **Sum** | | | **10** | **2400** | **100** |

- **Error** (or **residual**): how far off one prediction is.
- $n$: the number of predictions. Here $n = 6$.

The plain errors add up to only 10, because positives and negatives cancel. That's why every metric first squares the errors or drops their sign.

## 1. MSE: mean squared error

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2
$$

- **In words:** the average of the squared errors.
- **Example:** 2400 / 6 = **400**.
- **Units:** squared (£k²), which is hard to read. MSE is for training and comparing models, not for reporting.
- Big errors count much more: house 5 alone gives 1600 of the 2400.

> **Key idea:** the linear regression cost $J$ is just MSE divided by 2 (see [Linear regression](#/t/linreg)). Minimising one minimises the other.

## 2. RMSE: root mean squared error

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

- **In words:** the square root of MSE, which puts it back into the original units.
- **Example:** $\sqrt{400}$ = **20**, i.e. **£20k**.
- **Say it like this:** "A typical prediction is off by about £20k, with big misses weighted more."

## 3. MAE: mean absolute error

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left\lvert y_i - \hat{y}_i\right\rvert
$$

- **In words:** the average size of a miss, ignoring its direction.
- **Example:** 100 / 6 = **16.67**, i.e. about **£16.7k**.
- Every pound of error counts the same, so MAE is **less affected by outliers** than RMSE.

## 4. RMSE vs MAE when an outlier appears

Now say house 5 is badly predicted at 220 instead of 380 (an error of 200):

| | Original | House 5 badly wrong |
|---|---|---|
| MAE | 16.7 | 43.3 |
| RMSE | 20.0 | 82.5 |

MAE grew 2.6 times; RMSE grew 4.1 times. Squaring blows up the one big miss.

![Each house's share of the total error. Left: original predictions; house 5 already dominates MSE (red bars) more than MAE (teal bars). Right: with house 5 badly wrong, it supplies 98% of the MSE but a smaller share of the MAE.](figures/regmetrics-outlier.svg)

In the figure, look at house 5's red bar on the right: under MSE one bad prediction becomes almost the whole score.

**Which to use:**

- Big errors are especially bad (e.g. under-forecasting peak electricity demand) → **RMSE**.
- Every unit of error costs the same, or there are outliers → **MAE**.
- Report both. If RMSE is much bigger than MAE, there are a few big misses.

## 5. R²: how much better than guessing the mean?

R² compares your model with the laziest possible model: **always predict the average price**.

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

- $SS_{\text{res}}$: the sum of squared errors of **your model** = 2400.
- $SS_{\text{tot}}$: the sum of squared errors of the **"always predict the mean"** model. The mean price is 276.67, and this comes to 38,133.

**Example:** $R^2 = 1 - 2400 / 38133 = $ **0.937**.

![Left: the gold vertical lines are the misses of a model that always predicts the mean price; their squares add to SS_tot = 38,133. Right: the teal lines are the misses of your model; their squares add to SS_res = 2,400. R² is how much of the left-hand total the model removed.](figures/regmetrics-r2.svg)

In the figure, the gold misses on the left are long and the teal misses on the right are short. The model removed about 94% of the squared miss.

> **Say it like this:** "The model explains about 94% of the variation in house prices." It does **not** mean "94% accurate".

| R² | Meaning |
|---|---|
| 1 | Perfect predictions |
| 0 | No better than always predicting the mean |
| Below 0 | Worse than predicting the mean (possible on test data) |

**Traps:**

- R² **always goes up** on training data when you add features, even pure noise.
- R² can be **negative** on test data. A broken model predicting 400 for every house scores −2.39. That means "throw it away", not a bug.
- Only compare R² between models on the **same** data.

## 6. Adjusted R²: the noise tax

$$
R^2_{\text{adj}} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}
$$

- $n$: number of rows. $p$: number of features.
- It charges a small "tax" for every feature you add.

From Jag's statistics notes:

> **Plain r² asks "how well did I fit?" Adjusted r² asks "how well did I fit, given how many chances I gave myself?"**

- A **junk** feature raises plain R² a little by luck; the tax cancels it, so adjusted R² stays flat or drops.
- A **useful** feature raises R² by much more than the tax, so adjusted R² still goes up.

**Example:** our R² of 0.937 with $n = 6$ gives an adjusted R² of 0.921 if it used 1 feature, 0.895 with 2 features, and 0.843 with 3.

![50 rows with one real feature, then pure-noise columns added one at a time. Plain R² (red) climbs steadily from 0.58 to 0.74 on nothing but noise. Adjusted R² (teal) wanders around its starting value and ends lower than it began.](figures/regmetrics-adjr2.svg)

In the figure, the red line rising is an illusion; the flat teal line is the honest verdict that nothing useful was added.

## 7. MAPE: mean absolute percentage error

- **In words:** the average miss as a percentage of the true value.
- **Example:** the six % misses are 5.00, 4.00, 6.45, 5.56, 9.52, 3.33, which average to **5.64%**.
- Easy for stakeholders, but it **breaks when y is zero or close to zero** (you divide by y).
- sklearn's `mean_absolute_percentage_error` returns 0.0564, not 5.64. Multiply by 100.

## 8. Always look at the residuals

One number hides **where** the errors are.

![Simulated 80 houses. Left: predicted vs actual; a good model's points hug the dashed diagonal. Right: residuals against the prediction; a shapeless band around zero, mostly inside ±1 RMSE (shaded), is what you want.](figures/regmetrics-residuals.svg)

In the figure, the right panel shows no pattern, which is good. A curve or a funnel shape would mean the model is missing something (examples in [Linear regression](#/t/linreg)).

## 9. Code

```python
import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)

y_true = np.array([200, 250, 310, 180, 420, 300])   # actual prices (GBP k)
y_pred = np.array([210, 240, 330, 170, 380, 310])   # predictions

mse  = mean_squared_error(y_true, y_pred)            # 400.0
rmse = np.sqrt(mse)                                  # 20.0, back in GBP k
mae  = mean_absolute_error(y_true, y_pred)           # 16.67
r2   = r2_score(y_true, y_pred)                      # 0.937
mape = mean_absolute_percentage_error(y_true, y_pred) * 100   # 5.64 (%)

n, p = len(y_true), 1                                # rows, features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)        # 0.921 (no sklearn function)
```

> **Trap:** old tutorials write `mean_squared_error(..., squared=False)` for RMSE. That option has been removed from recent sklearn. Use `np.sqrt(...)` or `root_mean_squared_error`.

## Quick reference

| Metric | In words | Example | Better |
|---|---|---|---|
| MSE | Average squared miss | 400 | Lower |
| RMSE | Square root of MSE; typical miss in y's units | £20k | Lower |
| MAE | Average size of a miss; robust to outliers | £16.7k | Lower |
| R² | Share of variation explained vs predicting the mean | 0.937 | Higher (max 1) |
| Adjusted R² | R² with a tax per feature | 0.921 | Higher |
| MAPE | Average % miss; breaks near y = 0 | 5.64% | Lower |
