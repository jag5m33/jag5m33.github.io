# Regression metrics

Regression metrics turn a list of prediction errors into one number, and each one answers a slightly different question about how wrong a model that predicts numbers is.

## The running example: six house prices

Let's say a model has predicted the prices of six houses (in £k):

| House | Actual y | Predicted ŷ | Error y − ŷ | Error² | Size of error |
|---|---|---|---|---|---|
| 1 | 200 | 210 | −10 | 100 | 10 |
| 2 | 250 | 240 | 10 | 100 | 10 |
| 3 | 310 | 330 | −20 | 400 | 20 |
| 4 | 180 | 170 | 10 | 100 | 10 |
| 5 | 420 | 380 | 40 | 1,600 | 40 |
| 6 | 300 | 310 | −10 | 100 | 10 |
| **Sum** | | | **10** | **2,400** | **100** |

- **Error** (also called the **residual**): actual minus predicted, for one house.
- $n$: the number of predictions, here 6.

The plain errors add up to only 10 because positives and negatives cancel. That's why every metric first **squares** the errors or **drops their sign**.

> **Key idea:** most regression metrics don't care which model made the predictions. [Linear regression](#/t/linreg), a [decision tree](#/t/trees) regressor, a [random forest](#/t/randomforest) or [XGBoost](#/t/xgboost) regressor can all be scored with every metric below. What changes the choice is **which errors hurt**.

## MSE: mean squared error

> **What you need to know**
> - **In plain words:** the average of the squared errors
> - **In context:** 2,400 / 6 = **400** (£k²)
> - **Best with:** any regressor; it's what linear regression **minimises** while training
> - **Use it when:** training and comparing models; rarely for reporting, because the units are squared
> - **Watch out:** £k² means nothing to a manager; take the square root (RMSE) before you report

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

$y_i$ is the actual value, $\hat{y}_i$ the prediction, and the sum runs over all $n$ rows. House 5 alone gives 1,600 of the 2,400: squaring makes big misses count much more.

Andrew Ng's cost $J$ is MSE with $\frac{1}{2m}$ in front instead of $\frac{1}{n}$; the ½ only tidies the gradient-descent maths and doesn't change which model is best.

## RMSE: root mean squared error

> **What you need to know**
> - **In plain words:** the square root of MSE, so it's back in the units of y: "a typical miss"
> - **In context:** $\sqrt{400}$ = **20**, so a typical prediction is off by about **£20k**
> - **Best with:** any regressor; the usual headline score for linear regression and for tree/forest/XGBoost regressors
> - **Use it when:** big errors are much worse than small ones: under-forecasting peak electricity demand, under-stocking before Christmas, a house valuation that's off by £150k
> - **Watch out:** one bad outlier can dominate it (see below)

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

## MAE: mean absolute error

> **What you need to know**
> - **In plain words:** the average size of a miss, ignoring whether it was too high or too low
> - **In context:** 100 / 6 = **16.7**, so a typical miss of about **£16.7k**
> - **Best with:** any regressor
> - **Use it when:** outliers are noise you don't want to chase (a typo'd sale price), or every £ of error costs the same (delivery-time estimates)
> - **Watch out:** it treats one £40k miss the same as four £10k misses; if big misses are much worse, use RMSE

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i \rvert
$$

The bars $\lvert\ \rvert$ mean "drop the minus sign".

### RMSE vs MAE when one house goes badly wrong

Now say house 5 is predicted at 220 instead of 380 (a miss of 200):

| | Original | House 5 badly wrong | Grew by |
|---|---|---|---|
| MAE | 16.7 | 43.3 | 2.6 times |
| RMSE | 20.0 | 82.5 | 4.1 times |

![Each house's share of the total error. Left: original predictions; house 5 already dominates MSE (red bars) more than MAE (teal bars). Right: with house 5 badly wrong, it supplies 98% of the MSE but a smaller share of the MAE.](figures/regmetrics-outlier.svg)

In the figure, look at house 5's red bar on the right: under squared error one bad prediction becomes almost the whole score.

> **Say it like this:** "I report both. If RMSE is much bigger than MAE, a few big misses are hiding in there."

## R²: how much better than guessing the mean?

> **What you need to know**
> - **In plain words:** the share of the variation in y that the model explains, compared with a lazy model that always predicts the **average**
> - **In context:** the mean price is £276.7k; always predicting it gives squared errors totalling 38,133; our model's total is 2,400; so R² = 1 − 2,400 / 38,133 = **0.937**
> - **Best with:** any regressor; it's what `.score()` returns for every sklearn regressor; most often quoted for linear regression
> - **Use it when:** explaining a model to non-specialists ("explains 94% of the variation"); comparing models **on the same data**
> - **Watch out:** it is **not** "94% accurate"; and it always rises on training data when you add features, even useless ones
>
> **Skip:** the "sum of squares" names beyond the two totals below

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

$SS_{\text{res}}$ is the sum of squared errors of **your model** (2,400). $SS_{\text{tot}}$ is the sum of squared errors of the **"always predict the mean"** model (38,133).

![Left: the gold lines are the misses of a model that always predicts the mean price; their squares add to 38,133. Right: the teal lines are your model's misses; their squares add to 2,400. R² is how much of the left-hand total the model removed.](figures/regmetrics-r2.svg)

In the figure, the gold misses on the left are long and the teal ones on the right are short: the model removed about 94% of the squared miss.

| R² | Meaning |
|---|---|
| 1 | perfect predictions |
| 0 | no better than always predicting the mean (the **baseline**) |
| below 0 | worse than the mean: a model predicting 400 for every house scores −2.39 |

## Adjusted R²: the noise tax

> **What you need to know**
> - **In plain words:** R² with a small penalty for every feature you add
> - **In context:** our R² of 0.937 on 6 houses becomes 0.921 if the model used 1 feature, 0.895 with 2 and 0.843 with 3
> - **Best with:** linear regression with several features
> - **Use it when:** deciding whether an extra feature is worth keeping
> - **Watch out:** it's a training-data check; validation scores are still the real test
>
> **Skip:** the formula

From Jag's statistics notes: **"Plain r² asks 'how well did I fit?' Adjusted r² asks 'how well did I fit, given how many chances I gave myself?'"** A junk feature nudges R² up by luck, and the penalty cancels it; a useful feature beats the penalty.

![50 rows with one real feature, then pure-noise columns added one at a time. Plain R² (red) climbs steadily on nothing but noise. Adjusted R² (teal) stays around its starting value.](figures/regmetrics-adjr2.svg)

In the figure, the rising red line is an illusion; the flat teal line is the honest verdict that nothing useful was added. More in [Correlation](#/t/correlation).

## MAPE: mean absolute percentage error

> **What you need to know**
> - **In plain words:** the average miss as a percentage of the true value
> - **In context:** the six misses are 5.0%, 4.0%, 6.5%, 5.6%, 9.5% and 3.3% of the true prices; their average is **5.6%**
> - **Best with:** any regressor
> - **Use it when:** talking to stakeholders ("off by about 6% on average"); comparing across scales (cheap and expensive products)
> - **Watch out:** it **breaks when y is zero or near zero** (you divide by y); sklearn returns 0.056, not 5.6, so multiply by 100
>
> **Skip:** the formula

## Which one?

| Situation | Use | Why |
|---|---|---|
| Report a typical miss in real units | RMSE or MAE | same units as y |
| Big misses are much worse (peak demand, stock-outs) | **RMSE** | squaring punishes big misses |
| Outliers are noise; every £ costs the same | **MAE** | each unit of error counts once |
| Explain to non-specialists how much it captures | **R²** | 0 = mean baseline, 1 = perfect |
| Is this extra feature worth it? | adjusted R² (and validation RMSE) | penalises useless features |
| Percentages for a business audience, y never near 0 | MAPE | easy to read |
| Training a linear regression | MSE | it's the cost being minimised |

> **Trap:** one number hides **where** the errors are. Always plot the residuals too ([Linear regression](#/t/linreg) shows what good and bad patterns look like).

## Code

```python
import numpy as np
from sklearn.metrics import (mean_squared_error, root_mean_squared_error,
                             mean_absolute_error, r2_score)

y_true = np.array([200, 250, 310, 180, 420, 300])   # actual prices (£k)
y_pred = np.array([210, 240, 330, 170, 380, 310])   # model's predictions

print(mean_squared_error(y_true, y_pred))         # MSE  = 400.0
print(root_mean_squared_error(y_true, y_pred))    # RMSE = 20.0, in £k
print(mean_absolute_error(y_true, y_pred))        # MAE  = 16.67
print(r2_score(y_true, y_pred))                   # R²   = 0.937
```

> **Trap:** old tutorials write `mean_squared_error(..., squared=False)` for RMSE. That option has been removed from recent sklearn; use `root_mean_squared_error` or `np.sqrt(...)`.

## Interview one-liners

- **RMSE vs MAE?** "RMSE punishes big misses more because it squares them; MAE treats every unit of error the same, so it's less affected by outliers."
- **What does R² of 0.94 mean?** "The model explains about 94% of the variation in y compared with always predicting the mean. Not 94% accuracy."
- **Can R² be negative?** "Yes, on new data, if the model is worse than predicting the mean."
- **Why adjusted R²?** "Plain R² always rises when you add features; adjusted R² only rises if the feature helps more than chance."

## Quick reference

| Metric | In words | Example | Better |
|---|---|---|---|
| MSE | average squared miss | 400 | lower |
| RMSE | √MSE; typical miss in y's units | £20k | lower |
| MAE | average size of a miss | £16.7k | lower |
| R² | variation explained vs predicting the mean | 0.937 | higher (max 1) |
| Adjusted R² | R² with a penalty per feature | 0.921 (1 feature) | higher |
| MAPE | average % miss; breaks near y = 0 | 5.6% | lower |
