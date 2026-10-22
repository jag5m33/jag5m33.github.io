# Scaling & encoding

Putting numeric features on comparable scales and turning categories into numbers, so that models which measure distance or follow gradients treat every feature fairly.

## Terms first

- **Scaling** (feature scaling): changing a numeric feature's range or spread without changing its order. Largest stays largest.
- **Normalisation**: usually means squeezing into a fixed range such as $[0, 1]$ (min–max). Note that people use the word loosely.
- **Standardisation**: shifting to mean 0 and standard deviation 1 (z-score).
- **Encoding**: turning a categorical feature (text labels like "red", "S/M/L") into numbers.
- **Fit / transform**: *fit* learns the numbers the scaler needs (min, max, mean…) from data; *transform* applies them.

## Why scale: the cost-contour picture

Andrew Ng's example: predict house price from $x_1$ = size (300–2,000 ft²) and $x_2$ = bedrooms (0–5).

Because $x_1$ is in the hundreds or thousands, a tiny change in its weight $w_1$ changes the prediction a lot. A change in $w_2$ barely matters. So the cost $J(w_1, w_2)$ is a long, thin valley: very steep in the $w_1$ direction, very flat in the $w_2$ direction.

![Contours of the cost J. Left, unscaled: tall thin ellipses, and gradient descent (wine) zigzags across the valley. Right, scaled: near-circles, and gradient descent (teal) goes almost straight to the minimum (gold star).](figures/scaling-contours.svg)

In the left panel, each step is mostly *across* the steep direction and overshoots, so it bounces from side to side and takes many steps to crawl along the flat direction. The learning rate has to be small to stop it blowing up in the steep direction, which makes the flat direction even slower. In the right panel, the bowl is round, so the gradient points at the minimum and a few steps get there.

> **Key idea:** scaling doesn't change *what* the best model is; it changes how *easy* it is to find, and, for distance-based models, which features get a say.

Ng's rule of thumb: aim for each feature to sit roughly in $-1$ to $1$. Something like $-3$ to $3$ or $-0.3$ to $0.3$ is fine; $-100$ to $100$ or $-0.001$ to $0.001$ is worth rescaling.

## The worked feature

Every scaler below uses the same five values, one of them an outlier:

$$
x = 20,\ 30,\ 40,\ 50,\ 160
$$

Summary numbers (computed): min $= 20$, max $= 160$, mean $\mu = 60$, standard deviation $\sigma = 50.99$ (population version, which sklearn uses), median $= 40$, $Q_1 = 30$, $Q_3 = 50$, IQR $= 20$.

## 1. Min–max scaling

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**Where it comes from:** subtract the min so the smallest value becomes 0; divide by the range so the largest becomes 1. Output is always in $[0, 1]$ (on the training data).

**Worked:** range $= 160 - 20 = 140$.

| $x$ | $x - 20$ | $\div 140$ |
|---|---|---|
| 20 | 0 | **0.000** |
| 30 | 10 | **0.071** |
| 40 | 20 | **0.143** |
| 50 | 30 | **0.214** |
| 160 | 140 | **1.000** |

> **Trap:** one outlier sets the max, so the four normal values are crushed into $0$ to $0.21$. And a new value outside the training range escapes $[0,1]$: train on $20, 30, 40, 50$, then 60 becomes $1.33$.

## 2. Z-score standardisation

$$
z = \frac{x - \mu}{\sigma}
$$

**Where it comes from:** subtract the mean so the column is centred at 0; divide by the standard deviation so one unit means "one typical deviation". This is the same z-score as in [Descriptive statistics](#/t/descriptive).

**Worked:** $\mu = 60$, $\sigma = 50.99$.

| $x$ | $x - 60$ | $\div 50.99$ |
|---|---|---|
| 20 | −40 | **−0.784** |
| 30 | −30 | **−0.588** |
| 40 | −20 | **−0.392** |
| 50 | −10 | **−0.196** |
| 160 | 100 | **1.961** |

Result: mean 0, SD 1. No fixed range. The outlier still pulls $\mu$ and $\sigma$, so the normal values are bunched below zero.

> **Say it like this:** a z-score says how many standard deviations a value is from the mean. Standardising is the default for linear models, logistic regression, SVMs, PCA and neural nets.

## 3. Mean normalisation

Ng's variant: centre on the mean, divide by the range.

$$
x' = \frac{x - \mu}{x_{\max} - x_{\min}}
$$

**Worked:** $(20 - 60)/140 = -0.286$, …, $(160 - 60)/140 = 0.714$. Values land roughly in $[-1, 1]$ with mean 0. sklearn has no class for it; `StandardScaler` is used instead in practice.

## 4. Max-abs scaling

$$
x' = \frac{x}{\max \lvert x \rvert}
$$

Divides by the largest absolute value, so output is in $[-1, 1]$. It **doesn't shift** the data, so zeros stay zeros. That matters for **sparse matrices** (like [TF-IDF](#/t/tfidf) output): subtracting a mean would turn millions of zeros into non-zeros.

**Worked:** $20/160 = 0.125$, $30/160 = 0.188$, $40/160 = 0.25$, $50/160 = 0.313$, $160/160 = 1$.

## 5. Robust scaling

$$
x' = \frac{x - \text{median}}{\text{IQR}} = \frac{x - \text{median}}{Q_3 - Q_1}
$$

Uses the median and the [interquartile range](#/t/quartiles), which ignore the extremes, so one outlier can't stretch the scale.

**Worked:** median $= 40$, IQR $= 50 - 30 = 20$.

| $x$ | $x - 40$ | $\div 20$ |
|---|---|---|
| 20 | −20 | **−1.0** |
| 30 | −10 | **−0.5** |
| 40 | 0 | **0.0** |
| 50 | 10 | **0.5** |
| 160 | 120 | **6.0** |

The normal values keep a sensible spread ($-1$ to $0.5$) and the outlier is clearly flagged as far away.

## Comparing scalers on a skewed feature

![The same right-skewed feature (400 values plus two outliers) after each scaler. The shape never changes; only the numbers on the axis do. Min-max and max-abs crush the bulk near 0; robust keeps the bulk spread around 0.](figures/scaling-skewed.svg)

Look at the dashed median line in each panel: the histograms are identical shapes. **Scaling is linear, so it never fixes skew.** If the shape is the problem, use a [log transform](#/t/logs) first, then scale.

| scaler | formula | output range | outliers | use when |
|---|---|---|---|---|
| MinMaxScaler | $(x-\min)/(\max-\min)$ | $[0,1]$ | very sensitive | bounded data, images, neural-net inputs |
| StandardScaler | $(x-\mu)/\sigma$ | unbounded, mean 0, SD 1 | sensitive | default for most models |
| Mean normalisation | $(x-\mu)/(\max-\min)$ | about $[-1,1]$ | very sensitive | Ng's course examples |
| MaxAbsScaler | $x/\max\lvert x\rvert$ | $[-1,1]$ | very sensitive | sparse data (keeps zeros) |
| RobustScaler | $(x-\text{median})/\text{IQR}$ | unbounded | resistant | data with outliers |

## Fit on train only

The scaler *learns* numbers from the data (min, max, mean, SD, median, IQR). If it learns them from all rows, information from the validation and test sets leaks into training. This is the same rule as in [Feature engineering](#/t/features) and [TF-IDF](#/t/tfidf):

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # learn mean and SD from TRAIN, then scale train
X_val_s   = scaler.transform(X_val)         # reuse train's mean and SD
X_test_s  = scaler.transform(X_test)        # never fit on test
```

## Which models need scaling

| model | needs scaling? | why |
|---|---|---|
| Linear / logistic regression trained by **gradient descent** | yes | contour picture above |
| **Regularised** models (Ridge, Lasso, logistic with C) | yes | the penalty $\lambda\sum w_j^2$ punishes weights on small-unit features unfairly |
| **KNN**, **K-means**, SVM | yes | distances are dominated by the feature with the biggest units |
| **PCA** | yes | finds directions of biggest variance; big-unit features win |
| **Neural networks** | yes | gradient descent, plus activations saturate on large inputs |
| Decision trees, random forests, **XGBoost** / boosting | no | splits are thresholds ("is $x > 50$?"); rescaling moves the threshold, not the split |
| Naive Bayes | usually no | each feature is modelled separately |

## Encoding categorical features

Models need numbers. A **categorical** feature has labels, not quantities. Two kinds (see [Variable types](#/t/vartypes)):

- **Nominal**: no order (colour, city, clause type).
- **Ordinal**: a real order (S < M < L; low < medium < high severity).

### One-hot encoding

Make one 0/1 column per category. Exactly one column is 1 in each row.

![One-hot encoding of colour into three 0/1 columns. With drop='first' the colour_blue column (dashed) is removed; a row of all zeros then means blue.](figures/scaling-onehot.svg)

In the figure, each row has a single shaded 1. No category is "bigger" than another, which is exactly right for nominal data.

### The dummy variable trap

The three one-hot columns always add up to 1: $\text{blue} + \text{green} + \text{red} = 1$. A linear model also has an intercept $b$, which is a hidden column of 1s. So one column is an exact copy of a combination of the others: **perfect multicollinearity**.

**Checked numerically:** stack the intercept column with the three one-hot columns (4 columns) and the matrix rank is only 3. One column carries no new information, so the weights have no unique solution (infinitely many $w, b$ give the same predictions), and the coefficients can't be interpreted.

**Fix for linear models:** `OneHotEncoder(drop='first')`. Keep $k-1$ columns; the dropped category becomes the **baseline**, and each remaining weight means "difference from blue".

| colour | colour_green | colour_red |
|---|---|---|
| red | 0 | 1 |
| green | 1 | 0 |
| blue | 0 | 0 |

> **Trap:** with regularisation (which sklearn's logistic regression has by default) the trap is mostly harmless, and tree models don't care. It bites hardest in plain, unregularised linear regression and in statsmodels where you read the coefficients.

### Ordinal encoding

For ordered categories, map to integers **in the right order**. Tell sklearn the order; otherwise it sorts alphabetically.

| size | `OrdinalEncoder(categories=[['S','M','L']])` | `OrdinalEncoder()` (alphabetical) |
|---|---|---|
| S | 0 | 2 |
| L | 2 | 0 |
| M | 1 | 1 |

The default gets L < M < S, which is wrong. His accidents decision-tree notebook ([py_acc/notebooks/dcdm.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/dcdm.ipynb)) imports `OrdinalEncoder` and converts boolean columns to 0/1, which is a safe ordinal encoding.

### Label encoding pitfalls

`LabelEncoder` turns `blue, green, red` into `0, 1, 2`. It is meant for the **target** $y$, not for features. Used on a nominal feature it invents an order (red > green > blue, and red − green = green − blue) that a linear model or KNN will take literally. Trees cope better but can still need more splits.

### High cardinality

**Cardinality** = number of distinct categories. One-hot on a column with 10,000 postcodes makes 10,000 columns, mostly zeros. Options: group rare categories into "other" (`OneHotEncoder(min_frequency=...)`), use frequency or target encoding (target encoding must be fitted inside CV folds or it leaks $y$), or use a tree model. Unseen categories at test time crash the default encoder; `handle_unknown='ignore'` turns them into an all-zero row.

## Code: ColumnTransformer + Pipeline

One object that scales numbers, encodes categories and fits the model, all learned on train only:

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

numeric  = ['temperature', 'visibility']   # roughly symmetric -> StandardScaler
skewed   = ['distance']                    # outliers -> RobustScaler
nominal  = ['weather', 'state']            # no order -> one-hot
ordinal  = ['size']                        # S < M < L

preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('rob', RobustScaler(),   skewed),
    ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal),  # avoid dummy trap; survive new categories
    ('ord', OrdinalEncoder(categories=[['S', 'M', 'L']]), ordinal),         # give the order explicitly
])

model = Pipeline([
    ('prep',  preprocess),
    ('clf',   LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)          # every scaler/encoder learns from train only
print(model.score(X_val, y_val))     # val is transformed with train's numbers
```

Note: older sklearn versions (before 1.0) raise an error when `drop='first'` and `handle_unknown='ignore'` are combined; current versions allow it (an unseen category becomes all zeros, same as the baseline).

## Caveats

- **Leakage:** fit scalers and encoders on train only; use a Pipeline inside cross-validation.
- **Outliers:** min–max and standard scaling are dragged by them; use robust scaling or deal with the outliers first.
- **Skew:** no scaler changes the shape. Log-transform first if needed.
- **Sparse data:** don't centre it (StandardScaler `with_mean=False`, or MaxAbsScaler).
- **Interpretation:** after standardising, a weight means "change in prediction per one SD", not per original unit.

## Quick reference

| term | one line |
|---|---|
| Min–max | $(x-\min)/(\max-\min)$, range $[0,1]$ |
| Z-score | $(x-\mu)/\sigma$, mean 0, SD 1 |
| Mean normalisation | $(x-\mu)/(\max-\min)$ |
| Max-abs | $x/\max\lvert x\rvert$, keeps zeros |
| Robust | $(x-\text{median})/\text{IQR}$, resists outliers |
| Needs scaling | gradient descent, regularised, KNN/K-means, SVM, PCA, neural nets |
| Doesn't | trees, forests, boosting |
| One-hot | one 0/1 column per category |
| Dummy trap | $k$ one-hot columns + intercept are collinear; use `drop='first'` |
| Ordinal | integers in a stated order |
| Label encoding | for the target only |
| ColumnTransformer | different preprocessing per column group |
