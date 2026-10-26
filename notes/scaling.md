# Scaling & encoding

Putting numeric features on similar scales and turning categories into numbers, so that models which measure distance or follow gradients treat every feature fairly.

## Terms first

- **Scaling** = changing a numeric feature's range without changing its order. The largest value stays the largest.
- **Normalisation** = usually squeezing into $[0, 1]$ (min–max). People use the word loosely.
- **Standardisation** = shifting to mean 0 and standard deviation 1 (z-score).
- **Encoding** = turning a category ("red", "S/M/L") into numbers.
- **Fit / transform** = *fit* learns the numbers the scaler needs (min, max, mean…); *transform* applies them.

## Why scale

Andrew Ng's example: predict house price from $x_1$ = size (300–2,000 ft²) and $x_2$ = bedrooms (0–5). Because size is in the thousands, a tiny change in its weight moves the prediction a lot, while the bedrooms weight barely matters. The cost surface becomes a long, thin valley.

![Contours of the cost J. Left, unscaled: tall thin ellipses, and gradient descent (wine) zigzags across the valley. Right, scaled: near-circles, and gradient descent (teal) goes almost straight to the minimum (gold star).](figures/scaling-contours.svg)

In the left panel, gradient descent bounces from side to side across the steep valley and takes many steps. In the right panel the bowl is round, so it heads almost straight for the minimum (gold star).

> **Key idea:** scaling doesn't change *what* the best model is; it changes how *easy* it is to find — and, for distance-based models, which features get a say.

Ng's rule of thumb: aim for each feature to sit roughly between −1 and 1.

Both examples below use the same five values, with 160 as an outlier: **20, 30, 40, 50, 160**.

## Min–max scaling

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

- $x_{\min}$, $x_{\max}$ = the smallest and largest values in the training data.
- Subtracting the min makes the smallest value 0; dividing by the range makes the largest 1.

**Worked:** min 20, max 160, so the range is 140.

| $x$ | $x - 20$ | $\div 140$ |
|---|---|---|
| 20 | 0 | **0.000** |
| 30 | 10 | **0.071** |
| 40 | 20 | **0.143** |
| 50 | 30 | **0.214** |
| 160 | 140 | **1.000** |

> **Trap:** one outlier sets the max, so the four normal values are squashed into 0 to 0.21. Also, new data can escape $[0, 1]$: train on 20, 30, 40, 50 and a new 60 becomes 1.33.

## Z-score standardisation

$$
z = \frac{x - \mu}{\sigma}
$$

- $\mu$ = the mean; $\sigma$ = the standard deviation (typical distance from the mean).
- $z$ = how many standard deviations a value sits above (+) or below (−) the mean. Same z-score as in [Descriptive statistics](#/t/descriptive).

**Worked:** $\mu = 60$, $\sigma = 50.99$.

| $x$ | $x - 60$ | $\div 50.99$ |
|---|---|---|
| 20 | −40 | **−0.784** |
| 30 | −30 | **−0.588** |
| 40 | −20 | **−0.392** |
| 50 | −10 | **−0.196** |
| 160 | 100 | **1.961** |

The result has mean 0 and SD 1, with no fixed range. The outlier still pulls the mean up, so the normal values bunch below zero.

> **Say it like this:** "A z-score says how many standard deviations a value is from the mean. Standardising is my default for linear and logistic regression, KNN, SVMs, PCA and neural nets."

**Robust scaling** (`RobustScaler`) is the outlier-proof option: it uses the median and the [IQR](#/t/quartiles) instead of mean and SD, so one extreme value can't stretch the scale.

![The same right-skewed feature (400 values plus two outliers) after min-max, standard and robust scaling. The shape never changes; only the numbers on the axis do. Min-max crushes the bulk near 0; robust keeps the bulk spread around 0.](figures/scaling-skewed.svg)

In the figure, every histogram has the same shape — only the axis numbers change. **Scaling never fixes skew.** If the shape is the problem, use a [log transform](#/t/logs) first.

## Fit on train only

The scaler *learns* numbers (mean, SD, min, max). Learn them from training rows only, or test information leaks into training.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # learn mean and SD from TRAIN
X_val_s   = scaler.transform(X_val)         # reuse train's mean and SD
X_test_s  = scaler.transform(X_test)        # never fit on test
```

## Which models need scaling

| Model | Scale? | Why |
|---|---|---|
| Linear / logistic regression (gradient descent) | yes | the valley picture above |
| Regularised models (Ridge, Lasso) | yes | the penalty treats all weights alike, so units matter |
| KNN, K-means, SVM, PCA | yes | big-unit features dominate distances |
| Neural networks | yes | gradient descent |
| Decision trees, random forests, XGBoost | no | splits are thresholds ("is $x > 50$?"); scaling just moves the threshold |

## Encoding categories

Models need numbers. Two kinds of category (see [Variable types](#/t/vartypes)):
- **Nominal** = no order (colour, city).
- **Ordinal** = a real order (S < M < L).

### One-hot encoding

Make one 0/1 column per category. In each row exactly one column is 1.

![One-hot encoding of colour into three 0/1 columns. With drop='first' the colour_blue column (dashed) is removed; a row of all zeros then means blue.](figures/scaling-onehot.svg)

In the figure, each row has a single shaded 1. No colour is "bigger" than another — exactly right for nominal data.

### The dummy variable trap

- With three colours, the three one-hot columns **always add up to 1** in every row.
- A linear model already has an intercept $b$, which acts like a hidden column of 1s.
- So one column can be worked out from the others: if green = 0 and red = 0, it **must** be blue. That column adds no new information.
- This is **perfect multicollinearity** (one column is an exact copy of a mix of others). The model can't decide how to split the effect between the copies, so the weights can't be trusted or read.

**Fix:** drop one column (`drop='first'`). The dropped category becomes the **baseline**, and each remaining weight means "difference from blue".

| colour | colour_green | colour_red |
|---|---|---|
| red | 0 | 1 |
| green | 1 | 0 |
| blue | 0 | 0 |

> **Trap:** the trap matters most for plain linear regression where you read the weights. Tree models don't care.

### Ordinal encoding

For ordered categories, map to integers **in the right order** — and tell sklearn the order, or it sorts alphabetically (L < M < S, which is wrong).

### Label encoding pitfall

`LabelEncoder` turns `blue, green, red` into `0, 1, 2`. It is meant for the **target** $y$. On a nominal feature it invents an order (red > green > blue) that a linear model or KNN takes literally.

### Many categories

**Cardinality** = number of distinct categories. One-hot on 10,000 postcodes makes 10,000 columns. Options: group rare ones into "other", or use a tree model. `handle_unknown='ignore'` stops the encoder crashing on a category it hasn't seen.

## Code: everything in one Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

preprocess = ColumnTransformer([
    # numbers: z-score
    ("num", StandardScaler(), ["temperature", "visibility"]),
    # no order: one-hot, drop one column to avoid the dummy trap
    ("nom", OneHotEncoder(drop="first", handle_unknown="ignore"), ["weather"]),
    # ordered: give the order S < M < L
    ("ord", OrdinalEncoder(categories=[["S", "M", "L"]]), ["size"]),
])
model = Pipeline([("prep", preprocess),
                  ("clf", LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)          # every scaler/encoder learns from train only
print(model.score(X_val, y_val))
```

His accidents notebook ([dcdm.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/dcdm.ipynb)) imports `OrdinalEncoder` and turns True/False columns into 0/1.

## Caveats

- **Leakage:** fit scalers and encoders on train only; use a Pipeline.
- **Outliers:** min–max and z-score are dragged by them; use robust scaling.
- **Skew:** no scaler changes the shape; log-transform first.
- **Reading weights:** after standardising, a weight means "change per one SD", not per original unit.

## Quick reference

| Term | One line |
|---|---|
| Min–max | $(x - \min)/(\max - \min)$, range 0 to 1 |
| Z-score | $(x - \mu)/\sigma$, mean 0, SD 1 |
| Robust | uses median and IQR; resists outliers |
| Needs scaling | gradient descent, regularised, KNN, SVM, PCA, neural nets |
| Doesn't | trees, forests, boosting |
| One-hot | one 0/1 column per category |
| Dummy trap | all one-hot columns + intercept repeat information; drop one |
| Ordinal | integers in a stated order |
| Label encoding | for the target only |
