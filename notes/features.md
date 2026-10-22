# Feature engineering

Making better input columns from the ones you have, using what you know about the problem, then keeping only the columns that actually help.

## What a feature is

- **Feature** (also *input variable*, *predictor*, written $x_j$): one column the model learns from. Size of a house, hour of an accident, a word count.
- **Target** ($y$): the column you want to predict.
- **Feature engineering**: using your knowledge of the problem to *create* new features from existing ones, or to *transform* them into a shape the model can use.
- **Feature selection**: choosing *which* features to keep.

> **Key idea:** a linear model can only add up its inputs, $f(x) = w_1x_1 + w_2x_2 + \dots + b$. If the pattern you need is a product, a curve or a cycle, you have to hand the model a column that already contains it.

## 1. New features from domain knowledge

This is Andrew Ng's house-price example. Each house plot has a **frontage** (width along the road) and a **depth**. You could fit

$$
f(x) = w_1\,\text{frontage} + w_2\,\text{depth} + b
$$

but you *know* that what buyers pay for is land area, and area is width times depth. So you create a third feature:

$$
x_3 = \text{area} = \text{frontage} \times \text{depth}
\qquad
f(x) = w_1x_1 + w_2x_2 + w_3x_3 + b
$$

| frontage (m) | depth (m) | **area (m²)** |
|---|---|---|
| 10 | 20 | 200 |
| 15 | 30 | 450 |
| 8 | 25 | 200 |

Rows 1 and 3 have very different shapes but the same area. Without $x_3$ the model would have to *discover* the multiplication, which a sum of weights cannot do. The model can still decide how much weight to give $x_1$, $x_2$ and $x_3$.

> **Say it like this:** feature engineering is putting what I already know about the problem into a column, so the model doesn't have to learn it from scratch.

## 2. Polynomial features

### The idea

If the data curve, add powers of $x$ as extra columns:

$$
f(x) = w_1 x + w_2 x^2 + w_3 x^3 + b
$$

**This is still linear regression.** "Linear" means linear *in the weights* $w$: each weight multiplies one column and the results are added. The columns themselves can be anything, including $x^2$. So you use the same cost function and the same gradient descent; you just feed it more columns.

![Same 14 points fitted with degree 1, 2, 3 and 12. Degree 1 is a straight line through a U shape (underfit). Degrees 2 and 3 follow the curve. Degree 12 bends to hit individual points (overfit).](figures/features-poly.svg)

In the figure, compare the second panel (degree 2, teal) with the last (degree 12, orange): the orange curve swings wildly between points because it has learned the noise. That is the [bias & variance](#/t/biasvar) trade-off in one picture.

### Worked example: what PolynomialFeatures creates

For one row with two features $a = 2$, $b = 3$:

| degree | columns created | values |
|---|---|---|
| 2 | $a,\ b,\ a^2,\ ab,\ b^2$ | 2, 3, 4, 6, 9 |
| 3 | adds $a^3,\ a^2b,\ ab^2,\ b^3$ | …, 8, 12, 18, 27 |

The $ab$ column is an **interaction** (next section). With `interaction_only=True` you get only the products, no squares: for $a, b, c$ that is $a, b, c, ab, ac, bc$.

### Feature explosion

With $n$ starting features and degree $d$, the number of columns (without the bias column) is

$$
\binom{n + d}{d} - 1
$$

It comes from counting every way to multiply up to $d$ of the $n$ features together (repeats allowed).

![Number of columns after PolynomialFeatures, by degree, for 2, 5, 10 and 20 starting features. The y-axis is logarithmic; 20 features at degree 5 give 53,129 columns.](figures/features-explosion.svg)

Look at the blue line: 20 features become 230 at degree 2 and 53,129 at degree 5. More columns means more weights to learn, more chance to overfit, and slower training.

## 3. Interactions

An **interaction** is a product of two features, $x_1 \times x_2$. It lets the effect of one feature depend on the value of another.

- Example: rain makes accidents more likely, and darkness does too, but rain *at night* may be worse than the two effects added together. `rain × is_night` captures that.
- Area $=$ frontage $\times$ depth is an interaction you chose on purpose.
- Tree models find interactions on their own by splitting on one feature then another. Linear and logistic regression cannot, so you add them by hand.

## 4. Ratios

A **ratio** divides one feature by another, so the model sees a *rate* instead of two raw sizes.

| raw features | ratio | why |
|---|---|---|
| price, area | price per m² | compares houses of different sizes |
| debt, income | debt-to-income | the thing lenders actually reason about |
| clause errors, clause count | error rate per class | fair across big and small classes |

> **Trap:** division by zero. Decide what a zero denominator means (missing? infinite?) and handle it before the model sees `inf`.

## 5. Datetime features

A raw timestamp like `2021-03-05 08:15` is useless to most models as it stands. Pull out the parts that carry the pattern. In pandas, the `.dt` accessor does it once the column is a real datetime.

His US accidents EDA ([py_acc/notebooks/eda.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb)) does exactly this:

```python
df_temp_2.start_time = pd.to_datetime(df_temp_2.start_time)   # text -> real datetime
df_temp_2['start_year'] = df_temp_2.start_time.dt.year
df_temp_2['start_hour'] = df_temp_2.start_time.dt.hour         # rush hours
df_temp_2['start_day'] = df_temp_2.start_time.dt.weekday       # 0-6 mon - sun
df_temp_2['start_month'] = df_temp_2.start_time.dt.month       # seasons
```

And his GB electricity EDA ([py_elec_grid_anomaly/notebooks/eda.ipynb](https://github.com/jag5m33/py_elec_grid_anomaly/blob/main/notebooks/eda.ipynb)) builds a timestamp from the settlement period first:

```python
# Minutes after midnight for each period: period 1 = 0, period 2 = 30, period 3 = 60 ...
time_period = (df_merged.settlement_period - 1)*30
df_merged['timestamp'] = df_merged.settlement_date + pd.to_timedelta(time_period, unit='m')
```

### Worked example

| start_time | hour | weekday | month | is_weekend |
|---|---|---|---|---|
| 2021-03-05 08:15 | 8 | 4 (Fri) | 3 | False |
| 2021-12-25 23:40 | 23 | 5 (Sat) | 12 | True |

### Cyclical features

Hour, weekday and month go round in a circle: 23:00 is one hour from 00:00, but as numbers they are 23 apart. A fix is to put each value on a circle with sine and cosine:

$$
\text{hour\_sin} = \sin\!\left(\frac{2\pi \cdot \text{hour}}{24}\right),\qquad
\text{hour\_cos} = \cos\!\left(\frac{2\pi \cdot \text{hour}}{24}\right)
$$

Worked: hour 8 → $(0.866,\ -0.500)$; hour 23 → $(-0.259,\ 0.966)$; hour 0 → $(0,\ 1)$, which is right next to hour 23.

![Left: hour as a plain number puts 23:00 and 00:00 far apart. Right: the same hours as (sin, cos) points on a circle, where 23:00 sits next to 00:00.](figures/features-cyclic.svg)

In the right panel, follow the circle clockwise from 00:00 at the top: the gap from 23:00 back to 00:00 is the same as any other one-hour step. Tree models usually cope with plain `hour`; linear models and neural nets benefit from sin/cos.

## 6. Binning

**Binning** (or *discretising*) cuts a continuous feature into ranges and replaces each value with its range.

```python
ages = pd.Series([12, 25, 47, 68])
pd.cut(ages, bins=[0, 18, 40, 65, 120],
       labels=['child', 'young adult', 'middle', 'senior'])
# -> child, young adult, middle, senior
```

- **Use it** when the effect really is step-like (an age threshold in a rule), or to make a noisy feature easier to read.
- **Cost:** you throw away information inside each bin (25 and 39 become the same), and the bin edges are your choice. Then the bins need [encoding](#/t/scaling).

## 7. Log transforms

For **right-skewed** features (incomes, clause lengths, house prices), a log squashes the long tail so a few huge values stop dominating. See [Logs & exponents](#/t/logs) for the maths.

$$
x' = \log(1 + x)
$$

The $+1$ (`np.log1p`) keeps zeros legal, since $\log 0$ is undefined. Worked: $0 \to 0$, $9 \to 2.30$, $99 \to 4.61$, $999 \to 6.91$. A thousand-fold range becomes a 0–7 range. The same idea is behind `sublinear_tf` in [TF-IDF](#/t/tfidf).

## 8. Feature selection

More features is not automatically better. Each useless column adds noise, slows training and raises variance. Three common ways to choose:

| method | how it works | example |
|---|---|---|
| **Manual / domain** | you drop columns you know are irrelevant or leak the answer | drop `end_time` when predicting accident severity at the start |
| **L1 (Lasso)** | regularisation pushes useless weights to exactly 0 | `Lasso(alpha=0.1)` |
| **Importance-based** | fit a model, keep the features it uses most | tree `feature_importances_`, `SelectFromModel` |
| Filter / univariate | score each feature against $y$ on its own | `SelectKBest(f_classif, k=20)` |

**Worked L1 example:** 10 features, only the first three actually drive $y$ ($y = 3x_1 - 2x_2 + 1.5x_3 + \text{noise}$). After scaling and `Lasso(alpha=0.1)`, the learned weights are

$$
3.13,\ -1.75,\ 1.49,\ 0.01,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0
$$

The seven noise features are switched off. (See [Regularisation](#/t/regularisation) for why L1 does this and L2 doesn't.)

### The leakage trap

**Data leakage** is when information the model shouldn't have (validation or test rows, or the future) sneaks into training. With feature selection it happens quietly: you pick the features that correlate best with $y$ **using all the rows**, then cross-validate. The validation folds have already voted on which features to keep.

![Left (wrong): features are chosen using every row, so the validation folds have already been seen. Right: selection is fitted on the training folds only, inside each fold, so the score is honest.](figures/features-leakage.svg)

**Proof with pure noise:** 100 rows, 5,000 random features, random 0/1 labels. Nothing can be predicted, so the honest accuracy is about 50%.

| approach | 5-fold CV accuracy |
|---|---|
| select 20 best features on all rows, then CV | **0.85** (fake) |
| `Pipeline([select, model])` inside CV | **0.48** (true) |

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('select', SelectKBest(f_classif, k=20)),     # chosen again inside every fold
    ('model',  LogisticRegression(max_iter=1000)),
])
scores = cross_val_score(pipe, X, y, cv=5)        # selection never sees the validation fold
print(scores.mean())
```

> **Trap:** anything that *learns from the data* (selection, scaling, a vectoriser, imputing with a mean) is part of the model and must be fitted on the training part only. Put it in a `Pipeline` and let cross-validation refit it each fold.

## Code: polynomial features end to end

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# a tiny frame with two inputs
X = pd.DataFrame({'a': [2, 1, 4], 'b': [3, 5, 2]})

poly = PolynomialFeatures(degree=2, include_bias=False)  # no column of 1s (the model has b)
X_poly = poly.fit_transform(X)                           # a, b, a^2, ab, b^2
print(poly.get_feature_names_out())                      # ['a' 'b' 'a^2' 'a b' 'b^2']

# in a real model: expand, THEN scale (x^2 has a much bigger range than x), then fit
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)
# model.fit(X_train, y_train); model.score(X_val, y_val)
```

## Caveats

- **Feature explosion.** Degree and interactions multiply the column count; see the figure above.
- **Overfitting.** More columns means more flexibility. Choose the degree on validation data, and add [regularisation](#/t/regularisation).
- **Leakage.** Features built from the target, from the future, or with statistics computed on all rows. Fit every learned step on train only.
- **Scaling after polynomial features.** If $x$ runs 1–1,000, then $x^2$ runs 1–1,000,000 and $x^3$ to $10^9$. Gradient descent struggles badly unless you [scale](#/t/scaling) after expanding (Ng makes this point when he introduces polynomial regression).
- **Interpretability.** A weight on `a b^2` is hard to explain to anyone.

## Advantages / disadvantages

| advantages | disadvantages |
|---|---|
| Lets simple, fast models fit curves and interactions | Can create thousands of columns quickly |
| Puts domain knowledge directly into the data | Easy to overfit or leak |
| Often a bigger win than switching model | Takes time and judgement; not automatic |
| Selection makes models smaller and easier to read | Selection done wrong gives over-optimistic scores |

## Quick reference

| term | one line |
|---|---|
| Feature | one input column $x_j$ |
| Engineered feature | a new column built from others using domain knowledge (area = frontage × depth) |
| Polynomial features | $x, x^2, x^3, \dots$ as extra columns; still linear in $w$ |
| Number of poly columns | $\binom{n+d}{d} - 1$ |
| Interaction | $x_1 \times x_2$; effect of one depends on the other |
| Ratio | $x_1 / x_2$; turns sizes into rates |
| Datetime parts | `.dt.hour`, `.dt.weekday`, `.dt.month` |
| Cyclical encoding | $\sin(2\pi h/24)$, $\cos(2\pi h/24)$ |
| Binning | `pd.cut` into ranges |
| Log transform | `np.log1p(x)` for right skew |
| L1 selection | Lasso sets useless weights to 0 |
| Leakage in selection | choose features inside each CV fold (Pipeline) |
