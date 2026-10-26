# Feature engineering

Making better input columns from the ones you have, using what you know about the problem, then keeping only the columns that actually help.

## What a feature is

- **Feature** (input, predictor, $x_j$) = one column the model learns from, e.g. hour of an accident or a word count.
- **Target** ($y$) = the column you want to predict.
- **Feature engineering** = using your knowledge of the problem to *create* or *reshape* features.
- **Feature selection** = choosing *which* features to keep.

> **Key idea:** a linear model only adds up its inputs: $f(x) = w_1x_1 + w_2x_2 + b$. If the pattern is a product, a curve or a cycle, you have to hand the model a column that already contains it.

## 1. New features from domain knowledge

Andrew Ng's house example: each plot has a **frontage** (width along the road) and a **depth**. But buyers pay for land **area**, so make a new column:

$$
\text{area} = \text{frontage} \times \text{depth}
$$

| frontage (m) | depth (m) | **area (m²)** |
|---|---|---|
| 10 | 20 | 200 |
| 15 | 30 | 450 |
| 8 | 25 | 200 |

Rows 1 and 3 have different shapes but the same area. A sum of weights can't discover a multiplication on its own, so you give it the answer as a column.

> **Say it like this:** "Feature engineering is putting what I already know about the problem into a column, so the model doesn't have to learn it from scratch."

## 2. Polynomial features and interactions

If the data curve, add powers of $x$ as extra columns:

$$
f(x) = w_1 x + w_2 x^2 + b
$$

This is **still linear regression** — "linear" means each weight multiplies one column and the results are added. The column can be $x^2$. Same cost function, same gradient descent, just more columns.

![Same 14 points fitted with degree 1, 2, 3 and 12. Degree 1 is a straight line through a U shape (underfit). Degrees 2 and 3 follow the curve. Degree 12 bends to hit individual points (overfit).](figures/features-poly.svg)

In the figure, degree 2 (teal) follows the curve nicely, while degree 12 (orange) swings wildly to hit every point — it has learned the noise. That is [bias & variance](#/t/biasvar) in one picture.

**What it creates.** For one row with $a = 2$, $b = 3$, degree 2 gives columns $a, b, a^2, ab, b^2$ = **2, 3, 4, 6, 9**.

- The $ab$ column is an **interaction** = a product of two features. It lets the effect of one depend on the other: rain at night may be worse than rain plus night added together, so `rain × is_night` captures it.
- Tree models find interactions by themselves; linear and logistic regression need them added by hand.

![Number of columns after PolynomialFeatures, by degree, for 2, 5, 10 and 20 starting features. The y-axis is logarithmic; 20 features at degree 5 give 53,129 columns.](figures/features-explosion.svg)

In the figure, follow the blue line: 20 features become 230 columns at degree 2 and 53,129 at degree 5. More columns = slower training and more overfitting. This is **feature explosion**.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),  # a, b, a^2, ab, b^2
    StandardScaler(),        # scale AFTER expanding: x^2 has a much bigger range
    LinearRegression(),
)
# model.fit(X_train, y_train); model.score(X_val, y_val)
```

## 3. Ratios

A **ratio** divides one feature by another, turning two sizes into a rate: price per m², debt-to-income, errors per class.

> **Trap:** division by zero. Decide what a zero bottom number means before the model sees `inf`.

## 4. Date and time features

A raw timestamp is useless to most models. Pull out the parts that carry the pattern with pandas' `.dt`. From his accidents EDA ([eda.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/eda.ipynb)):

```python
df_temp_2.start_time = pd.to_datetime(df_temp_2.start_time)  # text -> date
df_temp_2['start_year'] = df_temp_2.start_time.dt.year
df_temp_2['start_hour'] = df_temp_2.start_time.dt.hour         # rush hours
df_temp_2['start_day'] = df_temp_2.start_time.dt.weekday       # 0-6 mon - sun
df_temp_2['start_month'] = df_temp_2.start_time.dt.month       # seasons
```

| start_time | hour | weekday | month |
|---|---|---|---|
| 2021-03-05 08:15 | 8 | 4 (Fri) | 3 |
| 2021-12-25 23:40 | 23 | 5 (Sat) | 12 |

**Cyclical features.** Hours go round in a circle: 23:00 is one hour from 00:00, but as numbers they are 23 apart. The fix is to place each hour on a circle using two columns, sin and cos of $2\pi \times \text{hour} / 24$. Hour 23 → (−0.259, 0.966) and hour 0 → (0, 1): close together, as they should be.

![Left: hour as a plain number puts 23:00 and 00:00 far apart. Right: the same hours as (sin, cos) points on a circle, where 23:00 sits next to 00:00.](figures/features-cyclic.svg)

In the right panel, the step from 23:00 back to 00:00 is the same as any other one-hour step. Tree models cope fine with plain `hour`; linear models benefit from sin/cos.

## 5. Binning and log transforms

- **Binning** = cutting a number into ranges, e.g. `pd.cut(ages, bins=[0, 18, 40, 65, 120])`. Useful when the effect really is step-like. Cost: 25 and 39 become the same. The bins then need [encoding](#/t/scaling).
- **Log transform** = $\log(1 + x)$ (`np.log1p`) for **right-skewed** columns like incomes or clause lengths. It squashes the long tail: $0 \to 0$, $9 \to 2.30$, $99 \to 4.61$, $999 \to 6.91$. The $+1$ keeps zeros allowed. See [Logs & exponents](#/t/logs).

## 6. Feature selection

More features is not automatically better: useless columns add noise and overfitting.

| Method | How | Example |
|---|---|---|
| **Domain knowledge** | drop columns that are irrelevant or leak the answer | drop `end_time` when predicting severity at the start |
| **L1 (Lasso)** | the penalty pushes useless weights to exactly 0 | `Lasso(alpha=0.1)` |
| **Importance** | keep what a fitted model uses most | tree `feature_importances_` |

**Lasso example:** 10 features, only the first three really drive $y$. After scaling and `Lasso(alpha=0.1)` the weights are 3.13, −1.75, 1.49, 0.01, then 0 for the other six — the noise features are switched off. See [Regularisation](#/t/regularisation).

### The leakage trap

**Data leakage** = information the model shouldn't have (test rows, or the future) sneaking into training. With selection it happens quietly: you pick the best features **using all rows**, then cross-validate — so the validation rows have already voted.

![Left (wrong): features are chosen using every row, so the validation folds have already been seen. Right: selection is fitted on the training folds only, inside each fold, so the score is honest.](figures/features-leakage.svg)

**Proof with pure noise:** 100 rows, 5,000 random features, random 0/1 labels. Nothing can be predicted, so the honest accuracy is about 50%.

| Approach | 5-fold CV accuracy |
|---|---|
| pick 20 best features on all rows, then CV | **0.85** (fake) |
| selection inside a `Pipeline` | **0.48** (true) |

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("select", SelectKBest(f_classif, k=20)),   # redone inside every fold
    ("model", LogisticRegression(max_iter=1000)),
])
print(cross_val_score(pipe, X, y, cv=5).mean())              # honest score
```

> **Trap:** anything that *learns from data* (selection, scaling, filling with a mean) must be fitted on training rows only. Put it in a `Pipeline`.

## Caveats

- **Feature explosion** — polynomials and interactions multiply columns fast.
- **Overfitting** — pick the degree on validation data; add [regularisation](#/t/regularisation).
- **Scale after polynomial features** — if $x$ runs to 1,000, $x^2$ runs to 1,000,000 and gradient descent struggles ([scaling](#/t/scaling)).
- **Harder to explain** — a weight on `a b^2` means little to anyone.

| Advantages | Disadvantages |
|---|---|
| Lets simple models fit curves and interactions | Can create thousands of columns |
| Puts domain knowledge into the data | Easy to overfit or leak |
| Often a bigger win than switching model | Takes time and judgement |

## Quick reference

| Term | One line |
|---|---|
| Feature | one input column $x_j$ |
| Engineered feature | new column from domain knowledge (area = frontage × depth) |
| Polynomial features | $x, x^2, \dots$ as extra columns; still linear regression |
| Interaction | $x_1 \times x_2$; effect of one depends on the other |
| Ratio | $x_1 / x_2$; sizes become rates |
| Date parts | `.dt.hour`, `.dt.weekday`, `.dt.month` |
| Cyclical encoding | sin and cos of $2\pi \times$ hour / 24 |
| Binning | `pd.cut` into ranges |
| Log transform | `np.log1p(x)` for right skew |
| L1 selection | Lasso sets useless weights to 0 |
| Leakage in selection | select inside each CV fold (Pipeline) |
