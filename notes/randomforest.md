# Random forest

A random forest is a crowd of decision trees, each grown on a bootstrap sample of the rows and allowed to look at only a random handful of features at each split, so the trees disagree in useful ways and their average is more accurate than any one of them.

## Before you start

This chapter builds on [Decision trees](#/t/trees) (information gain, root and decision nodes) and [Bagging & bootstrap sampling](#/t/bagging) (bootstrap samples, out-of-bag rows, and the variance formula $\rho\sigma^2 + (1-\rho)\sigma^2/B$). Same 10 animals again.

## The idea in one breath

A random forest uses **the same tree structure as a decision tree**. Every tree in it is an ordinary tree: root node, decision nodes, leaves, splits chosen by information gain (or Gini). What makes it a *forest* is that it adds **two kinds of randomness**:

1. **Row subsampling with replacement**: each tree is trained on its own **bootstrap sample** of the rows (see [Bagging](#/t/bagging)). This is the "bagging" part.
2. **Random feature choice at the splits**: when a node looks for its best split, it may only choose among a **random subset of the features** (`max_features` of them), not all of them.

Then the trees are combined: majority vote (or averaged probability) for classification, average for regression.

> **Key idea (to be precise about point 2):** the random feature subset is drawn **again at every split**, at every node of every tree, not just once for the root. The root is simply the first place you see it happen, so it's the easiest place to picture. With `max_depth=1` trees (stumps) the root is the only split, so there it really is "only the root". In a full-depth tree, each decision node gets its own fresh draw.

![One tree in a random forest, as an illustration. The root drew only "whiskers", so it had to split on whiskers. Each child then drew its own feature (ear shape on one side, face shape on the other).](figures/randomforest-everysplit.svg)

In the figure, each gold box shows which feature(s) that node was allowed to consider. They differ from node to node, because the draw is repeated at every split.

> **Trap:** "Choose a random set of features once per tree, then grow the tree normally" is a real method, but it's a different one: the **random subspace method**. sklearn's `BaggingClassifier(max_features=...)` and xgboost's `colsample_bytree` work that way (per tree). Random forest in sklearn draws per split, which is what xgboost calls `colsample_bynode`.

## Why randomise the features at all?

### The problem with bagging alone

In bagging, every split may look at every feature. If one feature is clearly the strongest, it wins the root in most bootstrap samples, so most trees start with the same question and end up with similar shapes. Similar trees make similar mistakes: a high correlation $\rho$.

### Seen on the cat table

On the full table, ear shape wins the root (gain 0.278 against 0.125 and 0.035, from the trees chapter). I grew 100 trees each way (computed in Python, `random_state=0`, entropy criterion):

- **Bagged trees**: bootstrap sample, but all 3 features allowed at every split.
- **Random forest**: bootstrap sample, and `max_features=1`, so each split sees **one** randomly chosen feature.

![Root feature of 100 trees. Bagged trees chose ear shape 59 times; the random forest spread its roots across all three features.](figures/randomforest-roots.svg)

| Root feature | Bagged trees | Random forest (`max_features=1`) |
|---|---|---|
| Ear shape | **59** | 30 |
| Face shape | 16 | 44 |
| Whiskers | 25 | 26 |

In the figure, the gold bars lean heavily towards ear shape. The bootstrap alone shuffles the rows enough to unseat it in 41 trees, but it is still the usual root. With one random feature per split, each feature becomes the root roughly a third of the time (the dotted line). Face shape, the weakest root split on the full data (gain 0.035), now starts 44 trees. Those trees are forced to grow a different shape.

Why not exactly 33/33/33? Chance: 100 trees is a small sample. Grow 5,000 trees with the same settings and the roots split 33.8% / 33.6% / 32.4% (ear / face / whiskers).

### What that does to $\rho$

Recall from the bagging chapter: the variance of the average of $B$ trees is

$$
\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

The first term, $\rho\sigma^2$, is a floor that more trees can't remove. Random features lower $\rho$, which lowers the floor. Each individual tree gets a bit worse (it sometimes has to use a weaker feature, so $\sigma^2$ and its bias rise slightly), but the average improves because the trees' errors overlap less.

I measured this on a 20-feature synthetic dataset (1,000 rows, 6 informative features, 10% flipped labels), using the average pairwise correlation between trees' predicted probabilities on the test rows:

![Left: average correlation between trees falls from 0.50 (bagging) to 0.36 (random forest). Right: test accuracy for one tree, bagging and random forest.](figures/randomforest-rho.svg)

| | One tree | Bagging (200 trees) | Random forest (200 trees) |
|---|---|---|---|
| Average correlation between trees, $\rho$ | – | 0.50 | **0.36** |
| Test accuracy (300 rows) | 0.777 | 0.847 | **0.850** |
| OOB score | – | 0.856 | 0.854 |

In the figure, the left panel is the whole point of random forests: the trees are less alike. On the right, both ensembles beat the single tree by about 7 points, and the forest edges out bagging here. (On the tiny cat table, depth-1 stumps show the same effect, with $\rho$ falling from 0.43 to 0.17. Full-depth trees on just 3 binary features end up using all three features anyway, so there the gap disappears.)

> **Say it like this:** "Bagging makes the trees different by giving them different rows. Random forest also gives each split a different set of features to choose from, so even a dominant feature can't make every tree look the same. Less correlated trees means the average has lower variance."

## `max_features`: how many features per split?

Let $p$ be the number of features.

| Setting | Features tried at each split | Effect |
|---|---|---|
| `max_features=None` or `1.0` | all $p$ | No feature randomness: this is plain bagged trees |
| `"sqrt"` | $\sqrt{p}$ (rounded down) | Default for `RandomForestClassifier` |
| `"log2"` | $\log_2 p$ | Even fewer, more randomness |
| an integer, e.g. `3` | exactly that many | |
| a float, e.g. `0.33` | that fraction of $p$ | $p/3$ is the classic suggestion for regression |
| `1` | one feature, chosen at random | Most random; each tree is weak, $\rho$ is very low |

- **Classification default**: `RandomForestClassifier(max_features="sqrt")`. With $p = 3$ cat features, $\sqrt{3} = 1.73$, rounded down to **1**, so the default on the cat table is exactly the `max_features=1` forest above.
- **Regression default**: `RandomForestRegressor(max_features=1.0)`, i.e. **all features** (checked in sklearn 1.9; the default has been 1.0 since version 1.1). So out of the box, sklearn's regression forest is really **bagged regression trees** with no feature randomness. Set `max_features` yourself (e.g. `0.33` or `"sqrt"`) if you want a true random forest for regression.
- **Smaller** `max_features` → less correlated trees, but each tree is weaker. **Larger** → stronger trees, more alike. It's the main dial to tune.

## Making a prediction

1. Send the new example down **every** tree. Each tree walks one path to a leaf, exactly as a single tree does.
2. **Classification**: the original random forest takes a **majority vote** of the trees' classes. sklearn instead **averages the trees' predicted probabilities** and picks the class with the highest average (a "soft vote"). The two usually agree.
3. **Regression**: average the trees' numbers.

On the cat table, the 100-tree forest gives the new animal from the bagging chapter (floppy ears, round face, whiskers) an average P(cat) of **0.62**, so it predicts **cat**. That matches the only training animal with floppy ears and whiskers (animal 2, a cat).

### OOB score

Each tree has its out-of-bag rows (about 37% of the training rows), so a random forest gets the same free validation estimate as bagging: `oob_score=True`, then read `oob_score_`. On the two-moons data below the forest's OOB score was 0.882, against 0.917 on the held-out test set. On the 20-feature data it was 0.854 against 0.850.

## Code: the cat table

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],
              [1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])   # pointy, round, whiskers
y = np.array([1,1,0,0,1,1,0,1,0,0])                        # 1 = cat

rf = RandomForestClassifier(
    n_estimators=100,      # 100 trees
    max_features=1,        # ONE random feature per split (same as "sqrt" here)
    criterion="entropy",   # information gain, as in the trees chapter
    bootstrap=True,        # each tree gets a bootstrap sample of the rows (default)
    oob_score=True,        # score rows with the trees that didn't see them
    random_state=0,
)
rf.fit(X, y)

roots = [t.tree_.feature[0] for t in rf.estimators_]   # root feature of each tree
print("roots (ear, face, whiskers):", np.bincount(roots, minlength=3))
print("P(cat) for floppy/round/whiskers:", rf.predict_proba([[0, 1, 1]])[0, 1])
print("OOB score:", rf.oob_score_)
print("importances:", rf.feature_importances_.round(3))
```

```text
roots (ear, face, whiskers): [30 44 26]
P(cat) for floppy/round/whiskers: 0.62
OOB score: 0.7
importances: [0.538 0.217 0.245]
```

- `rf.estimators_` is the list of the 100 fitted `DecisionTreeClassifier`s. Each one is an ordinary tree you could print with `export_text`.
- The OOB score of 0.7 is 7 of 10 animals. With 10 rows it is very noisy; don't read much into it.
- Even though ear shape is the root of only 30 trees, it still gets the highest importance (0.538): it removes the most impurity overall, wherever it appears in the trees.

## Code: one tree vs bagging vs random forest

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=0)   # 2 noisy features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "one tree": DecisionTreeClassifier(random_state=0),
    "bagging":  BaggingClassifier(DecisionTreeClassifier(), n_estimators=200,
                                  random_state=0),
    "forest":   RandomForestClassifier(n_estimators=200, oob_score=True,
                                       random_state=0),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)                                   # train on 280 rows
    print(f"{name:9s} test accuracy: {m.score(X_te, y_te):.3f}")
print("forest OOB score:", round(models["forest"].oob_score_, 3))
```

```text
one tree  test accuracy: 0.883
bagging   test accuracy: 0.925
forest    test accuracy: 0.917
forest OOB score: 0.882
```

![Decision boundaries on the two-moons data. One deep tree draws jagged islands around noisy points; bagging and the random forest both draw smoother boundaries.](figures/randomforest-boundaries.svg)

In the figure, compare the left panel's small rectangles around single points with the two ensembles. Both ensembles keep the overall moon shape and drop most of the islands. Here bagging scored one point higher than the forest (0.925 vs 0.917, a single test example out of 120). With only **2** features, `"sqrt"` means 1 feature per split, which is a blunt form of randomness. Random forests show their advantage more clearly with many features, as in the 20-feature example above. Always compare on your own validation data.

## Feature importance

### Impurity-based (`feature_importances_`)

For each feature: add up the impurity decrease of every split that uses it (weighted by how many training rows reach that node), average over all trees, and scale to sum to 1. This is the same calculation as for a single tree, averaged over the forest.

- **Pro**: free, instant, comes with the fitted model.
- **Con 1**: computed on the **training** data, so it reflects what the trees used to fit, including noise.
- **Con 2**: **biased towards high-cardinality features** (features with many distinct values: IDs, continuous measurements, categories with many levels). Such a feature offers many possible thresholds, so by chance it often finds a split that looks good on the training rows.

### Permutation importance

For each feature: take a validation or test set, **shuffle that one column** (breaking its link with the target), and measure how much the score drops. A big drop means the model really relies on the feature. A drop near zero means it doesn't. Use `sklearn.inspection.permutation_importance`.

![Importance of 7 features: 3 useful, 2 continuous noise, a random ID unique to every row, and a random yes/no column. Impurity importance gives every continuous noise column about 0.09; permutation importance gives all four noise columns about zero.](figures/randomforest-importance.svg)

In the figure, the gold bars (impurity) give the three continuous noise columns, including a random ID, 0.086–0.093 each, while the random yes/no column gets 0.013. The noise columns aren't useful; they just have more values to split on. The teal bars (permutation, on 300 test rows) are 0.17–0.22 for the useful features and between −0.008 and 0 for every noise column.

> **Trap:** Correlated features share importance. If two columns carry the same information, a forest splits between them, so each looks only half as important. With permutation importance, shuffling one of them hurts little because its twin covers for it. Group correlated features before drawing conclusions.

## Key hyperparameters (sklearn)

| Parameter | Default | What it does | Typical tuning |
|---|---|---|---|
| `n_estimators` | 100 | Number of trees | More is better until it plateaus (often 200–500); costs time, doesn't overfit |
| `max_features` | `"sqrt"` (classifier), `1.0` (regressor) | Features tried at each split | Main dial: try `"sqrt"`, `"log2"`, 0.2–0.5 |
| `max_depth` | `None` (grow until leaves are pure) | Maximum depth of each tree | Usually leave deep; limit if data is noisy or memory is tight |
| `min_samples_leaf` | 1 | Minimum rows in a leaf | 1–5 for classification, larger (5+) to smooth regression or noisy data |
| `bootstrap` | `True` | Bootstrap rows for each tree | `False` = every tree sees all rows (only feature randomness), no OOB score |
| `max_samples` | `None` (= $n$) | Bootstrap sample size, when `bootstrap=True` | Smaller (e.g. 0.5) = faster, more diverse trees |
| `class_weight` | `None` | Weights for classes | `"balanced"` or `"balanced_subsample"` (recomputed per bootstrap sample) for imbalance |
| `n_jobs` | `None` (1 core) | Cores used to train/predict in parallel | `-1` = all cores |
| `oob_score` | `False` | Compute the OOB score | `True` when you want a free validation check |
| `random_state` | `None` | Seed for bootstrap and feature draws | Set it for reproducible results |

## Comparison: tree vs bagging vs random forest vs XGBoost

| | Decision tree | Bagged trees | Random forest | XGBoost |
|---|---|---|---|---|
| How trees are built | One tree on all rows | Many deep trees, each on a bootstrap sample | Many deep trees, bootstrap sample **and** random features at each split | Many small trees, each fit to the errors (residuals) of the trees so far |
| Parallel or sequential | – | Parallel (independent) | Parallel (independent) | Sequential (each tree needs the previous ones) |
| Mainly reduces | – | Variance | Variance (more than bagging, via lower $\rho$) | Bias (and variance through regularisation) |
| Sources of randomness | None (except tie-breaks) | Rows | Rows + features per split | Optional: `subsample` (rows), `colsample_*` (features) |
| Split criterion | Entropy / Gini | Entropy / Gini | Entropy / Gini | Similarity score and gain from residuals |
| Combines by | – | Vote / average | Vote / average | Sum of $\eta \times$ tree outputs |
| Typical tree depth | Tuned | Deep | Deep | Shallow (3–8, default 6) |
| Tuning effort | Low | Low | Low (good defaults) | Medium–high |
| Interpretability | High: one flowchart | Low | Low | Low |
| Free validation | No | OOB score | OOB score | No (use early stopping on a validation set) |

## Assumptions & caveats

- **No distribution or scaling assumptions.** Like single trees, splits only use the order of feature values, so no standardising is needed and outliers in the inputs matter little.
- **Rows are assumed independent.** Bootstrapping treats rows as exchangeable. With time series or grouped data (several rows per customer), the OOB score is too optimistic; validate with time-based or grouped splits instead.
- **Class imbalance.** Leaves vote for the majority, and a rare class can be scarce in some bootstrap samples. Use `class_weight="balanced_subsample"`, choose the decision threshold from a PR curve, and report precision/recall ([Classification metrics](#/t/clsmetrics)).
- **Probabilities are averages of leaf fractions.** They're often pulled towards the middle (rarely near 0 or 1). If you need calibrated probabilities, check a calibration curve.
- **High-cardinality bias** in impurity importance (shown above), and **correlated features share importance**.
- **Leakage.** A forest exploits a leaked feature as happily as one tree does.
- **Missing values.** Recent sklearn versions (1.4+) let random forests handle `NaN` directly; with older versions you must impute first.

## Use cases

- **Tabular data** with mixed numeric/categorical features, non-linear effects and interactions, when you want a strong model with little tuning.
- A **robust baseline** before trying boosting: if XGBoost can't beat a default random forest, something else is wrong.
- When you want an **OOB estimate** and parallel training on many cores.
- Feature screening: which features matter (with permutation importance for the final ranking).
- Many practical areas: credit risk, churn, ecology and genomics (many features, modest rows), sensor data.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Usually much more accurate than one tree | Not interpretable as a single flowchart |
| Good results with default settings | Large memory footprint (hundreds of deep trees) |
| Trees train in parallel; OOB score for free | Slower to predict than one tree or a linear model |
| Hard to overfit by adding trees | Usually a little less accurate than well-tuned boosting on tabular data |
| No scaling; handles non-linearities and interactions | Impurity importance is biased; correlated features share credit |
| Copes well with many features and modest row counts | Doesn't extrapolate beyond the training range |

## Limitations

- **No extrapolation.** Predictions are averages of leaf values from the training range. For a regression trend that keeps rising, the forest flattens at the edge.
- **Bias isn't the target.** If each deep tree can't capture the pattern, averaging won't help; boosting attacks bias directly.
- **Sparse, very high-dimensional data** (e.g. TF-IDF text with 200,000 columns) suits linear models and Naive Bayes better: most random feature subsets contain only zeros for a given document.
- **Size.** Fully grown trees on large data can take gigabytes; limit `max_depth`, `min_samples_leaf` or `max_samples` if needed.

## Quick reference

| Term | Meaning |
|---|---|
| Random forest | Bagged decision trees + a random feature subset at each split |
| Randomness 1 | Row subsampling with replacement (bootstrap sample) per tree |
| Randomness 2 | `max_features` features drawn afresh at **every** split |
| Random subspace method | Features drawn once per tree (different method; `colsample_bytree`) |
| Why it works | Lower correlation $\rho$ → lower floor $\rho\sigma^2$ in the variance of the average |
| Cat table roots (100 trees) | Bagging 59/16/25, forest 30/44/26 (ear/face/whiskers) |
| `max_features` default | `"sqrt"` (classifier), `1.0` = all features (regressor) |
| Prediction | Majority vote (original) / averaged probabilities (sklearn) / average (regression) |
| OOB score | Accuracy using only trees that didn't see each row |
| Impurity importance | Training-set, biased towards many-valued features |
| Permutation importance | Score drop when one column is shuffled, on held-out data |
| Reduces | Variance |
