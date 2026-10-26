# Bagging & bootstrap sampling

Bagging trains many copies of the same model on random resamples of the training rows and combines their votes, which smooths out the wobbles that make a single deep decision tree unreliable.

## Before you start

This chapter builds on [Decision trees](#/t/trees) (nodes, leaves, information gain) and on the idea of variance from [Bias & variance](#/t/biasvar). It reuses the same 10 animals (ear shape, face shape, whiskers, cat or not).

**Terms used below**

- **Ensemble**: one model made by combining many models (each one is called a **base model** or **base learner**).
- **Variance** (of a model): how much its predictions change if you train it on a slightly different sample of data. High variance = unstable.
- **Bias**: error that comes from the model being too simple to capture the real pattern. It doesn't go away with more data.
- **Bagging**: short for **b**ootstrap **agg**regat**ing**. "Bootstrap" = resample the rows; "aggregating" = combine the models.

## Why combine many models?

### Wisdom of crowds

Ask one person to guess the number of sweets in a jar and they may be way off. Average the guesses of 100 people and the answer is often surprisingly close. Each person's error is partly random: some guess too high, some too low, and the errors cancel when you average.

### The catch: errors must not all be the same

Averaging only helps if the models make **different** mistakes. If all 100 people copied the same wrong guess, the average would be just as wrong. So an ensemble needs base models that are:

1. **Reasonably good on their own** (better than a coin flip).
2. **Different from each other**, so their errors are not perfectly correlated.

Bagging gets point 2 by giving every model a different random version of the training data. Deep decision trees are the classic choice for point 1. They have low bias and very high variance, and variance is exactly what averaging removes.

> **Key idea:** An ensemble beats its members when the members are individually decent and their mistakes don't line up.

## Bootstrap sampling

### Definition

A **bootstrap sample** is a new dataset made by drawing rows from the original **at random, with replacement**, until it has the same number of rows as the original ($n$).

- **With replacement** means a row goes back into the pot after it is drawn, so it can be drawn again. The same row can appear 2, 3 or more times in one sample.
- Because some rows are drawn more than once, other rows are **never drawn**. Those left-out rows are called **out-of-bag (OOB)** rows for that sample.
- This is also called **row subsampling with replacement**. It is the "bootstrap" from statistics, where it is used to estimate the uncertainty of a statistic by resampling.

### Picture it

![Ten numbered rows (teal = cat, wine = not cat) and three bootstrap samples drawn from them with a seeded random generator. Gold outlines mark rows that were drawn more than once; the right-hand column lists the rows each sample missed.](figures/bagging-bootstrap.svg)

In the figure, look at sample 1: rows 3, 5 and 9 each appear twice, so rows 6, 8 and 10 never get in. Every sample still has 10 rows, but only 7 or 8 **different** ones. Each sample is a slightly different version of the same data, and that difference is what makes the trees differ.

### The maths: how many rows get left out?

Take one particular row, say row 1. Each draw picks one of $n$ rows with equal chance.

| Step | Quantity | Formula |
|---|---|---|
| 1 | Chance row 1 is picked on one draw | $\frac{1}{n}$ |
| 2 | Chance row 1 is **not** picked on one draw | $1 - \frac{1}{n}$ |
| 3 | Chance it's missed on all $n$ draws (draws are independent, so multiply) | $\left(1 - \frac{1}{n}\right)^n$ |
| 4 | What that tends to as $n$ grows | $e^{-1} \approx 0.368$ |

Step 4 is a standard limit: $\left(1 + \frac{x}{n}\right)^n \to e^{x}$ as $n \to \infty$, here with $x = -1$ (see [Logs & exponents](#/t/logs) for $e$).

So about **36.8%** of rows are left out of each bootstrap sample, and about **1 − 0.368 = 63.2%** of the original rows appear in it at least once.

**Worked for two sizes** (computed in Python):

| $n$ | $P(\text{row never drawn}) = (1 - 1/n)^n$ | Share of rows in the sample | Expected unique rows |
|---|---|---|---|
| 10 | $0.9^{10} = 0.3487$ | 0.6513 | 6.5 of 10 |
| 1000 | $0.999^{1000} = 0.3677$ | 0.6323 | 632 of 1000 |
| limit | $e^{-1} = 0.3679$ | 0.6321 | 63.2% |

![The chance that a given row is never drawn, for n from 1 to 100. It climbs quickly and levels off at e to the power minus 1, about 0.368.](figures/bagging-oob.svg)

In the figure, the curve is already 0.349 at $n = 10$ and barely moves after that. For any realistic dataset, "about a third of rows are out-of-bag" is a safe rule.

> **Say it like this:** "Each bootstrap sample is the same size as the data, but because rows can repeat, it only contains about 63% of the distinct rows. The other 37% are out-of-bag."

## Out-of-bag (OOB) rows: a free validation set

Every tree in a bagged ensemble has its own OOB rows: rows it never saw while training. That gives a validation score without setting any data aside:

1. For each training row, find the trees for which it was OOB (about 37% of them).
2. Let only those trees vote on that row.
3. Compare the vote with the row's true label.
4. The accuracy over all rows is the **OOB score**.

It's a fair estimate because no row is ever judged by a tree that trained on it. With 200 trees and 280 training rows, each row is OOB for about $200 \times 0.367 \approx 73$ trees, which is plenty of voters.

> **Trap:** The OOB score is computed on the training set, so anything you did before splitting (scaling, feature selection, target encoding fitted on all rows) can still leak into it. It replaces a validation split; it doesn't replace a final untouched test set.

## Bagged decision trees

### The recipe

1. Choose $B$, the number of trees (`n_estimators`).
2. For $b = 1, \dots, B$: draw a bootstrap sample and grow **one deep, unpruned tree** on it.
3. Combine the trees:
   - **Classification**: majority vote of the $B$ predicted classes (or, as sklearn does, average the predicted probabilities and take the largest).
   - **Regression**: average the $B$ predicted numbers.

The trees are **independent**: tree 7 doesn't know tree 6 exists. So they can all be trained at the same time (in parallel), unlike boosting ([XGBoost](#/t/xgboost)) where each tree depends on the previous one.

### Small worked example: three stumps on the cat table

To keep it hand-sized, use the three bootstrap samples from the figure above and grow a **depth-1 tree** (a **stump**: one split, two leaves) on each, with entropy as in the trees chapter. A new animal arrives: **floppy ears, round face, whiskers present**.

| Tree | Bootstrap sample (row numbers) | Info gain: ear / face / whiskers | Root chosen | Leaf the new animal lands in | P(cat) | Vote |
|---|---|---|---|---|---|---|
| 1 | 1, 2, 3, 3, 4, 5, 5, 7, 9, 9 | 0.256 / 0.006 / **0.610** | Whiskers | present: 4 cats of 5 | 0.80 | cat |
| 2 | 1, 2, 3, 3, 6, 7, 8, 9, 9, 10 | **0.557** / 0.007 / 0.322 | Ear shape | floppy: 1 cat of 7 | 0.14 | not cat |
| 3 | 2, 3, 4, 5, 5, 6, 7, 7, 8, 10 | **0.278** / 0.125 / 0.125 | Ear shape | floppy: 1 cat of 5 | 0.20 | not cat |

- **Majority vote:** 1 cat, 2 not cat, so the ensemble says **not cat**.
- **Averaged probability** (sklearn's way): $(0.80 + 0.14 + 0.20)/3 = 0.38$, below 0.5, so also **not cat**.

![The three stumps. Tree 1 happened to get a sample where whiskers split best; trees 2 and 3 chose ear shape. The outlined leaf is where the new animal lands in each tree.](figures/bagging-votes.svg)

In the figure, notice that tree 1 asks a different question at its root. Its bootstrap sample had no whisker-less cats at all, so whiskers looked perfect. That is the bootstrap creating diversity. Also notice that two of the three trees still chose ear shape: bagging alone often produces trees that look alike. Keep that in mind for the variance formula below.

All numbers in the table come from running sklearn's `DecisionTreeClassifier(max_depth=1, criterion="entropy")` on the samples drawn with `np.random.default_rng(2)`.

### What it does to the decision boundary

![Left: one fully grown tree on a noisy two-moons dataset, with a jagged boundary and small islands around single points. Right: 200 bagged trees, with a smoother boundary.](figures/bagging-boundary.svg)

In the figure, the single tree (left) draws tiny rectangles around individual noisy points. Each bagged tree does the same, but around **different** points, because each saw a different sample. Averaging 200 of them keeps the shape they agree on and washes out most of the islands. On the 120 held-out points, test accuracy rose from 0.883 (one tree) to 0.925 (bagging). The OOB score was 0.889, a slightly pessimistic but sensible estimate.

## Why bagging reduces variance, not bias

### Bias stays about the same

Every bagged tree is the same kind of model as a single deep tree, trained on similar data. On average it makes the same systematic errors, so the average of many such trees has about the **same bias**. If one tree is too simple for the problem (high bias), bagging 500 of them won't fix it. That is the job of boosting.

### Variance drops, by how much?

Let each tree's prediction (at some input) have variance $\sigma^2$, and let any two trees' predictions have correlation $\rho$ (rho, between 0 and 1). The variance of the average of $B$ trees is:

$$
\operatorname{Var}\left(\frac{1}{B}\sum_{b=1}^{B} f_b\right) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2
$$

**Where it comes from.** The variance of a sum is the sum of all the variances plus all the covariances:

- There are $B$ variance terms, each $\sigma^2$.
- There are $B(B-1)$ covariance terms (every ordered pair of different trees), each $\rho\sigma^2$.
- Dividing the sum by $B$ divides its variance by $B^2$:

$$
\frac{B\sigma^2 + B(B-1)\rho\sigma^2}{B^2} = \frac{\sigma^2}{B} + \rho\sigma^2 - \frac{\rho\sigma^2}{B} = \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{B}
$$

**Reading it piece by piece:**

| Piece | What it is | What happens as you add trees |
|---|---|---|
| $\sigma^2$ | How much a single tree wobbles | Fixed by the type of tree |
| $\frac{(1-\rho)\sigma^2}{B}$ | The independent part of the trees' errors | Shrinks towards 0 as $B$ grows |
| $\rho\sigma^2$ | The shared part: errors all trees make together | **Never shrinks**, however many trees |

**Worked numbers** ($\sigma^2 = 1$):

| $\rho$ | $B = 1$ | $B = 10$ | $B = 100$ | Floor ($B \to \infty$) |
|---|---|---|---|---|
| 0.3 | 1.00 | 0.37 | 0.307 | 0.3 |
| 0.6 | 1.00 | 0.64 | 0.604 | 0.6 |

![Variance of the average against the number of trees, for four correlations. Each curve flattens onto its floor, rho times sigma squared.](figures/bagging-variance.svg)

In the figure, with $\rho = 0$ (completely independent trees) the variance heads to 0. With $\rho = 1$ (identical trees) averaging does nothing at all. Real bagged trees sit in between, and after a few dozen trees adding more barely helps: you hit the floor $\rho\sigma^2$.

> **Key idea:** More trees shrink only the $(1-\rho)\sigma^2/B$ part. To go lower you must lower $\rho$: make the trees **less alike**.

### The catch, and what comes next

Bootstrap samples overlap a lot (each has ~63% of the same rows), so bagged trees are strongly correlated. Worse, if one feature is very strong, almost every tree picks it at the root, and the trees end up with the same shape (as two of our three stumps did with ear shape). [Random forest](#/t/randomforest) attacks exactly this: it also limits which features each split may look at, which lowers $\rho$ and so lowers the floor.

## Code: bagged trees in sklearn

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# 400 noisy points in two interleaving half-moons
X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeClassifier(random_state=0)          # one deep, unpruned tree
tree.fit(X_tr, y_tr)

bag = BaggingClassifier(
    DecisionTreeClassifier(),   # base model: one deep tree (the `estimator`)
    n_estimators=200,           # B = number of bootstrap samples / trees
    bootstrap=True,             # sample rows WITH replacement (the default)
    oob_score=True,             # score rows with trees that didn't see them
    random_state=0,             # makes the bootstrap draws repeatable
    n_jobs=-1,                  # train trees in parallel on all CPU cores
)
bag.fit(X_tr, y_tr)

print("one tree test acc:", round(tree.score(X_te, y_te), 3))
print("bagging test acc: ", round(bag.score(X_te, y_te), 3))
print("bagging OOB score:", round(bag.oob_score_, 3))   # free validation estimate
```

```text
one tree test acc: 0.883
bagging test acc:  0.925
bagging OOB score: 0.889
```

Notes on the code:

- `BaggingClassifier` works with **any** base model, not just trees. For regression use `BaggingRegressor`, which averages.
- In sklearn 1.2+ the base model argument is called `estimator` (the old name `base_estimator` was removed in 1.4). Passing it first, positionally, works in all recent versions.
- `bag.estimators_samples_` holds the row indices each tree trained on. On this data, each tree left out between 31% and 42% of the 280 training rows, 36.7% on average, as the maths predicts.
- `max_samples` (default 1.0) sets the bootstrap sample size as a fraction of $n$; `max_features` (default 1.0) lets each tree see only a random subset of columns (that is per tree, not per split; see the random forest chapter for the difference).

## Assumptions & caveats

- **Needs an unstable base model.** Bagging helps high-variance models (deep trees). Bagging a stable, high-bias model such as [linear regression](#/t/linreg) changes almost nothing, because every copy is nearly identical.
- **Doesn't fix bias.** If each tree underfits, the average underfits too.
- **Correlated trees limit the gain.** The $\rho\sigma^2$ floor. Strong dominant features make it worse.
- **Rows are assumed exchangeable.** Bootstrapping treats rows as independent draws. For time series (see [Autocorrelation](#/t/autocorr)) or grouped data (several rows per patient), random row sampling mixes past and future, or the same patient across samples, and the OOB score becomes too optimistic.
- **Class imbalance.** A rare class can be missing or nearly missing from some bootstrap samples. Use `class_weight` on the trees, or stratified/balanced resampling, and judge with precision and recall ([Classification metrics](#/t/clsmetrics)).
- **No scaling needed** when the base model is a tree, for the same reason as single trees: splits only use the order of values.
- **Leakage still applies.** Bagging a model that uses a leaked feature just gives you many confident wrong models.

## Use cases

- You have a model that fits the training data well but is **unstable**: deep trees are the textbook case.
- You want a strong, low-effort improvement over one tree, with a **built-in validation estimate** (OOB) when data is scarce.
- Training can be **parallelised** across cores or machines, because trees are independent.
- As the stepping stone to random forests, which are bagged trees plus one extra trick and are used far more often in practice.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Reduces variance; usually beats one deep tree clearly | Doesn't reduce bias |
| OOB score gives validation for free | Gains stop at the $\rho\sigma^2$ floor; trees are often too alike |
| Trees are independent, so training is parallel | Loses the readability of a single tree |
| Works with any base model | B times the training cost and memory of one model |
| Few settings to tune; more trees never overfits in the usual sense | Usually beaten by random forest on the same data |

## Limitations

- **Interpretability is gone.** You can't draw 200 trees as one flowchart. Feature importance (averaged over trees) is the usual summary.
- **Diminishing returns.** Past a certain number of trees, accuracy flattens. More trees cost time but don't hurt accuracy.
- **Same blind spots as trees.** No extrapolation beyond the training range; axis-aligned splits (smoother, but still a staircase).
- **Plain bagging is rarely the final choice.** In practice people reach straight for [random forest](#/t/randomforest) (less correlated trees) or [XGBoost](#/t/xgboost) (lower bias).

## Quick reference

| Term | Meaning / formula |
|---|---|
| Bagging | Bootstrap aggregating: one model per bootstrap sample, then vote or average |
| Bootstrap sample | $n$ rows drawn with replacement from $n$ rows |
| P(row never drawn) | $(1 - 1/n)^n \to e^{-1} \approx 0.368$ |
| Unique rows per sample | about 63.2% (0.651 at $n = 10$, 0.632 at $n = 1000$) |
| OOB rows | The ~36.8% of rows a given tree never saw |
| OOB score | Accuracy when each row is judged only by trees that didn't train on it |
| Combine (classification) | Majority vote, or average probability (sklearn) |
| Combine (regression) | Average |
| Variance of the average | $\rho\sigma^2 + (1-\rho)\sigma^2/B$ |
| Reduces | Variance, not bias |
| Floor | $\rho\sigma^2$: lower it by making trees less alike (random forest) |
| sklearn | `BaggingClassifier(DecisionTreeClassifier(), n_estimators=..., oob_score=True)` |
