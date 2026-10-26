# XGBoost

XGBoost builds many small decision trees one after another, each one trained to fix the errors the previous trees left behind, and adds their outputs together to make a prediction.

## Before you start

This chapter builds on [Decision trees](#/t/trees): nodes, leaves, purity, and choosing the split with the highest gain. XGBoost uses the same idea of "pick the split that makes the children purer", but it measures purity with residuals instead of entropy.

**Terms used below**

- **Ensemble**: a model made by combining many models.
- **Residual**: actual minus predicted, $r = y - p$. If a cat ($y = 1$) is predicted $p = 0.5$, its residual is $+0.5$.
- **Log-odds**: $\log\frac{p}{1-p}$. This is the scale XGBoost adds trees on for classification, the same scale as [logistic regression](#/t/logreg). $p = 0.5$ gives log-odds 0.
- **Sigmoid**: turns log-odds back into a probability, $p = \frac{1}{1 + e^{-\text{log-odds}}}$.

## How XGBoost builds on decision trees

Each tree in XGBoost is the same kind of tree as in the [trees chapter](#/t/trees): a root question, decision nodes, leaves, splits chosen greedily one node at a time. Three things change:

| | Decision tree | XGBoost tree |
|---|---|---|
| What a tree predicts | The class directly (majority in the leaf) | A **correction** (leaf output) added to the running log-odds |
| What it is trained on | The labels | The **residuals** the earlier trees left behind |
| How splits are scored | Entropy and information gain | **Similarity score** and its gain (worked below) |
| How many trees | One | Many **small** trees, built **in sequence** |

[Bagging](#/t/bagging) and [random forest](#/t/randomforest) also combine many trees, but their trees are deep, **independent** and built **in parallel** on random resamples, then averaged. XGBoost's trees depend on each other: tree 2 can't be built until tree 1 has made its predictions. The next section compares the two approaches side by side.

## One tree vs random forest vs boosting

![Left: bagging builds independent trees on bootstrap samples and averages them. Right: boosting builds trees in sequence, and each one fits the residuals of the trees before it.](figures/xgboost-bagboost.svg)

- **One decision tree**: easy to read, but high variance. Change a few rows and the tree can change completely.
- **Bagging (random forest)**: train many deep trees **in parallel**. Each tree sees a different bootstrap sample (rows drawn at random with replacement) and, at each split, a random subset of features. Then average their votes. Averaging cancels out the trees' individual quirks, so it lowers **variance**.
- **Boosting (gradient boosting, XGBoost)**: train many small trees **in sequence**. Tree 1 makes a rough guess. Tree 2 is trained on the residuals tree 1 left, tree 3 on what's still left, and so on. The prediction is the starting value plus a small step ($\eta$) times each tree's output. Each tree removes some systematic error, so this lowers **bias**.

In the figure, the arrows on the right carry residuals forward. That is the whole difference: forests average independent trees, boosting chains dependent ones.

![Residuals for the 10 animals as trees are added (real xgboost, depth-2 trees). They start at ±0.5 and shrink towards 0.](figures/xgboost-residuals.svg)

In the figure, each bar is one animal's residual. Teal bars are cats that still need a higher probability; wine bars are non-cats that still need a lower one. As trees are added, the bars shrink: each tree fixes part of what is left.

> **Key idea:** Bagging = parallel trees, averaged, less variance. Boosting = sequential trees, each fixing the last one's errors, less bias.

**XGBoost** ("eXtreme Gradient Boosting") is gradient boosting with extras:

- regularisation built into the split and leaf maths ($\lambda$, $\gamma$)
- second-order (curvature) information for each example
- handling of missing values
- a very fast, parallel implementation

## The data: same 10 animals

The same table as in the [trees chapter](#/t/trees), with features as 1/0 (1 = pointy, round, whiskers present):

| # | Pointy ears | Round face | Whiskers | Cat $y$ |
|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 |
| 2 | 0 | 0 | 1 | 1 |
| 3 | 0 | 1 | 0 | 0 |
| 4 | 1 | 0 | 1 | 0 |
| 5 | 1 | 1 | 1 | 1 |
| 6 | 1 | 1 | 0 | 1 |
| 7 | 0 | 0 | 0 | 0 |
| 8 | 1 | 1 | 0 | 1 |
| 9 | 0 | 1 | 0 | 0 |
| 10 | 0 | 1 | 0 | 0 |

Counts (checked in Python): pointy 5 (4 cats), floppy 5 (1); round 7 (4), not round 3 (1); whiskers 4 (3), none 6 (2).

## Worked example, step by step

This follows the standard hand-worked method for XGBoost classification, as taught in StatQuest's and Krish Naik's walkthroughs. Settings: $\lambda = 1$ (regularisation), $\eta = 0.3$ (learning rate, the xgboost default), $\gamma = 0$, trees of depth 1 (one split each). Every number below was computed in Python.

### Step 1: initial prediction

Start every animal at $p = 0.5$, which is log-odds $= \log\frac{0.5}{0.5} = 0$. This is a "know nothing" guess (xgboost's `base_score=0.5`).

### Step 2: residuals

$r_i = y_i - p_i$. Cats get $1 - 0.5 = +0.5$; non-cats get $0 - 0.5 = -0.5$.

| Animals | $y$ | $p$ | Residual | $p(1-p)$ |
|---|---|---|---|---|
| Cats: 1, 2, 5, 6, 8 | 1 | 0.5 | +0.5 | 0.25 |
| Not cats: 3, 4, 7, 9, 10 | 0 | 0.5 | −0.5 | 0.25 |

### Step 3: similarity score

For any node (a group of animals):

$$
\text{Similarity} = \frac{\left(\sum r_i\right)^2}{\sum p_i(1-p_i) + \lambda}
$$

**Where it comes from.** The numerator is the squared total residual: how strongly the group "pulls" in one direction. The denominator $\sum p(1-p)$ measures how many examples there are, weighted by how uncertain each prediction still is. (It is the second derivative of log-loss, called the **hessian**, which xgboost reports as `cover`.) $\lambda$ is added so small groups can't score high by chance. In XGBoost's derivation, the similarity is (up to a factor of ½) the amount the loss drops if this node gets its best possible output value.

**Link to purity.** Similarity is the boosting version of purity from the trees chapter:

- If the residuals in a node **agree** (all positive, or all negative), they add up to something big, and the similarity is **high**. That is a **pure** node.
- If they **cancel** (some +0.5, some −0.5), the sum is near 0, and the similarity is **near 0**. That is a **mixed** node.

![Four residuals that agree give similarity 2.0; four that cancel give similarity 0.](figures/xgboost-similarity.svg)

In the figure, both nodes have four animals, so their denominators are equal ($4 \times 0.25 + 1 = 2$). Only the sum of residuals differs: $2^2/2 = 2$ against $0^2/2 = 0$.

**Root similarity.** All 10 animals: $\sum r = 5(0.5) + 5(-0.5) = 0$, so

$$
\text{Sim}_{\text{root}} = \frac{0^2}{10 \times 0.25 + 1} = \frac{0}{3.5} = 0
$$

### Step 4: try every split and compute the gain

$$
\text{Gain} = \text{Sim}_{\text{left}} + \text{Sim}_{\text{right}} - \text{Sim}_{\text{root}}
$$

This is the same idea as information gain: how much purer the children are than the parent.

**Ear shape**

| Branch | Animals | Sum of residuals | $\sum p(1-p)$ | Similarity |
|---|---|---|---|---|
| Pointy | 1, 4, 5, 6, 8 (4 cats) | $4(0.5) + 1(-0.5) = 1.5$ | 1.25 | $1.5^2 / 2.25 = 1.000$ |
| Floppy | 2, 3, 7, 9, 10 (1 cat) | $1(0.5) + 4(-0.5) = -1.5$ | 1.25 | $(-1.5)^2 / 2.25 = 1.000$ |
| **Gain** | | | | $1 + 1 - 0 = $ **2.000** |

**Face shape**

| Branch | Animals | Sum of residuals | $\sum p(1-p)$ | Similarity |
|---|---|---|---|---|
| Round | 7 animals (4 cats) | $4(0.5) + 3(-0.5) = 0.5$ | 1.75 | $0.25 / 2.75 = 0.091$ |
| Not round | 3 animals (1 cat) | $1(0.5) + 2(-0.5) = -0.5$ | 0.75 | $0.25 / 1.75 = 0.143$ |
| **Gain** | | | | **0.234** |

**Whiskers**

| Branch | Animals | Sum of residuals | $\sum p(1-p)$ | Similarity |
|---|---|---|---|---|
| Present | 4 animals (3 cats) | $3(0.5) + 1(-0.5) = 1.0$ | 1.00 | $1 / 2 = 0.500$ |
| Absent | 6 animals (2 cats) | $2(0.5) + 4(-0.5) = -1.0$ | 1.50 | $1 / 2.5 = 0.400$ |
| **Gain** | | | | **0.900** |

![Gain for each candidate root split in tree 1. Ear shape has the largest gain.](figures/xgboost-gains.svg)

In the figure, the order is the same as with entropy in the trees chapter: **ear shape > whiskers > face shape**. Different maths, same conclusion: ear shape gives the purest children.

### Step 5: gamma pruning

$\gamma$ (gamma) is the minimum gain a split must achieve to be kept:

$$
\text{keep the split only if } \text{Gain} - \gamma > 0
$$

With $\gamma = 0$, the ear split (gain 2.0) is kept. If we set $\gamma = 2.1$, then $2.0 - 2.1 < 0$, the split would be pruned, and tree 1 would be a single leaf. (I checked this in xgboost: with `gamma=2.1` the dump is just `leaf=0`.) A larger $\gamma$ means fewer, more conservative splits.

### Step 6: leaf output values

$$
\text{Output} = \frac{\sum r_i}{\sum p_i(1-p_i) + \lambda}
$$

This is the same as the similarity formula, but **without squaring** the numerator. The output keeps its sign, so it says which way to push the prediction.

| Leaf | Sum of residuals | Denominator | Output |
|---|---|---|---|
| Pointy | 1.5 | 1.25 + 1 = 2.25 | **+0.6667** |
| Floppy | −1.5 | 1.25 + 1 = 2.25 | **−0.6667** |

Without $\lambda$, the outputs would be $\pm 1.2$. So $\lambda$ **shrinks** leaf values towards 0, which makes each tree less extreme.

### Step 7: update the log-odds

$$
\text{new log-odds} = \text{old log-odds} + \eta \times \text{output}
$$

- Pointy: $0 + 0.3 \times 0.6667 = +0.2$
- Floppy: $0 + 0.3 \times (-0.6667) = -0.2$

$\eta$ (the learning rate) means each tree takes only a small step. Many small steps generalise better than a few big ones.

### Step 8: convert to probability with the sigmoid

$$
p = \frac{1}{1 + e^{-\text{log-odds}}}
$$

- Pointy: $p = 1/(1 + e^{-0.2}) = 0.5498$
- Floppy: $p = 1/(1 + e^{0.2}) = 0.4502$

### Step 9: new residuals

| Animal | Ear | $y$ | New $p$ | New residual | Was |
|---|---|---|---|---|---|
| 1, 5, 6, 8 (cats, pointy) | Pointy | 1 | 0.5498 | +0.4502 | +0.5 |
| 4 (not cat, pointy) | Pointy | 0 | 0.5498 | −0.5498 | −0.5 |
| 2 (cat, floppy) | Floppy | 1 | 0.4502 | +0.5498 | +0.5 |
| 3, 7, 9, 10 (not cats, floppy) | Floppy | 0 | 0.4502 | −0.4502 | −0.5 |

Eight animals moved closer to their true label. The two "odd" animals (cat 2 with floppy ears, non-cat 4 with pointy ears) moved slightly further away. Log-loss over all 10 fell from 0.6931 to 0.6381.

### Step 10: the second tree

Repeat steps 3–8 using the new residuals and the new $p(1-p)$ values (now $0.5498 \times 0.4502 = 0.2475$ for every animal):

| Split | Sim left | Sim right | Gain |
|---|---|---|---|
| Ear shape | 0.6992 | 0.6992 | **1.3985** |
| Whiskers | 0.4073 | 0.3262 | 0.7335 |
| Face shape | 0.0742 | 0.1163 | 0.1905 |

- Ear shape wins again, with a smaller gain: the easy part of the error is already gone.
- Leaf outputs: $\pm 1.2508/(1.2376 + 1) = \pm 0.5590$.
- Log-odds become $\pm(0.2 + 0.3 \times 0.559) = \pm 0.3677$, so $p$ = 0.5909 (pointy) and 0.4091 (floppy). Log-loss drops to 0.5996.

With depth 1, every tree can only use one feature, so ear shape keeps winning. With `max_depth=2`, later trees also split on face shape and whiskers. After 30 depth-2 trees, the training probabilities are 0.925–0.953 for the cats with pointy ears, 0.845 for cat 2, and 0.04–0.19 for the non-cats.

### Check against the real library

```python
import numpy as np
import xgboost as xgb

X = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],
              [1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])   # pointy, round, whiskers
y = np.array([1,1,0,0,1,1,0,1,0,0])                        # 1 = cat

model = xgb.XGBClassifier(
    n_estimators=1,        # one tree, so we can compare with the hand calculation
    max_depth=1,           # one split
    learning_rate=0.3,     # eta
    reg_lambda=1,          # lambda in the similarity and output formulas
    gamma=0,               # no pruning
    base_score=0.5,        # start every prediction at p = 0.5
    min_child_weight=0,    # don't block small leaves (10 rows is tiny)
)
model.fit(X, y)
print(model.get_booster().get_dump(with_stats=True)[0])   # the tree as text
print(model.predict_proba(X)[:, 1].round(4))               # P(cat) per animal
```

```text
0:[f0<1] yes=1,no=2,missing=2,gain=2,cover=2.5
	1:leaf=-0.200000018,cover=1.25
	2:leaf=0.200000018,cover=1.25

[0.5498 0.4502 0.4502 0.5498 0.5498 0.5498 0.4502 0.5498 0.4502 0.4502]
```

How to read it, and how it compares with the hand calculation:

- `f0<1` is "pointy < 1", i.e. floppy goes to node 1 (yes) and pointy to node 2 (no). Same split.
- `gain=2` matches our gain of 2.000. In tree 2 the library reports `gain=1.39845`, matching our 1.3985.
- `cover=1.25` is $\sum p(1-p)$ in each leaf, and `cover=2.5` at the root. These match the denominators before $\lambda$ is added.
- **The one difference:** the dump shows `leaf=±0.2`, not $\pm 0.6667$. xgboost stores the leaf value **already multiplied by $\eta$**: $0.3 \times 0.6667 = 0.2$. In tree 2 it stores $\pm 0.1677 = 0.3 \times 0.559$. The maths is the same; only where $\eta$ is applied differs.
- The predicted probabilities match ours exactly (0.5498 / 0.4502), and after 2 trees (0.5909 / 0.4091).
- `min_child_weight=0` switches off the minimum-cover rule. For this split it makes no difference (each leaf's cover, 1.25, is above the default of 1), but on 10 rows, deeper splits would create leaves with cover below 1, and the default would block them.

> **Trap:** The XGBoost paper writes the split gain with an extra factor of ½ and subtracts $\gamma$ inside the formula. The number the library prints after `gain=` is the plain "left + right − root", which is why it matches the hand calculation directly. $\gamma$ is applied separately, when xgboost decides whether to keep the split: with `gamma=1.9` the split stays and still prints `gain=2`; with `gamma=2.1` it is removed.

## Hyperparameters

| Parameter (sklearn name) | What it does | If overfitting | If underfitting |
|---|---|---|---|
| `n_estimators` | Number of trees (boosting rounds) | Fewer, or use early stopping | More |
| `learning_rate` (eta, $\eta$) | Step size: how much of each tree's output is added. Default 0.3 | Lower (and add trees) | Raise |
| `max_depth` | Maximum depth of each tree. Default 6 | Lower (3–6 is typical) | Raise |
| `min_child_weight` | Minimum $\sum p(1-p)$ (cover) a leaf must have. Default 1 | Raise | Lower |
| `gamma` ($\gamma$) | Minimum gain needed to keep a split. Default 0 | Raise | Lower |
| `reg_lambda` ($\lambda$) | L2 penalty: shrinks leaf outputs, lowers similarity. Default 1 | Raise | Lower |
| `reg_alpha` ($\alpha$) | L1 penalty on leaf outputs: can push small leaves to exactly 0. Default 0 | Raise | Lower |
| `subsample` | Fraction of rows sampled for each tree | Lower (0.6–0.9) | Raise towards 1 |
| `colsample_bytree` | Fraction of features sampled for each tree | Lower | Raise towards 1 |
| `scale_pos_weight` | Weight on the positive class, for imbalance. Rule of thumb: negatives ÷ positives | (tunes recall vs precision, not overfitting) | |
| `early_stopping_rounds` | Stop if the validation score hasn't improved for this many rounds | Use it (e.g. 50) | Increase patience |

`learning_rate` and `n_estimators` work together: halve the learning rate and you usually need about twice as many trees.

## Code: training with early stopping

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

# synthetic data: 1,500 rows, 20 features, 6 of them useful, 12% noisy labels
X, y = make_classification(n_samples=1500, n_features=20, n_informative=6,
                           flip_y=0.12, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0)

model = xgb.XGBClassifier(
    n_estimators=600,            # upper limit; early stopping picks the real number
    learning_rate=0.1,           # smaller steps than the default 0.3
    max_depth=6,
    eval_metric="logloss",       # score watched on the eval set
    early_stopping_rounds=50,    # stop after 50 rounds with no improvement
)
model.fit(X_tr, y_tr,
          eval_set=[(X_tr, y_tr), (X_va, y_va)],  # last = early stopping
          verbose=False)

print("best round:", model.best_iteration)  # round with best val score
curves = model.evals_result()               # log-loss per round

# feature importance: total gain each feature contributed across all splits
imp = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
print(imp.sort_values(ascending=False).head())
```

![Training log-loss keeps falling, while validation log-loss bottoms out at round 64. Early stopping keeps the best round and stops 50 rounds later.](figures/xgboost-earlystop.svg)

In the figure, the dashed line marks the best validation round. After it, the trees only fit training noise, the same overfitting story as with tree depth.

**Feature importance in xgboost** comes in several types:

- `weight`: how many splits use the feature
- `gain`: average gain per split
- `total_gain`: total gain across all splits
- `cover`: how many examples (by hessian) the feature's splits affect

`model.feature_importances_` in the sklearn wrapper uses `gain` by default for tree models. Different types can rank features differently, so state which one you used.

## In my accidents project

In [US Accidents](#/p/acc) the XGBoost model uses the same interface as my decision tree class ([src/xgboost.py](https://github.com/jag5m33/py_acc/blob/main/src/xgboost.py)):

```python
self.model = xgb.XGBClassifier(objective = 'binary:logistic',
                          random_state = 42)
def train(self):
    print("training model for binary classification")
    self.model.fit(self.x_train,self.y_train)
    print(f" number of trees across full model: {len(self.model.get_booster().get_dump())}")
```

Validation results (from the README), against the same split and the same 0.5 threshold:

| Model | Precision | Recall | F1 | PR-AUC | Accuracy |
|---|---|---|---|---|---|
| Baseline | – | – | – | 0.340 | 0.660 |
| Decision tree (depth 10, min leaf 500) | 0.783 | 0.689 | 0.733 | 0.822 | 0.830 |
| XGBoost (default settings) | 0.821 | 0.763 | 0.791 | 0.871 | 0.863 |

- XGBoost, with **no tuning at all**, beat the depth-limited tree on every metric. The biggest gain was in recall (+7.4 points), meaning more severe accidents were caught.
- Train and validation scores were within 0.3 points, so there was no sign of overfitting.
- Not done yet (listed in the README as next steps): tuning on validation, early stopping, choosing a threshold from the PR curve, and scoring the held-back test set.
- About 34% of rows are severe. That is moderately imbalanced, so `scale_pos_weight` (≈ 66/34 ≈ 1.9) is one option to try if recall matters more than precision.

## Comparison table

| | Single tree | Random forest | Gradient boosting | XGBoost |
|---|---|---|---|---|
| How trees are built | One tree | Many independent deep trees on bootstrap samples | Many shallow trees in sequence, each fitting residuals | Same as gradient boosting, plus regularised splits and leaves |
| Mainly reduces | – | Variance | Bias | Bias (with regularisation for variance) |
| Combines by | – | Averaging votes | Adding $\eta \times$ tree outputs | Adding $\eta \times$ tree outputs |
| Typical tree depth | Tuned (often deep) | Deep | Shallow (3–8) | Shallow (default 6) |
| Regularisation | Depth and leaf limits | Randomness and averaging | Learning rate, depth | $\lambda$, $\alpha$, $\gamma$, learning rate, subsampling |
| Missing values | Recent sklearn versions handle NaN; older ones need imputation | Same as single tree | Depends on library | Built in: learns a default direction |
| Interpretability | High | Low | Low | Low |
| Tuning effort | Low | Low (good defaults) | Medium | Medium–high |
| Typical accuracy on tabular data | Lowest | Good | Very good | Very good |

## Assumptions & caveats

- **No distribution or scaling assumptions.** Like single trees, splits depend only on the order of values.
- **Overfits with too many rounds or too-deep trees.** Training error always keeps falling, so use a validation set with early stopping.
- **Needs tuning.** Defaults are often decent (as in my project), but the best results need tuning of learning rate, depth, regularisation and subsampling, ideally on a validation set or with cross-validation.
- **Less interpretable.** Hundreds of trees can't be read by eye. Use feature importance (and say which type) or SHAP values to explain predictions.
- **Class imbalance.** By default, every example counts equally. Use `scale_pos_weight`, judge with precision/recall/PR-AUC, and choose the threshold deliberately ([Classification metrics](#/t/clsmetrics)).
- **No extrapolation.** Predictions outside the training range of a feature stay flat at the edge-leaf value. This matters for time trends and for regression.
- **Leakage.** A powerful learner finds leaked features very quickly. Check suspiciously good scores.
- **Correlated features.** Importance can be split between correlated features in an arbitrary way, so don't read too much into the ranking of near-duplicates.

## Use cases

- Structured / tabular data (rows and columns of mixed numeric and categorical features). This is where boosted trees are usually the strongest choice.
- Classification and regression where accuracy matters more than a model you can read line by line: risk scoring, churn, fraud, demand forecasting with engineered features.
- Large datasets with missing values, where its speed and built-in handling of missing values help.
- A strong benchmark to beat before trying anything more complex.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Often the most accurate model on tabular data | Many hyperparameters to tune |
| Regularisation built in ($\lambda$, $\alpha$, $\gamma$) | Overfits if rounds or depth aren't controlled |
| Handles missing values natively | Hard to interpret compared with one tree |
| No scaling needed; captures non-linearities and interactions | Sequential: trees within a model can't be trained in parallel (only splits can) |
| Fast, parallelised split finding | Can't extrapolate beyond the training range |
| Early stopping and feature importance built in | Not suited to raw images, audio or text without feature engineering |

## Limitations

- Boosting fits noise if pushed too far, so noisy labels need stronger regularisation and early stopping.
- Results can depend on hyperparameter choices more than with random forests.
- Probabilities are not always well calibrated. Check with a calibration curve if you need true probabilities, not just rankings.
- On unstructured data (images, raw text), deep learning usually wins. XGBoost's strength is tabular data.

## Quick reference

| Term | Meaning / formula |
|---|---|
| Bagging | Parallel independent trees, averaged; lowers variance |
| Boosting | Sequential trees fitting residuals; lowers bias |
| Initial prediction | $p = 0.5$, log-odds 0 |
| Residual | $r = y - p$ |
| Similarity | $(\sum r)^2 / (\sum p(1-p) + \lambda)$: high when pure, near 0 when mixed |
| Gain | $\text{Sim}_L + \text{Sim}_R - \text{Sim}_{\text{root}}$ |
| Gamma pruning | Keep split only if $\text{Gain} - \gamma > 0$ |
| Leaf output | $\sum r / (\sum p(1-p) + \lambda)$ |
| Update | log-odds $\leftarrow$ log-odds $+ \eta \times$ output |
| Probability | $p = 1/(1 + e^{-\text{log-odds}})$ |
| Cat example, tree 1 | Ear gain 2.000 > whiskers 0.900 > face 0.234; outputs ±0.667; p = 0.550 / 0.450 |
| xgboost dump | `leaf` = $\eta \times$ output; `cover` = $\sum p(1-p)$ |
