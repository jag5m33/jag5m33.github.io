# XGBoost

XGBoost builds many small decision trees one after another, each one trained to fix the errors the previous trees left behind, and adds their outputs together to make a prediction.

## Terms used below

This chapter builds on [Decision trees](#/t/trees). XGBoost uses the same idea, "pick the split that makes the children purer", but it measures purity with residuals instead of entropy.

- **Residual**: actual minus predicted, $r = y - p$. If a cat ($y = 1$) is predicted $p = 0.5$, its residual is $+0.5$.
- **Log-odds**: the "score" scale XGBoost adds trees on, the same $z$ as in [logistic regression](#/t/logreg). A score of 0 means $p = 0.5$.
- **Sigmoid**: turns that score back into a probability, $p = 1/(1 + e^{-\text{score}})$.

## Bagging vs boosting

![Left: bagging builds independent trees on bootstrap samples and averages them. Right: boosting builds trees in sequence, and each one fits the residuals of the trees before it.](figures/xgboost-bagboost.svg)

- **Bagging / random forest** ([Bagging](#/t/bagging), [Random forest](#/t/randomforest)): many deep trees built **in parallel**, each on a random resample, then they vote. Fixes **variance** (instability).
- **Boosting (XGBoost)**: many small trees built **in sequence**. Tree 1 makes a rough guess. Tree 2 is trained on the errors tree 1 left, tree 3 on what's still left, and so on. Fixes **bias** (too simple).

In the figure, the arrows on the right carry residuals forward. That is the whole difference: forests average independent trees; boosting chains dependent ones.

![Residuals for the 10 animals as trees are added (real xgboost, depth-2 trees). They start at ±0.5 and shrink towards 0.](figures/xgboost-residuals.svg)

In the figure, each bar is one animal's residual. Teal bars are cats that still need a higher probability; wine bars are non-cats that still need a lower one. As trees are added, the bars shrink: each tree fixes part of what is left.

> **Key idea:** bagging = parallel trees, voted, less variance. Boosting = trees in a chain, each fixing the last one's errors, less bias.

**XGBoost** ("eXtreme Gradient Boosting") is gradient boosting with built-in regularisation, handling of missing values and a very fast implementation.

## The data: same 10 animals

The same table as in the [trees chapter](#/t/trees), with features as 1/0:

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

## Worked example, step by step

This follows the standard hand-worked method taught in StatQuest's and Krish Naik's walkthroughs. Settings: $\lambda = 1$, learning rate $\eta = 0.3$, $\gamma = 0$, and each tree has just one split. Every number was computed in Python.

### Step 1: start at 0.5

Every animal starts at $p = 0.5$ (score 0). A "know nothing" guess.

### Step 2: residuals

$r = y - p$. Cats get $1 - 0.5 = +0.5$; non-cats get $0 - 0.5 = -0.5$.

| Animals | $y$ | $p$ | Residual |
|---|---|---|---|
| Cats: 1, 2, 5, 6, 8 | 1 | 0.5 | +0.5 |
| Not cats: 3, 4, 7, 9, 10 | 0 | 0.5 | −0.5 |

### Step 3: similarity score

For any group of animals (a node):

$$
\text{Similarity} = \frac{(\text{sum of residuals})^2}{\sum p(1-p) + \lambda}
$$

- **Top:** add up the residuals, then square.
- **Bottom:** for each animal add $p(1-p)$; at $p = 0.5$ that's $0.25$ each, so it is basically "how many animals, weighted by how unsure we still are". Then add $\lambda$.
- $\lambda$ (lambda) is a regularisation number. It stops tiny groups from scoring high by luck.

**This is purity again:**

- Residuals that **agree** (all + or all −) add up to something big: **high** similarity. A **pure** node.
- Residuals that **cancel** (some +, some −) add up to about 0: similarity **near 0**. A **mixed** node.

![Four residuals that agree give similarity 2.0; four that cancel give similarity 0.](figures/xgboost-similarity.svg)

In the figure, both nodes have four animals, so the bottom is the same ($4 \times 0.25 + 1 = 2$). Only the sum differs: $2^2/2 = 2$ against $0^2/2 = 0$.

**Root:** all 10 animals, sum of residuals $= 5(0.5) + 5(-0.5) = 0$, so similarity $= 0^2 / 3.5 = 0$.

### Step 4: try every split, compute the gain

$$
\text{Gain} = \text{Sim}_{\text{left}} + \text{Sim}_{\text{right}} - \text{Sim}_{\text{root}}
$$

Same idea as information gain: how much purer the children are than the parent.

**Ear shape**

| Branch | Animals | Sum of residuals | $\sum p(1-p)$ | Similarity |
|---|---|---|---|---|
| Pointy | 1, 4, 5, 6, 8 (4 cats) | $4(0.5) + 1(-0.5) = 1.5$ | 1.25 | $1.5^2 / 2.25 = 1.000$ |
| Floppy | 2, 3, 7, 9, 10 (1 cat) | $1(0.5) + 4(-0.5) = -1.5$ | 1.25 | $(-1.5)^2 / 2.25 = 1.000$ |
| **Gain** | | | | $1 + 1 - 0 = $ **2.000** |

**Face shape**

| Branch | Animals | Sum of residuals | $\sum p(1-p)$ | Similarity |
|---|---|---|---|---|
| Round | 7 animals (4 cats) | $0.5$ | 1.75 | $0.25 / 2.75 = 0.091$ |
| Not round | 3 animals (1 cat) | $-0.5$ | 0.75 | $0.25 / 1.75 = 0.143$ |
| **Gain** | | | | **0.234** |

**Whiskers**

| Branch | Animals | Sum of residuals | $\sum p(1-p)$ | Similarity |
|---|---|---|---|---|
| Present | 4 animals (3 cats) | $1.0$ | 1.00 | $1 / 2 = 0.500$ |
| Absent | 6 animals (2 cats) | $-1.0$ | 1.50 | $1 / 2.5 = 0.400$ |
| **Gain** | | | | **0.900** |

![Gain for each candidate root split in tree 1. Ear shape has the largest gain.](figures/xgboost-gains.svg)

In the figure, the order is the same as with entropy in the trees chapter: **ear shape > whiskers > face shape**. Different maths, same conclusion.

### Step 5: gamma pruning

$\gamma$ (gamma) is the minimum gain a split must reach to be kept: keep it only if $\text{Gain} - \gamma > 0$.

With $\gamma = 0$, the ear split (gain 2.0) is kept. With $\gamma = 2.1$, $2.0 - 2.1 < 0$, so the split would be cut and tree 1 would be a single leaf. Bigger $\gamma$ = fewer, more careful splits.

### Step 6: leaf outputs

$$
\text{Output} = \frac{\text{sum of residuals}}{\sum p(1-p) + \lambda}
$$

The same as similarity, but **not squared**, so it keeps its sign and says which way to push.

| Leaf | Sum of residuals | Bottom | Output |
|---|---|---|---|
| Pointy | 1.5 | 1.25 + 1 = 2.25 | **+0.6667** |
| Floppy | −1.5 | 1.25 + 1 = 2.25 | **−0.6667** |

Without $\lambda$ the outputs would be $\pm 1.2$, so $\lambda$ makes each tree less extreme.

### Step 7: update with the learning rate

$$
\text{new score} = \text{old score} + \eta \times \text{output}
$$

- Pointy: $0 + 0.3 \times 0.6667 = +0.2$
- Floppy: $0 + 0.3 \times (-0.6667) = -0.2$

$\eta$ (eta, the learning rate) means each tree only takes a small step. Many small steps generalise better than a few big ones.

### Step 8: back to a probability

Use the sigmoid:

- Pointy: $p = 1/(1 + e^{-0.2}) = 0.5498$
- Floppy: $p = 1/(1 + e^{0.2}) = 0.4502$

### Step 9: new residuals, then repeat

| Animal | Ear | $y$ | New $p$ | New residual | Was |
|---|---|---|---|---|---|
| 1, 5, 6, 8 (cats, pointy) | Pointy | 1 | 0.5498 | +0.4502 | +0.5 |
| 4 (not cat, pointy) | Pointy | 0 | 0.5498 | −0.5498 | −0.5 |
| 2 (cat, floppy) | Floppy | 1 | 0.4502 | +0.5498 | +0.5 |
| 3, 7, 9, 10 (not cats, floppy) | Floppy | 0 | 0.4502 | −0.4502 | −0.5 |

Eight animals moved closer to their true label. The two odd ones out (cat 2 with floppy ears, non-cat 4 with pointy ears) moved slightly further away. Log loss over all 10 fell from 0.6931 to 0.6381.

Tree 2 repeats steps 3–8 on these new residuals. Ear shape wins again, but with a smaller gain (1.3985), because the easy part of the error is already gone. After tree 2, $p$ = 0.5909 (pointy) and 0.4091 (floppy). With deeper trees, later trees also use face shape and whiskers to fix cat 2 and non-cat 4.

Running the real `xgboost` library with the same settings gives the same split, the same gain (2) and the same probabilities (0.5498 / 0.4502); its printed leaf values are $\pm 0.2$ because it stores them already multiplied by $\eta$.

## Key hyperparameters

| Parameter | What it does | If overfitting |
|---|---|---|
| `n_estimators` | Number of trees | Fewer, or use early stopping |
| `learning_rate` ($\eta$) | Size of each tree's step. Default 0.3 | Lower (and add trees) |
| `max_depth` | Depth of each tree. Default 6 | Lower (3–6 is typical) |
| `gamma` ($\gamma$) | Minimum gain to keep a split. Default 0 | Raise |
| `reg_lambda` ($\lambda$) | Shrinks leaf outputs. Default 1 | Raise |

`learning_rate` and `n_estimators` work together: halve the learning rate and you usually need about twice as many trees.

## Code: training with early stopping

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1500, n_features=20, n_informative=6,
                           flip_y=0.12, random_state=0)       # noisy toy data
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0)

model = xgb.XGBClassifier(
    n_estimators=600,            # upper limit; early stopping picks the real number
    learning_rate=0.1,           # smaller steps than the default 0.3
    eval_metric="logloss",       # score watched on the validation set
    early_stopping_rounds=50,    # stop after 50 rounds with no improvement
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print("best round:", model.best_iteration)
```

![Training log-loss keeps falling, while validation log-loss bottoms out at round 64. Early stopping keeps the best round and stops 50 rounds later.](figures/xgboost-earlystop.svg)

In the figure, the dashed line marks the best validation round. After it, new trees only fit training noise: the same overfitting story as with tree depth.

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

Validation results (from the README), same split and the same 0.5 threshold:

| Model | Precision | Recall | F1 | PR-AUC | Accuracy |
|---|---|---|---|---|---|
| Baseline | – | – | – | 0.340 | 0.660 |
| Decision tree (depth 10, min leaf 500) | 0.783 | 0.689 | 0.733 | 0.822 | 0.830 |
| XGBoost (default settings) | 0.821 | 0.763 | 0.791 | 0.871 | 0.863 |

- XGBoost, with **no tuning at all**, beat the tree on every metric. The biggest gain was recall (+7.4 points): more severe accidents caught.
- Train and validation scores were within 0.3 points, so no sign of overfitting.
- Next steps (in the README): tuning, early stopping, choosing a threshold from the PR curve, and scoring the test set.
- About 34% of rows are severe. `scale_pos_weight` (about 66/34 ≈ 1.9) is one option if recall matters more than precision.

## Comparison: tree vs bagging vs random forest vs XGBoost

| | Decision tree | Bagged trees | Random forest | XGBoost |
|---|---|---|---|---|
| How trees are built | One tree on all rows | Many deep trees, each on a bootstrap sample | Bootstrap sample **and** random features at every split | Many small trees, each fit to the residuals so far |
| Parallel or in sequence | – | Parallel | Parallel | In sequence |
| Mainly reduces | – | Variance | Variance (more than bagging) | Bias |
| Split chosen by | Entropy / Gini | Entropy / Gini | Entropy / Gini | Similarity score and gain |
| Combines by | – | Vote | Vote | Adding $\eta \times$ each tree's output |
| Typical depth | Tuned | Deep | Deep | Shallow (default 6) |
| Tuning effort | Low | Low | Low | Medium–high |
| Readable? | Yes, one flowchart | No | No | No |

## Assumptions & caveats

- **No scaling needed.** Splits only use the order of values.
- **Overfits with too many trees or too-deep trees.** Use a validation set with early stopping.
- **Needs tuning** for the best results (learning rate, depth, regularisation).
- **Hard to read.** Hundreds of trees; use feature importance (say which type) to explain.
- **Class imbalance.** Use `scale_pos_weight`, judge with precision/recall/PR-AUC, and choose the threshold on purpose ([Classification metrics](#/t/clsmetrics)).
- **Leakage.** A powerful model finds leaked features fast. Be suspicious of amazing scores.

## Use cases

- **Tabular data** (rows and columns). This is where boosted trees are usually the strongest choice.
- Risk scoring, churn, fraud, demand forecasting with engineered features.
- A strong benchmark to beat before trying anything more complex.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Often the most accurate model on tabular data | Many settings to tune |
| Regularisation built in ($\lambda$, $\gamma$) | Overfits if trees/depth aren't controlled |
| Handles missing values itself | Hard to interpret |
| No scaling; captures interactions | Trees must be built one after another |
| Early stopping built in | Can't predict beyond the training range |

## Limitations

- Noisy labels need stronger regularisation and early stopping, or boosting fits the noise.
- On images, audio and raw text, deep learning usually wins.

## Quick reference

| Term | Meaning / formula |
|---|---|
| Boosting | trees in sequence, each fixing the residuals; lowers bias |
| Start | $p = 0.5$ (score 0) |
| Residual | $r = y - p$ |
| Similarity | (sum of r)² / (sum of p(1 − p) + λ): high when pure, near 0 when mixed |
| Gain | Sim left + Sim right − Sim root |
| Gamma | keep a split only if Gain − γ > 0 |
| Leaf output | sum of r / (sum of p(1 − p) + λ) |
| Update | score ← score + η × output |
| Cat example, tree 1 | ear gain 2.000 > whiskers 0.900 > face 0.234; outputs ±0.667; p = 0.550 / 0.450 |
