# Decision trees

A decision tree classifies an example by asking a short chain of yes/no questions about its features, choosing each question so the groups it creates are as pure as possible.

## The shared example: cat or not cat?

This is the example from Andrew Ng's lecture. We have 10 animals, 5 of them cats, and three features that each take only two values:

- **Ear shape**: pointy or floppy
- **Face shape**: round or not round
- **Whiskers**: present or absent

| # | Ear shape | Face shape | Whiskers | Cat? |
|---|---|---|---|---|
| 1 | Pointy | Round | Present | 1 |
| 2 | Floppy | Not round | Present | 1 |
| 3 | Floppy | Round | Absent | 0 |
| 4 | Pointy | Not round | Present | 0 |
| 5 | Pointy | Round | Present | 1 |
| 6 | Pointy | Round | Absent | 1 |
| 7 | Floppy | Not round | Absent | 0 |
| 8 | Pointy | Round | Absent | 1 |
| 9 | Floppy | Round | Absent | 0 |
| 10 | Floppy | Round | Absent | 0 |

The individual rows are my own reconstruction, checked in Python to match the lecture's counts (pointy: 4 of 5 are cats; round: 4 of 7; whiskers present: 3 of 4). The [XGBoost](#/t/xgboost) chapter uses the same 10 rows.

## What a tree is

- **Root node**: the first question. It sees all the training data.
- **Decision node**: any later question. It sees only the examples that reached it.
- **Branch**: one answer to a question (for example "pointy").
- **Leaf node**: the end of a path. No question, just a prediction: the majority class of the examples that landed there.
- **Depth**: how many questions from the root to a node.

![A decision tree for the cat example. Gold boxes ask questions, coloured boxes at the bottom are leaves that give the prediction.](figures/trees-anatomy.svg)

In the figure, follow one animal from the top: pointy ears, then a round face, ends in the "Cat" leaf. To predict, you only ever walk one path.

Learning a tree means answering two questions again and again:

1. **Which feature should this node split on?** The one with the highest information gain.
2. **When should we stop?** When the node is pure, or a limit (like maximum depth) is reached.

## Purity

**Purity** describes how mixed the classes are inside a node.

- **Pure**: every example in the node has the same class (all cats, or no cats).
- **Impure**: the node holds a mix of both classes.

Every split tries to make the child nodes purer than their parent. A pure node needs no more questions.

![Three nodes with 5 animals each. The two outer nodes are pure, the middle one is mixed.](figures/trees-purity.svg)

In the figure, teal circles are cats and wine circles are not cats. The middle node is mixed, so its entropy is high (0.97). The outer nodes are pure, so their entropy is 0.

## Entropy: a number for impurity

$$
H(p_1) = -p_1 \log_2 p_1 - (1 - p_1)\log_2(1 - p_1)
$$

- $p_1$ = fraction of examples in the node that are cats.
- $1 - p_1$ = fraction that are not cats.
- $\log_2$ = log base 2 (see [Logs & exponents](#/t/logs)).
- By convention $0 \times \log_2 0 = 0$, so a pure node has entropy 0.

In words: entropy is 0 when the node is pure and 1 when it is a 50/50 mix (you can't guess better than a coin flip).

![The entropy curve. It is 0 at both ends (pure nodes) and peaks at 1 when the node is half cats.](figures/trees-entropy.svg)

In the figure, the curve is symmetric. 4 cats out of 5 ($p_1 = 0.8$) and 1 cat out of 5 ($p_1 = 0.2$) both have entropy 0.72. Entropy measures how mixed a node is, not which class is winning.

### Worked example: root entropy

The root holds all 10 animals and 5 are cats, so $p_1 = 0.5$.

| Step | Calculation | Value |
|---|---|---|
| 1. Fraction of cats | $p_1 = 5/10$ | 0.5 |
| 2. Fraction not cats | $1 - p_1$ | 0.5 |
| 3. Log of each | $\log_2 0.5$ | −1 |
| 4. First term | $-0.5 \times (-1)$ | 0.5 |
| 5. Second term | $-0.5 \times (-1)$ | 0.5 |
| 6. Add | $H = 0.5 + 0.5$ | **1.0** |

> **Say it like this:** "The root is a 50/50 mix, so its entropy is 1, the maximum. Every split should push that number down."

## Information gain

**Information gain** is how much a split lowers entropy:

$$
\text{Gain} = H(\text{root}) - \Big( w_{\text{left}} \, H(\text{left}) + w_{\text{right}} \, H(\text{right}) \Big)
$$

- $w_{\text{left}}$ = fraction of the parent's examples that go left; $w_{\text{right}}$ the same for the right.
- The bracket is the **weighted average entropy** of the two children.

**Why weight by size?** A tiny pure branch shouldn't count as much as a big mixed one. Sending 1 animal left (pure) and 9 right (still mixed) has barely helped.

### Worked example: all three features at the root

$H(\text{root}) = 1.0$ for every candidate.

![The three candidate root splits, with the cats and non-cats in each branch and the entropy of each branch.](figures/trees-rootsplits.svg)

**Feature 1: Ear shape**

| | Animals | Cats | $p_1$ | Weight $w$ | $H$ | $w \times H$ |
|---|---|---|---|---|---|---|
| Pointy (left) | 5 | 4 | 0.80 | 0.5 | 0.7219 | 0.3610 |
| Floppy (right) | 5 | 1 | 0.20 | 0.5 | 0.7219 | 0.3610 |
| Weighted entropy | | | | | | 0.7219 |
| **Gain** | | | | | 1 − 0.7219 | **0.2781** |

**Feature 2: Face shape**

| | Animals | Cats | $p_1$ | Weight $w$ | $H$ | $w \times H$ |
|---|---|---|---|---|---|---|
| Round (left) | 7 | 4 | 0.571 | 0.7 | 0.9852 | 0.6897 |
| Not round (right) | 3 | 1 | 0.333 | 0.3 | 0.9183 | 0.2755 |
| Weighted entropy | | | | | | 0.9651 |
| **Gain** | | | | | 1 − 0.9651 | **0.0349** |

**Feature 3: Whiskers**

| | Animals | Cats | $p_1$ | Weight $w$ | $H$ | $w \times H$ |
|---|---|---|---|---|---|---|
| Present (left) | 4 | 3 | 0.75 | 0.4 | 0.8113 | 0.3245 |
| Absent (right) | 6 | 2 | 0.333 | 0.6 | 0.9183 | 0.5510 |
| Weighted entropy | | | | | | 0.8755 |
| **Gain** | | | | | 1 − 0.8755 | **0.1245** |

### Which feature wins?

| Rank | Feature | Gain | Why |
|---|---|---|---|
| 1 | Ear shape | 0.28 | Both branches are 80/20, the most lopsided split |
| 2 | Whiskers | 0.12 | "Present" is 75% cats, but "absent" is still mixed |
| 3 | Face shape | 0.03 | Round is 57% cats, almost a coin flip |

The tree is **greedy**: at each node it takes the biggest gain right now. So **ear shape becomes the root**.

> **Key idea:** information gain ranks the features at each node, and the highest gain wins. The ranking is recomputed at every node, using only the examples that reached it.

## One level down

Repeat the same process on each branch, using only that branch's animals.

### Left branch: pointy ears (5 animals, 4 cats, H = 0.7219)

| Split | Left: animals (cats), H | Right: animals (cats), H | Weighted H | Gain |
|---|---|---|---|---|
| Face shape | round: 4 (4), 0 | not round: 1 (0), 0 | 0.0000 | **0.7219** |
| Whiskers | present: 3 (2), 0.9183 | absent: 2 (2), 0 | 0.5510 | 0.1710 |

Face shape splits this branch perfectly: every pointy-eared, round-faced animal is a cat. Useless at the root, perfect here.

### Right branch: floppy ears (5 animals, 1 cat, H = 0.7219)

| Split | Left: animals (cats), H | Right: animals (cats), H | Weighted H | Gain |
|---|---|---|---|---|
| Face shape | round: 3 (0), 0 | not round: 2 (1), 1.0 | 0.4000 | 0.3219 |
| Whiskers | present: 1 (1), 0 | absent: 4 (0), 0 | 0.0000 | **0.7219** |

Whiskers splits this branch perfectly. Every leaf is now pure, so the tree stops.

![The finished tree: ear shape at the root, face shape under pointy, whiskers under floppy. All four leaves are pure.](figures/trees-final.svg)

In the figure, each node shows how many animals reach it and its entropy. Entropy goes from 1.00 at the root, to 0.72, to 0 in every leaf.

## Numbers, overfitting and stopping

**Number features** (like weight): the tree tries questions such as "weight ≤ 9.2?" at the midpoint between each pair of sorted values, and keeps the one with the highest gain.

**Overfitting**: if you never stop, the tree keeps splitting until each leaf holds one example. It has memorised the training data, noise included (see [Bias & variance](#/t/biasvar)).

![On a noisy toy dataset, training accuracy climbs towards 100% as max_depth grows, while validation accuracy peaks early and then drifts down.](figures/trees-overfit.svg)

In the figure, look at the widening gap between the gold and teal lines after depth 2–4. That gap is overfitting.

The fix is to stop early, tuning these on a validation set:

- `max_depth`: maximum number of questions per path.
- `min_samples_leaf`: every leaf must hold at least this many examples.
- `min_samples_split`: don't split a node smaller than this.

## Code: the cat table in sklearn

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# the 10 animals as 1/0 (1 = pointy, round, whiskers present, cat)
df = pd.DataFrame({
    "pointy":   [1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    "round":    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    "whiskers": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    "cat":      [1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
})
X, y = df[["pointy", "round", "whiskers"]], df["cat"]

# sklearn's default is 'gini'; 'entropy' matches these notes
tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
tree.fit(X, y)
print(export_text(tree, feature_names=list(X.columns)))   # same tree as by hand
```

The printout splits on `pointy` first, then `round` under pointy and `whiskers` under floppy: the tree we built by hand. (Gini is just another impurity measure; the splitting logic is the same.)

**Feature importance** (`feature_importances_`) adds up how much each feature lowered impurity across the whole tree; check it with permutation importance on validation data, because it can favour features with many distinct values.

## In my accidents project

In [US Accidents](#/p/acc) I wrapped sklearn's tree in a small class with train, validation and test scoring ([src/decision_tree.py](https://github.com/jag5m33/py_acc/blob/main/src/decision_tree.py)):

```python
self.model = DecisionTreeClassifier(max_depth = max_depth, # how many splits from root to leaf before making a decision; number of leaves double per level
                               min_samples_leaf= min_samples_leaf, # minimum samples per leaf; with a dataset of 2 million, 500 is adequate to make aconviced decision of equal disitribution
                               min_samples_split  = min_samples_split,
                               random_state= 42,
                               max_features= max_features,
                               class_weight= class_weight)
```

- **`max_depth=10`**: at most 10 questions per path, so at most 1,024 leaves, far fewer than 1.98 million training rows. The tree can't memorise them.
- **`min_samples_leaf=500`**: every leaf holds at least 500 accidents, so its "severe" probability is an average over many rows.
- **Result:** train and validation scores were within 0.3 points, so no sign of overfitting. On validation: precision 0.783, recall 0.689, F1 0.733, PR-AUC 0.822 (baseline 0.340).
- **Feature importance:** road type carried **64%**, then location (17%) and the engineered `constraint_count` (7%). This matched the EDA, where severe rates ran from 14.7% on local roads to 80.4% on interstates.

## Assumptions & caveats

- **No scaling needed.** Splits only compare "≤ threshold", so the size of the numbers doesn't matter.
- **No linearity or normality assumptions.**
- **Unstable.** A few changed rows can change the root split, and with it the whole tree.
- **Greedy.** Each split is the best one right now; the tree never looks ahead.
- **Class imbalance.** Leaves predict the majority class, so a rare class can be swamped. Use `class_weight="balanced"` and judge with precision/recall ([Classification metrics](#/t/clsmetrics)).
- **Leakage.** A tree will happily split on a leaked feature. In the accidents project, end time and duration were dropped for this reason.

## Use cases

- You need a model you can **explain** ("if interstate and night, then high risk").
- Tabular data with a mix of number and category features.
- A quick **baseline**, before ensembles.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Easy to read; each prediction is a path you can explain | Unstable to small data changes |
| No scaling needed | Overfits easily without limits |
| Captures non-linear effects and interactions | Greedy, may miss the best tree |
| Fast to train and predict | Usually less accurate than random forests or boosting |

## Limitations

- Predictions are step-shaped (constant inside each leaf).
- Can't predict outside the range of values seen in training.
- A depth-10 tree with up to 1,024 leaves, like mine, is no longer readable by eye.

## Where trees go next

1. [Bagging](#/t/bagging): many deep trees, each on a random resample of the rows, then vote.
2. [Random forest](#/t/randomforest): bagging plus random features at every split, so the trees differ more.
3. [XGBoost](#/t/xgboost): small trees built one after another, each fixing the previous trees' errors.

## Quick reference

| Term | Meaning / formula |
|---|---|
| Root / decision / leaf node | First question / later question / final prediction |
| Pure node | All one class (entropy 0) |
| Entropy | $-p_1\log_2 p_1 - (1-p_1)\log_2(1-p_1)$ |
| Entropy range | 0 (pure) to 1 (50/50) |
| Information gain | $H(\text{root}) - (w_L H_L + w_R H_R)$ |
| Cat example gains at root | ear 0.278, whiskers 0.125, face 0.035 |
| Greedy | take the biggest gain at each node |
| Stop early | `max_depth`, `min_samples_leaf`, `min_samples_split` |
