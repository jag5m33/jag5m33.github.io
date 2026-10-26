# Decision trees

A decision tree classifies an example by asking a short chain of yes/no questions about its features, choosing each question so the groups it creates are as pure as possible.

## The shared example: cat or not cat?

This is the example from Andrew Ng's lecture. We have 10 animals, 5 of them cats, and three binary features (each feature takes only two values):

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

The individual rows are my own reconstruction. They were checked in Python to match the counts from the lecture:

| Feature value | Animals | Cats |
|---|---|---|
| Ear pointy / floppy | 5 / 5 | 4 / 1 |
| Face round / not round | 7 / 3 | 4 / 1 |
| Whiskers present / absent | 4 / 6 | 3 / 2 |

The [XGBoost](#/t/xgboost) chapter uses the same 10 rows, so you can compare the two methods number for number.

## What a tree is

- **Root node**: the first question. It sees all the training data.
- **Decision node** (internal node): any later question. It sees only the examples that reached it.
- **Branch**: one answer to a question (for example "pointy"). It leads to the next node.
- **Leaf node**: the end of a path. It holds no question, only a prediction, which is the majority class of the training examples that landed there.
- **Depth**: the number of questions from the root to a node. The root is at depth 0.

![A decision tree for the cat example. Gold boxes ask questions, coloured boxes at the bottom are leaves that give the prediction.](figures/trees-anatomy.svg)

In the figure, follow one animal from the top: pointy ears, then a round face, ends in the "Cat" leaf. To predict, you only ever walk one path.

Learning a tree means answering two questions again and again:

1. **Which feature should this node split on?** (Answer: the one with the highest information gain.)
2. **When should we stop splitting?** (Answer: the stopping criteria further down.)

## Purity

**Purity** describes how mixed the classes are inside a node.

- A node is **pure** when every example in it has the same class (all cats, or no cats).
- A node is **impure** (mixed) when it holds examples from both classes.

The goal of every split is to make the child nodes purer than their parent. A pure node needs no more questions, because you already know the answer.

![Three nodes with 5 animals each. The two outer nodes are pure, the middle one is mixed.](figures/trees-purity.svg)

In the figure, teal circles are cats and wine circles are not cats. The middle node is mixed, so its entropy is high (0.97). The outer nodes are pure, so their entropy is 0.

## Entropy: a number for impurity

### Definition

Let $p_1$ be the fraction of examples in a node that are cats (class 1). Then $1 - p_1$ is the fraction that are not cats. **Entropy** is:

$$
H(p_1) = -p_1 \log_2 p_1 - (1 - p_1)\log_2(1 - p_1)
$$

- $\log_2$ is the logarithm base 2 (see [Logs & exponents](#/t/logs)). Because of base 2, entropy is measured in **bits**.
- **Convention:** $0 \cdot \log_2 0 = 0$. Strictly $\log_2 0$ is undefined, but as $p \to 0$ the product $p \log_2 p$ goes to 0, so we define it that way. This makes $H(0) = H(1) = 0$.

### Where it comes from

$-\log_2 p$ is the "surprise" of an event with probability $p$. A certain event ($p = 1$) gives zero surprise. A rare event gives a big surprise. Entropy is the **average surprise**: each class's surprise weighted by how often that class appears. A pure node has no surprise. A 50/50 node is the most uncertain, because you can't guess better than a coin flip.

### Shape

![The entropy curve. It is 0 at both ends (pure nodes) and peaks at 1 when the node is half cats.](figures/trees-entropy.svg)

In the figure, the curve is symmetric. 4 cats out of 5 ($p_1 = 0.8$) and 1 cat out of 5 ($p_1 = 0.2$) both have entropy 0.72. Entropy measures how mixed a node is, not which class is winning.

### Worked example: root entropy

The root holds all 10 animals and 5 are cats, so $p_1 = 5/10 = 0.5$.

| Step | Calculation | Value |
|---|---|---|
| 1. Fraction of cats | $p_1 = 5/10$ | 0.5 |
| 2. Fraction not cats | $1 - p_1$ | 0.5 |
| 3. Log of each | $\log_2 0.5$ | −1 |
| 4. First term | $-0.5 \times (-1)$ | 0.5 |
| 5. Second term | $-0.5 \times (-1)$ | 0.5 |
| 6. Add | $H(S) = 0.5 + 0.5$ | **1.0** |

> **Say it like this:** "The root is a 50/50 mix, so its entropy is 1 bit, the maximum. Every split should push that number down."

## Information gain

### Definition

**Information gain** is how much a split reduces entropy:

$$
\text{Gain} = H(\text{root}) - \Big( w_{\text{left}} \, H(\text{left}) + w_{\text{right}} \, H(\text{right}) \Big)
$$

- $w_{\text{left}}$ = fraction of the parent's examples that go left. $w_{\text{right}}$ is the same for the right branch, so $w_{\text{left}} + w_{\text{right}} = 1$.
- The bracket is the **weighted average entropy** of the children.

### Why weight by size

A tiny pure branch shouldn't count as much as a big mixed one. If a split sends 1 animal left (pure) and 9 right (still 50/50), it has barely helped. The weights make each child count in proportion to how many examples it holds.

### Worked example: all three features at the root

$H(\text{root}) = 1.0$ for every candidate, because the root is the same.

![The three candidate root splits, with the cats and non-cats in each branch and the entropy of each branch.](figures/trees-rootsplits.svg)

**Feature 1: Ear shape**

| | Animals | Cats | $p_1$ | Weight $w$ | $H$ | $w \times H$ |
|---|---|---|---|---|---|---|
| Pointy (left) | 5 | 4 | 0.80 | 5/10 = 0.5 | 0.7219 | 0.3610 |
| Floppy (right) | 5 | 1 | 0.20 | 5/10 = 0.5 | 0.7219 | 0.3610 |
| Weighted entropy | | | | | | 0.7219 |
| **Gain** | | | | | 1 − 0.7219 | **0.2781** |

**Feature 2: Face shape**

| | Animals | Cats | $p_1$ | Weight $w$ | $H$ | $w \times H$ |
|---|---|---|---|---|---|---|
| Round (left) | 7 | 4 | 0.571 | 7/10 = 0.7 | 0.9852 | 0.6897 |
| Not round (right) | 3 | 1 | 0.333 | 3/10 = 0.3 | 0.9183 | 0.2755 |
| Weighted entropy | | | | | | 0.9651 |
| **Gain** | | | | | 1 − 0.9651 | **0.0349** |

**Feature 3: Whiskers**

| | Animals | Cats | $p_1$ | Weight $w$ | $H$ | $w \times H$ |
|---|---|---|---|---|---|---|
| Present (left) | 4 | 3 | 0.75 | 4/10 = 0.4 | 0.8113 | 0.3245 |
| Absent (right) | 6 | 2 | 0.333 | 6/10 = 0.6 | 0.9183 | 0.5510 |
| Weighted entropy | | | | | | 0.8755 |
| **Gain** | | | | | 1 − 0.8755 | **0.1245** |

### How the gain orders the features

| Rank | Feature | Gain | Why |
|---|---|---|---|
| 1 | Ear shape | 0.28 | Both branches are 80/20, the most lopsided split available |
| 2 | Whiskers | 0.12 | "Present" is 75% cats, but "absent" is still quite mixed |
| 3 | Face shape | 0.03 | Round is 57% cats, almost a coin flip, so we learn very little |

The tree is **greedy**: at each node it takes the split with the biggest gain right now. So **ear shape becomes the root**. Face shape is almost useless at the root, but as you'll see, it becomes the perfect question one level down.

> **Key idea:** Information gain ranks the features at each node. The highest gain wins that node. The ranking is recomputed at every node, on only the examples that reached it.

## Recursing one level down

Now repeat the whole process separately on each branch, using only the examples in that branch. Ear shape is already used, and every animal in a branch has the same ear shape, so its gain there is 0.

### Left branch: pointy ears (5 animals, 4 cats, H = 0.7219)

Animals 1, 4, 5, 6, 8.

| Split | Left: animals (cats), H | Right: animals (cats), H | Weighted H | Gain |
|---|---|---|---|---|
| Face shape | round: 4 (4), 0 | not round: 1 (0), 0 | 0.0000 | **0.7219** |
| Whiskers | present: 3 (2), 0.9183 | absent: 2 (2), 0 | 0.5510 | 0.1710 |

Face shape splits this branch perfectly: every pointy-eared, round-faced animal is a cat. The gain equals the whole parent entropy, which is the largest gain possible.

### Right branch: floppy ears (5 animals, 1 cat, H = 0.7219)

Animals 2, 3, 7, 9, 10.

| Split | Left: animals (cats), H | Right: animals (cats), H | Weighted H | Gain |
|---|---|---|---|---|
| Face shape | round: 3 (0), 0 | not round: 2 (1), 1.0 | 0.4000 | 0.3219 |
| Whiskers | present: 1 (1), 0 | absent: 4 (0), 0 | 0.0000 | **0.7219** |

Whiskers splits this branch perfectly. Every leaf is now pure, so the tree stops.

![The finished tree: ear shape at the root, face shape under pointy, whiskers under floppy. All four leaves are pure.](figures/trees-final.svg)

In the figure, each node shows how many animals reach it and its entropy. Each split's gain is printed above it. Entropy goes from 1.00 at the root, to 0.72, to 0 in every leaf.

## Stopping criteria

If you never stop, the tree keeps splitting until every leaf holds one example, which means it has memorised the training set. Common rules for making a node a leaf:

| Rule | What it means | sklearn parameter |
|---|---|---|
| Node is pure | All examples have one class, so there's nothing left to learn | (always applied) |
| Maximum depth reached | The path from the root already has this many questions | `max_depth` |
| Too few examples to split | Don't split a node with fewer than this many examples | `min_samples_split` |
| Child would be too small | Don't allow a split that leaves a child with fewer than this many examples | `min_samples_leaf` |
| Gain too small | The best split improves purity by less than a threshold | `min_impurity_decrease` |

## Continuous features (short)

For a number such as weight, the tree turns the feature into a yes/no question: "weight ≤ t?". To find t:

1. Sort the distinct values.
2. Try the **midpoint** between each neighbouring pair as a threshold.
3. Compute the information gain for each threshold, exactly as above.
4. Keep the threshold with the highest gain. That becomes this feature's candidate split, and it competes with the other features as usual.

![Made-up weights for the 10 animals (top) and the gain for every candidate threshold (bottom). The best split is weight ≤ 9.2.](figures/trees-threshold.svg)

In the figure, all four animals at or below 9.2 lb are cats. Only 1 of the 6 heavier animals is a cat. That gives a gain of 0.61. The weights are invented for this illustration only.

## Overfitting and pruning

A deep tree can carve out a separate leaf for every odd training example, including noise. Training accuracy keeps rising while accuracy on new data stalls or falls (see [Bias & variance](#/t/biasvar)).

![On a noisy toy dataset, training accuracy climbs towards 100% as max_depth grows, while validation accuracy peaks early and then drifts down.](figures/trees-overfit.svg)

In the figure, look at the widening gap between the gold and teal lines after depth 2–4. That gap is overfitting.

Two ways to control it:

- **Pre-pruning** (stop early): set `max_depth`, `min_samples_leaf`, `min_samples_split` or `min_impurity_decrease`, and tune them on a validation set.
- **Post-pruning** (grow, then cut back): grow a full tree, then remove branches that don't help on validation data. sklearn's version is **cost-complexity pruning** (`ccp_alpha`). A larger `ccp_alpha` removes more branches. `cost_complexity_pruning_path` lists candidate values to try.

> **Trap:** One deep tree is a high-variance model: a few changed rows can give a completely different tree. Ensembles (random forests, [XGBoost](#/t/xgboost)) are the usual fix.

## Code: the cat table in sklearn

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# the 10 animals, encoded as 1/0 (1 = pointy, round, whiskers present, cat)
df = pd.DataFrame({
    "pointy":   [1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    "round":    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    "whiskers": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    "cat":      [1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
})
X = df[["pointy", "round", "whiskers"]]   # features
y = df["cat"]                             # label

# sklearn's default criterion is 'gini'; set 'entropy' to match these notes
tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
tree.fit(X, y)                            # grows the tree greedily, as worked above

print(export_text(tree, feature_names=list(X.columns)))   # the tree as text
plot_tree(tree, feature_names=X.columns, class_names=["not cat", "cat"], filled=True)
plt.show()                                # the same tree as a picture
```

Output of `export_text`. Since the features are 0/1, "pointy > 0.50" means pointy:

```text
|--- pointy <= 0.50
|   |--- whiskers <= 0.50
|   |   |--- class: 0
|   |--- whiskers >  0.50
|   |   |--- class: 1
|--- pointy >  0.50
|   |--- round <= 0.50
|   |   |--- class: 0
|   |--- round >  0.50
|   |   |--- class: 1
```

This is the same tree we built by hand.

## Feature importance

sklearn's `feature_importances_` adds up, for each feature, the impurity decrease of every split that uses it. Each decrease is weighted by how many training examples reach that node. Then everything is scaled so the importances sum to 1.

On the cat table, this gives ear shape 0.278, face shape 0.361 and whiskers 0.361. The root feature is not the most "important" here: the two second-level splits each removed more total entropy (5 animals × 0.72) than the root split did (10 animals × 0.28). Importance tells you how much each feature reduced impurity overall, not where it sits in the tree.

> **Trap:** Impurity-based importance is calculated on the training data. It tends to favour features with many distinct values. For a more reliable picture, check with permutation importance on validation data (`sklearn.inspection.permutation_importance`).

## In my accidents project

In [US Accidents](#/p/acc) I wrapped sklearn's tree in a small class with train, validation and test scoring and a precision–recall curve ([src/decision_tree.py](https://github.com/jag5m33/py_acc/blob/main/src/decision_tree.py)):

```python
self.model = DecisionTreeClassifier(max_depth = max_depth, # how many splits from root to leaf before making a decision; number of leaves double per level
                               min_samples_leaf= min_samples_leaf, # minimum samples per leaf; with a dataset of 2 million, 500 is adequate to make aconviced decision of equal disitribution
                               min_samples_split  = min_samples_split,
                               random_state= 42,
                               max_features= max_features,
                               class_weight= class_weight)
```

What the settings did:

- **`max_depth=10`**: at most 10 questions per path, so at most $2^{10} = 1024$ leaves. That is far fewer than 1.98 million training rows, so the tree can't memorise them.
- **`min_samples_leaf=500`**: every leaf must hold at least 500 accidents. Each leaf's "severe" probability is then an average over many rows, not a handful.
- **Result:** the train and validation scores were within 0.3 points of each other, so there was no sign of overfitting. On validation: precision 0.783, recall 0.689, F1 0.733, PR-AUC 0.822 (baseline 0.340).
- That tree used sklearn's default criterion (Gini), not entropy. The splitting logic is the same; only the impurity measure differs.
- **Feature importance:** road type carried **64%** of the importance, then location (longitude and latitude, 17% together) and the engineered `constraint_count` (7%). This matched the EDA, where severe rates ran from 14.7% on local roads to 80.4% on interstates. The weather flags got almost no importance, even though they had clear effects in the EDA. A likely reason is the trap above: the flags are rare, and road type takes most of the splits.

## Assumptions & caveats

- **No distribution assumptions.** Trees don't assume linearity, normality or equal variance. They only compare "≤ threshold".
- **No scaling needed.** Splits depend on the order of values, not their size, so standardising a feature doesn't change the tree.
- **High variance / instability.** Small changes in the training data can change the root split, and with it the whole tree.
- **Greedy, not optimal.** Each split is the best one locally. The tree never looks ahead, so it can miss a better overall tree (for example XOR-style patterns, where no single first split looks useful).
- **Biased toward many-valued features.** A feature with many distinct values (an ID, a high-cardinality category, a fine-grained number) has more possible thresholds, so it is more likely to find a split with high gain just by chance.
- **Axis-aligned splits.** Each question looks at one feature. A diagonal boundary, such as "y > x", has to be approximated by a staircase of many splits.
- **Class imbalance.** Leaves predict the majority class, so a rare class can be swamped. Use `class_weight="balanced"`, choose a threshold from the PR curve, and judge the model with precision/recall rather than accuracy ([Classification metrics](#/t/clsmetrics)).
- **Leakage still matters.** A tree will happily split on a leaked feature and look brilliant. In the accidents project, end time and duration were dropped for exactly this reason.

## Use cases

- You need a model you can **explain** to non-specialists ("if interstate and night, then high risk").
- Tabular data with a **mix of numeric and categorical** features, where the relationships are non-linear and features interact.
- A quick **baseline**, and a way to see which features matter before reaching for ensembles.
- Rules-based settings: triage, credit decisions and eligibility checks, where a flowchart is the natural form.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Easy to read and draw; each prediction is a path you can explain | High variance: unstable to small data changes |
| No scaling, no linearity or normality assumptions | Overfits easily without depth or leaf limits |
| Captures non-linear effects and interactions automatically | Greedy splitting may miss the best tree |
| Handles numeric and (encoded) categorical features | Axis-aligned splits approximate diagonal boundaries badly |
| Fast to train and predict | Usually less accurate than random forests or boosting |
| Gives feature importance as a by-product | Importance biased toward high-cardinality features |

## Limitations

- **Step-shaped predictions.** Outputs are constant inside each leaf, so a tree can't produce smooth probability or regression curves.
- **No extrapolation.** For regression, a tree can't predict outside the range of training targets. For classification, it treats values beyond the training range like the most extreme values it saw.
- **Rarely state of the art alone.** Its main role in practice is as the building block of random forests and gradient boosting.
- **Interpretability shrinks with depth.** A depth-10 tree with up to 1024 leaves, like mine, is no longer something you read by eye.

## Where trees go next

The same tree, used many times, in three steps:

1. [Bagging](#/t/bagging): grow many deep trees, each on a bootstrap sample of the rows (drawn with replacement), and average their votes. This lowers variance.
2. [Random forest](#/t/randomforest): bagging plus a random subset of features at every split, so the trees are less alike and the average improves further.
3. [XGBoost](#/t/xgboost): grow small trees one after another, each fitting the errors the previous trees left. This lowers bias.

## Quick reference

| Term | Meaning / formula |
|---|---|
| Root / decision / leaf node | First question / later question / final prediction |
| Pure node | All examples in it have one class (H = 0) |
| Entropy | $H(p_1) = -p_1\log_2 p_1 - (1-p_1)\log_2(1-p_1)$, with $0\log_2 0 = 0$ |
| Entropy range | 0 (pure) to 1 (50/50, binary case) |
| Information gain | $H(\text{root}) - (w_L H_L + w_R H_R)$ |
| Cat example gains at root | ear 0.278, whiskers 0.125, face 0.035 |
| Greedy | Pick the highest-gain split at each node, never look ahead |
| Continuous feature | Try midpoints between sorted values, keep the best gain |
| Pre-pruning | `max_depth`, `min_samples_leaf`, `min_samples_split`, `min_impurity_decrease` |
| Post-pruning | `ccp_alpha` (cost-complexity pruning) |
| sklearn default | `criterion='gini'`; use `criterion='entropy'` to match these notes |
