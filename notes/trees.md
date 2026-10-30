# Decision trees

A decision tree classifies an example by asking a short chain of yes/no questions about its features, choosing each question so the groups it creates are as pure as possible.

> **What you need to know**
> - the feature at the root (and at every node) is the one that **maximises purity**, measured by **information gain**
> - **entropy** $H(S)$ = how impure a node is: 0 when pure, 1 at a 50/50 mix
> - **information gain** = entropy before the split − weighted average entropy after it; highest gain wins
> - stop splitting when a node is 100% one class, at max depth, or when the gain is too small
> - many trees voting = an **ensemble** ([Bagging](#/t/bagging), [Random forest](#/t/randomforest), [XGBoost](#/t/xgboost))
>
> **Skip:** Gini's formula (another impurity measure; same idea); doing big trees by hand: `fit()` does it

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

1. **Which feature should this node split on?** As my notes put it: "the feature at the root node is selected to **maximise purity**". In practice: the one with the highest information gain.
2. **When should we stop splitting?** Three rules:
   - the node is **100% one class** (pure, nothing left to separate);
   - splitting would **exceed the maximum depth** (this reduces overfitting);
   - the **improvement in purity is below a threshold** (the split isn't worth it).

## Purity

**Purity** describes how mixed the classes are inside a node.

- **Pure**: every example in the node has the same class (all cats, or no cats).
- **Impure**: the node holds a mix of both classes.

Every split tries to make the child nodes purer than their parent. A pure node needs no more questions.

![Three nodes with 5 animals each. The two outer nodes are pure, the middle one is mixed.](figures/trees-purity.svg)

In the figure, teal circles are cats and wine circles are not cats. The middle node is mixed, so its entropy is high (0.97). The outer nodes are pure, so their entropy is 0.

## Entropy: a number for impurity

**Entropy** is "a measure of the impurity of a dataset" (a node's examples).

$$
H(S) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)
$$

- $S$ = the set of examples in the node.
- $p_1$ = fraction of examples in the node that are class 1 (cats).
- $p_0 = 1 - p_1$ = fraction that are class 0 (not cats).
- $\log_2$ = log base 2 (see [Logs & exponents](#/t/logs)).
- By convention $0 \times \log_2 0 = 0$, so a pure node has entropy 0.

In words: entropy is 0 when the node is pure and 1 when it is a 50/50 mix. In my notes: "if a split results in 50% one class & 50% other, P(x) = 50/100, H(S) = 1 → high impurity". And: "the lower/higher the probability of a class in a set after a split, the lower H(S)", i.e. the further $p_1$ is from 0.5, the purer the node.

![The entropy curve. It is 0 at both ends (pure nodes) and peaks at 1 when the node is half cats.](figures/trees-entropy.svg)

In the figure, the curve is symmetric. 4 cats out of 5 ($p_1 = 0.8$) and 1 cat out of 5 ($p_1 = 0.2$) both have entropy 0.72. Entropy measures how mixed a node is, not which class is winning.

**Cat root:** 10 animals, 5 cats, so $p_1 = p_0 = 0.5$ and $H = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 0.5 + 0.5 = 1$.

> **Say it like this:** "The root is a 50/50 mix, so its entropy is 1, the maximum. Every split should push that number down."

## Information gain

**Information gain** = entropy before the split minus the weighted average entropy after it:

$$
\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \, H(S_v)
$$

- $A$ = the feature we split on; $v$ = each of its values (each branch).
- $|S|$ = total samples before the split; $|S_v|$ = samples in branch $v$ after the split.
- $H(S_v)$ = entropy of branch $v$ after the split.
- With two branches this is just $H(S) - (w_{\text{left}} H_{\text{left}} + w_{\text{right}} H_{\text{right}})$, where $w = |S_v|/|S|$.

"The higher the info gain, the more favoured that split of the decision tree."

**Why weight by size?** A tiny pure branch shouldn't count as much as a big mixed one. Sending 1 animal left (pure) and 9 right (still mixed) has barely helped.

### Worked example 1: my 9 Yes / 5 No split

From my notebook: node $f_1$ has 14 samples, 9 Yes and 5 No. It splits into $f_2$ (6 Yes, 2 No) and $f_3$ (3 Yes, 3 No).

![My split drawn as a tree. The parent is 9 Yes / 5 No; the left child (6/2) is purer, the right child (3/3) is a 50/50 mix.](figures/trees-hissplit.svg)

In the figure, the left child is mostly Yes, so its entropy falls; the right child is 50/50, so its entropy is the maximum, 1.

| Step | Calculation | Value |
|---|---|---|
| Parent $H(f_1)$ | $-\tfrac{9}{14}\log_2\tfrac{9}{14} - \tfrac{5}{14}\log_2\tfrac{5}{14}$ | 0.94 |
| Left $H(f_2)$ | $-\tfrac{6}{8}\log_2\tfrac{6}{8} - \tfrac{2}{8}\log_2\tfrac{2}{8}$ | 0.81 |
| Right $H(f_3)$ | 3 Yes, 3 No: a 50/50 mix | 1 |
| Weighted after | $\tfrac{8}{14} \times 0.81 + \tfrac{6}{14} \times 1$ | 0.89 |
| **Gain** | $0.94 - 0.89$ | **0.049** |

- $f_2$: "higher prob. = lower entropy = higher purity".
- A small gain: the split helps only a little, because $f_3$ is still completely mixed.

> **Check:** with unrounded entropies (0.9403, 0.8113, 1) the gain is **0.048**. My 0.049 comes from rounding the H values to 0.94 and 0.81 first. Both are fine; just round at the end when you can.

### Worked example 2: cat example, all three features at the root

$H(\text{root}) = 1.0$ for every candidate.

![The three candidate root splits, with the cats and non-cats in each branch and the entropy of each branch.](figures/trees-rootsplits.svg)

| Feature | Left branch (cats / animals, H) | Right branch (cats / animals, H) | Weighted H after | Gain = 1 − weighted |
|---|---|---|---|---|
| Ear shape | pointy: 4/5, 0.722 | floppy: 1/5, 0.722 | 0.5 × 0.722 + 0.5 × 0.722 = 0.722 | **0.278** |
| Face shape | round: 4/7, 0.985 | not round: 1/3, 0.918 | 0.7 × 0.985 + 0.3 × 0.918 = 0.965 | 0.035 |
| Whiskers | present: 3/4, 0.811 | absent: 2/6, 0.918 | 0.4 × 0.811 + 0.6 × 0.918 = 0.875 | 0.125 |

Ear shape wins: both its branches are 80/20, the most lopsided split. Face shape is almost useless at the root (round is 57% cats, nearly a coin flip).

The tree is **greedy**: at each node it takes the biggest gain right now. So **ear shape becomes the root**.

> **Key idea:** information gain ranks the features at each node, and the highest gain wins. The ranking is recomputed at every node, using only the examples that reached it.

## One level down

Repeat the same steps on each branch, using only the animals that reached it. Under pointy ears, face shape splits perfectly (gain 0.72); under floppy ears, whiskers splits perfectly (gain 0.72). Every leaf is now pure, so the tree stops (stopping rule 1).

![The finished tree: ear shape at the root, face shape under pointy, whiskers under floppy. All four leaves are pure.](figures/trees-final.svg)

In the figure, each node shows how many animals reach it and its entropy. Entropy goes from 1.00 at the root, to 0.72, to 0 in every leaf.

## Numbers, overfitting and stopping

**Number features** (like weight): the tree tries questions such as "weight ≤ 9.2?" at the midpoint between each pair of sorted values, and keeps the one with the highest gain.

**Overfitting**: if you never stop, the tree keeps splitting until each leaf holds one example. It has memorised the training data, noise included (see [Bias & variance](#/t/biasvar)).

![On a noisy toy dataset, training accuracy climbs towards 100% as max_depth grows, while validation accuracy peaks early and then drifts down.](figures/trees-overfit.svg)

In the figure, look at the widening gap between the gold and teal lines after depth 2–4. That gap is overfitting.

The fix is my three stopping rules, as sklearn settings tuned on a validation set:

| My stopping rule | sklearn setting |
|---|---|
| Node is 100% one class | always on (a pure node is never split) |
| Would exceed max depth | `max_depth` |
| Purity improvement < a threshold | `min_impurity_decrease` |
| (also common) too few samples | `min_samples_leaf`, `min_samples_split` |

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

The printout splits on `pointy` first, then `round` under pointy and `whiskers` under floppy: the tree we built by hand.

## In my accidents project

In [US Accidents](#/p/acc) I used sklearn's tree with my stopping rules set by hand ([src/decision_tree.py](https://github.com/jag5m33/py_acc/blob/main/src/decision_tree.py)):

- **`max_depth=10`**: at most 10 questions per path, so at most 1,024 leaves, far fewer than 1.98 million training rows. The tree can't memorise them.
- **`min_samples_leaf=500`**: every leaf holds at least 500 accidents, so its "severe" probability is an average over many rows.
- **Result:** train and validation scores were within 0.3 points, so no sign of overfitting. On validation: precision 0.783, recall 0.689, F1 0.733, PR-AUC 0.822 (baseline 0.340).
- **Feature importance:** road type carried **64%**, then location (17%) and the engineered `constraint_count` (7%). This matched the EDA, where severe rates ran from 14.7% on local roads to 80.4% on interstates.

## Assumptions & caveats

- **No scaling needed**, and no linearity or normality assumptions.
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

- Predictions are step-shaped and can't go outside the training range.
- A depth-10 tree with up to 1,024 leaves, like mine, is no longer readable by eye.

## Ensemble trees: many trees vote

From my notes: build **multiple decision trees with different start (root) node features**. Each tree then takes different paths, with its nodes in a different order. **Run one sample through all the trees and let them vote**: the majority is the classification.

1. [Bagging](#/t/bagging): each tree trains on a random resample of the rows (sampling with replacement), then they vote.
2. [Random forest](#/t/randomforest): bagging plus random features at every split, so the roots (and every later node) really do differ.
3. [XGBoost](#/t/xgboost): small trees built one after another, each fixing the previous trees' errors.

## Interview one-liners

- **How does a tree pick a split?** It tries each feature and keeps the one with the highest information gain (biggest drop in entropy).
- **What is entropy?** A measure of impurity: 0 for a pure node, 1 for a 50/50 node.
- **When does it stop?** Pure node, max depth reached, or the gain is below a threshold.
- **Why do trees overfit?** Grown fully, they memorise every training row; limit depth or leaf size.
- **Why ensembles?** One tree is unstable; many different trees voting are more accurate.

## Quick reference

| Term | Meaning / formula |
|---|---|
| Root / decision / leaf node | First question / later question / final prediction |
| Pure node | All one class (entropy 0) |
| Entropy | $H(S) = -p_1\log_2 p_1 - p_0\log_2 p_0$ |
| Entropy range | 0 (pure) to 1 (50/50) |
| Information gain | $H(S) - \sum (\lvert S_v \rvert / \lvert S \rvert) H(S_v)$ |
| My 9Y/5N example | 0.94 − (8/14 × 0.81 + 6/14 × 1) = 0.048 (0.049 rounded early) |
| Cat example gains at root | ear 0.278, whiskers 0.125, face 0.035 |
| Stop splitting | pure node, max depth, gain below threshold |
| Ensemble | many trees, different roots, vote |
| Greedy | take the biggest gain at each node |
