# Random forest

A random forest is a crowd of decision trees, each grown on a bootstrap sample of the rows and allowed to look at only a random handful of features at each split, so the trees disagree in useful ways and their vote is more accurate than any one of them.

> **What you need to know**
> - with bagging alone, "sometimes the same split of tree happens with the same root node", so the trees end up alike
> - random forest **randomises the feature choice**: with N features, pick a random subset of **k < N** and compute information gain only on those
> - **and** each tree trains on its own bootstrap sample of rows (sampling with replacement, as in [Bagging](#/t/bagging))
> - the trees then **vote**; common rule of thumb $k \approx \sqrt{N}$ (sklearn's default)
>
> **Skip:** tuning theory; `max_features` and `n_estimators` are the settings that matter

## The idea, from my notes

This chapter builds on [Decision trees](#/t/trees) and [Bagging](#/t/bagging). Same 10 animals again.

1. **The problem:** in bagged trees, "sometimes the same split of tree happens with the same root node". One strong feature wins the root again and again.
2. **The fix: randomise the feature choice.** If $N$ features are available, pick a **random subset of $k$ features**, $k < N$, and iteratively compute information gain on that feature choice only. The best of those $k$ wins the split.
3. **AND row subsamples:** each tree is trained on a random bootstrap sample (sampling with replacement).
4. Run a new example through every tree; the trees **vote**.

> **Building on my note:** the random subset of $k$ features is **redrawn at EVERY split**, not only at the root. Each decision node gets its own fresh draw. So even deep in the tree, nodes can't all lean on the same strong feature. The common rule of thumb is $k \approx \sqrt{N}$: with 16 features, try 4 at each split.

Everything else is the same tree as before: root node, decision nodes, leaves, splits chosen by information gain.

![One tree in a random forest, as an illustration. The root drew only "whiskers", so it had to split on whiskers. Each child then drew its own feature (ear shape on one side, face shape on the other).](figures/randomforest-everysplit.svg)

In the figure, each gold box shows which feature that node was allowed to consider. They differ from node to node, because the draw is repeated at every split.

## Seen on the cat table: same root again and again

On the full table, ear shape wins the root (gain 0.278, against 0.125 and 0.035 from the trees chapter). I grew 100 trees each way in Python:

- **Bagged trees**: bootstrap sample, all 3 features allowed at every split.
- **Random forest**: bootstrap sample, and only **one** random feature allowed at each split (`max_features=1`).

![Root feature of 100 trees. Bagged trees chose ear shape 59 times; the random forest spread its roots across all three features.](figures/randomforest-roots.svg)

| Root feature | Bagged trees | Random forest |
|---|---|---|
| Ear shape | **59** | 30 |
| Face shape | 16 | 44 |
| Whiskers | 25 | 26 |

In the figure, the gold bars lean heavily towards ear shape. With one random feature per split, each feature becomes the root roughly a third of the time (the dotted line). Face shape, the weakest root split, now starts 44 trees, and those trees are forced to grow a different shape. (Not exactly a third each, just by chance with only 100 trees.)

Each single tree gets a little worse (sometimes it has to use a weaker feature), but the forest gets better, because the trees' mistakes overlap less.

> **Say it like this:** "Bagging makes trees different by giving them different rows. Random forest also gives each split a random set of features to choose from, so even a dominant feature can't make every tree look the same. More different trees means their mistakes cancel out better when they vote."

## `max_features` = my $k$

| Setting | Features tried at each split ($k$) |
|---|---|
| `"sqrt"` | $\sqrt{N}$ (default for classification) |
| a number, e.g. `3` | exactly that many |
| `None` or `1.0` | all $N$: no feature randomness, so this is just bagging |

- With 3 cat features, $\sqrt{3} = 1.73$, rounded down to **1**, so the default on the cat table is exactly the forest above.
- **Smaller** → more different trees, but each one is weaker. **Larger** → stronger trees, more alike. It's the main setting to tune.
- **Trap for regression:** `RandomForestRegressor` uses all features by default, so out of the box it is really just bagging. Set `max_features="sqrt"` (or e.g. `0.33`) yourself.

## Making a prediction

Send the new example down **every** tree and let them vote (regression: average). On the cat table, the 100-tree forest gives the new animal from the bagging chapter (floppy ears, round face, whiskers) an average P(cat) of **0.62**, so it predicts **cat**. Like bagging, it also gets a free **OOB score** (`oob_score=True`).

## Code: the cat table

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],
              [1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])   # pointy, round, whiskers
y = np.array([1,1,0,0,1,1,0,1,0,0])                        # 1 = cat

rf = RandomForestClassifier(
    n_estimators=100,      # 100 trees
    max_features=1,        # ONE random feature per split
    criterion="entropy",   # information gain, as in the trees chapter
    random_state=0,
)
rf.fit(X, y)

roots = [t.tree_.feature[0] for t in rf.estimators_]   # root feature of each tree
print(np.bincount(roots, minlength=3))                  # [30 44 26]
print(rf.predict_proba([[0, 1, 1]])[0, 1])              # 0.62
```

## One tree vs bagging vs random forest

![Decision boundaries on the two-moons data. One deep tree draws jagged islands around noisy points; bagging and the random forest both draw smoother boundaries.](figures/randomforest-boundaries.svg)

In the figure, the single tree (left) draws small boxes around individual noisy points. Both ensembles keep the overall moon shape and drop most of those islands. Test accuracy: one tree 0.883, bagging 0.925, random forest 0.917. Here bagging edged ahead by one test example: with only 2 features, "one random feature per split" is blunt. Forests show their advantage more with many features. Always compare on your own validation data.

**Feature importance:** `feature_importances_` is free but can favour features with many distinct values (like IDs), so confirm it with permutation importance on validation data.

## Key settings (sklearn)

| Parameter | Default | What it does |
|---|---|---|
| `n_estimators` | 100 | Number of trees. More is better until it levels off; doesn't overfit |
| `max_features` | `"sqrt"` | Features tried at each split. The main dial |
| `max_depth` | `None` | Maximum depth of each tree. Usually left deep |
| `min_samples_leaf` | 1 | Minimum rows in a leaf. Larger smooths noisy data |
| `class_weight` | `None` | `"balanced"` for imbalanced classes |

For how random forest compares with a single tree, bagging and boosting, see the comparison table in [XGBoost](#/t/xgboost).

## Interview one-liners

- **Random forest in one line?** Bagged decision trees where each split may only use a random subset of the features; the trees vote.
- **Why random features?** Otherwise one strong feature wins the root in most trees, the trees look alike and make the same mistakes.
- **How many features per split?** $k < N$, usually about $\sqrt{N}$ for classification (`max_features="sqrt"`).
- **Does adding trees overfit?** No; more trees just levels off. Deep trees are fine because the vote smooths them.

## Assumptions & caveats

- **No scaling needed.** Splits only use the order of values.
- **Rows treated as independent.** With time series or several rows per customer, the OOB score looks too good; use time-based or grouped splits.
- **Class imbalance.** Use `class_weight`, choose the threshold from a PR curve, and report precision/recall ([Classification metrics](#/t/clsmetrics)).
- **Correlated features share importance**, so each can look less important than it is.
- **Leakage.** A forest exploits a leaked feature as happily as one tree.

## Use cases

- **Tabular data** with non-linear effects and interactions, when you want a strong model with little tuning.
- A **robust baseline** before trying boosting.
- Many features and not that many rows (genomics, sensor data).

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Usually much more accurate than one tree | Not readable as a single flowchart |
| Good results with default settings | Large memory (hundreds of deep trees) |
| Trees train in parallel; OOB score for free | Slower to predict than one tree |
| Adding trees doesn't overfit | Usually a little behind well-tuned boosting |
| No scaling needed | Can't predict beyond the training range |

## Limitations

- **Doesn't fix bias.** If each tree can't capture the pattern, averaging won't help; boosting attacks that.
- **Very sparse text data** (like TF-IDF with 200,000 columns) suits linear models and Naive Bayes better.

## Quick reference

| Term | Meaning |
|---|---|
| Random forest | same trees + row bootstrap + random k of N features at every split |
| Rule of thumb | $k \approx \sqrt{N}$ |
| Randomness 1 | bootstrap sample of rows per tree |
| Randomness 2 | `max_features` drawn afresh at **every** split |
| Why it works | trees are more different, so their mistakes cancel when they vote |
| Cat table roots (100 trees) | bagging 59/16/25, forest 30/44/26 (ear/face/whiskers) |
| `max_features` default | `"sqrt"` (classifier), all features (regressor) |
| Reduces | variance, not bias |
