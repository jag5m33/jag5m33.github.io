# Bagging & bootstrap sampling

Bagging trains many copies of the same model on random resamples of the training rows and combines their votes, which smooths out the wobbles that make a single deep decision tree unreliable.

## Terms used below

This chapter builds on [Decision trees](#/t/trees) and reuses the same 10 animals (ear shape, face shape, whiskers, cat or not).

- **Ensemble**: one model made by combining many models.
- **Variance** (of a model): how much its predictions change if you train it on a slightly different sample. High variance = unstable. See [Bias & variance](#/t/biasvar).
- **Bagging**: short for **b**ootstrap **agg**regat**ing**. "Bootstrap" = resample the rows; "aggregating" = combine the models.

## Why combine many models?

Ask one person to guess the number of sweets in a jar and they may be way off. Average the guesses of 100 people and the answer is often surprisingly close: some guess too high, some too low, and the errors cancel.

**The catch:** this only works if people make **different** mistakes. If all 100 copied the same wrong guess, the average would be just as wrong.

A single deep decision tree is like one wild guesser: it fits its training data very closely, so a few changed rows can give a completely different tree. Bagging gives every tree a slightly different version of the data, so each tree makes different mistakes, and the vote cancels many of them out.

> **Key idea:** averaging many different trees smooths out their mistakes, but only if the trees really are different.

## Bootstrap sampling

A **bootstrap sample** is a new dataset made by drawing rows from the original **at random, with replacement**, until it has the same number of rows as the original.

- **With replacement**: a row goes back into the pot after it is drawn, so it can be drawn again. The same row can appear 2, 3 or more times.
- Because some rows repeat, other rows are **never drawn**. Those left-out rows are the **out-of-bag (OOB)** rows for that sample.

![Ten numbered rows (teal = cat, wine = not cat) and three bootstrap samples drawn from them with a seeded random generator. Gold outlines mark rows that were drawn more than once; the right-hand column lists the rows each sample missed.](figures/bagging-bootstrap.svg)

In the figure, look at sample 1: rows 3, 5 and 9 each appear twice, so rows 6, 8 and 10 never get in. Every sample still has 10 rows, but only 7 or 8 **different** ones. Each sample is a slightly different version of the same data, and that is what makes the trees differ.

### About 63% unique rows

On average, a bootstrap sample contains only about **63%** of the original rows; the other **37%** are left out. With 1,000 rows, expect about 632 different rows in each sample. (The chance a given row is never picked is $(1 - 1/n)^n$, which settles at about 0.37 for any decent-sized dataset.)

> **Say it like this:** "Each bootstrap sample is the same size as the data, but because rows can repeat, it only contains about 63% of the distinct rows. The other 37% are out-of-bag."

## Out-of-bag rows: a free validation set

Every tree has its own OOB rows: rows it never saw while training. So you can test each row using only the trees that didn't train on it:

1. For each training row, find the trees that left it out (about a third of them).
2. Let only those trees vote on that row.
3. Compare the vote with the true label.
4. The accuracy over all rows is the **OOB score**.

It's fair because no row is judged by a tree that trained on it, and you didn't have to set any data aside.

> **Trap:** the OOB score replaces a validation split, not a final untouched test set.

## Bagged decision trees

### The recipe

1. Choose how many trees to grow (`n_estimators`).
2. For each tree: draw a bootstrap sample and grow **one deep tree** on it.
3. Combine:
   - **Classification**: majority vote (sklearn averages the predicted probabilities, which is nearly the same).
   - **Regression**: average the predicted numbers.

The trees are **independent**: tree 7 doesn't know tree 6 exists. So they can be trained at the same time, unlike [XGBoost](#/t/xgboost) where each tree depends on the one before.

### Small worked example: three stumps on the cat table

To keep it hand-sized, grow a **stump** (a tree with just one split) on each of the three bootstrap samples from the figure. A new animal arrives: **floppy ears, round face, whiskers present**.

| Tree | Bootstrap sample (row numbers) | Root chosen | Leaf the new animal lands in | P(cat) | Vote |
|---|---|---|---|---|---|
| 1 | 1, 2, 3, 3, 4, 5, 5, 7, 9, 9 | Whiskers | present: 4 cats of 5 | 0.80 | cat |
| 2 | 1, 2, 3, 3, 6, 7, 8, 9, 9, 10 | Ear shape | floppy: 1 cat of 7 | 0.14 | not cat |
| 3 | 2, 3, 4, 5, 5, 6, 7, 7, 8, 10 | Ear shape | floppy: 1 cat of 5 | 0.20 | not cat |

- **Majority vote:** 1 cat, 2 not cat, so the ensemble says **not cat**.
- **Averaged probability** (sklearn's way): $(0.80 + 0.14 + 0.20)/3 = 0.38$, below 0.5, so also **not cat**.

![The three stumps. Tree 1 happened to get a sample where whiskers split best; trees 2 and 3 chose ear shape. The outlined leaf is where the new animal lands in each tree.](figures/bagging-votes.svg)

In the figure, tree 1 asks a different question at its root: its sample had no whisker-less cats, so whiskers looked perfect. That is the bootstrap creating variety. But notice that two of the three trees still chose ear shape: bagging alone often produces trees that look alike. Keep that in mind for random forests.

(Numbers from sklearn's `DecisionTreeClassifier(max_depth=1, criterion="entropy")` on samples drawn with `np.random.default_rng(2)`.)

### What it does to the decision boundary

![Left: one fully grown tree on a noisy two-moons dataset, with a jagged boundary and small islands around single points. Right: 200 bagged trees, with a smoother boundary.](figures/bagging-boundary.svg)

In the figure, the single tree (left) draws tiny boxes around individual noisy points. Each bagged tree does the same, but around **different** points. Averaging 200 of them keeps the shape they agree on and washes out most of the islands. Test accuracy rose from 0.883 (one tree) to 0.925 (bagging). The OOB score was 0.889.

## What bagging fixes, and what it doesn't

- **Fixes: variance (instability).** Different trees, different mistakes, averaged away.
- **Doesn't fix: bias (too simple).** If every tree is too simple for the problem, the average of 500 of them is still too simple. That is boosting's job ([XGBoost](#/t/xgboost)).
- **The limit:** the samples overlap a lot, and if one feature is very strong, almost every tree picks it at the root. The trees end up alike, and averaging alike trees doesn't help much. [Random forest](#/t/randomforest) fixes this by also randomising the features.

## Code: bagged trees in sklearn

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=0)   # noisy toy data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

bag = BaggingClassifier(
    DecisionTreeClassifier(),   # base model: one deep tree
    n_estimators=200,           # number of trees
    oob_score=True,             # score rows with trees that didn't see them
    random_state=0,             # repeatable bootstrap draws
    n_jobs=-1,                  # train trees in parallel
)
bag.fit(X_tr, y_tr)
print("test acc:", round(bag.score(X_te, y_te), 3))   # 0.925
print("OOB     :", round(bag.oob_score_, 3))           # 0.889
```

## Assumptions & caveats

- **Needs an unstable base model.** Bagging helps deep trees. Bagging [linear regression](#/t/linreg) changes almost nothing, because every copy comes out nearly the same.
- **Doesn't fix bias.**
- **Rows treated as independent.** For time series or several rows per patient, random resampling mixes past and future (or the same patient across samples), and the OOB score looks too good.
- **Class imbalance.** A rare class can be almost missing from some samples. Use `class_weight` and judge with precision/recall ([Classification metrics](#/t/clsmetrics)).
- **No scaling needed** with trees.

## Use cases

- You have a model that fits well but is **unstable** (deep trees are the textbook case).
- You want an easy improvement over one tree, with a free validation score (OOB).
- Training can run in **parallel**, because the trees are independent.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Usually beats one deep tree clearly | Doesn't reduce bias |
| OOB score gives validation for free | Trees often too alike, so gains level off |
| Trees train in parallel | Loses the readability of a single tree |
| Few settings; more trees doesn't overfit | Many times the cost of one tree |

## Limitations

- You can't draw 200 trees as one flowchart.
- After a few dozen trees, adding more barely helps.
- In practice people usually go straight to [random forest](#/t/randomforest) or [XGBoost](#/t/xgboost).

## Quick reference

| Term | Meaning |
|---|---|
| Bagging | one model per bootstrap sample, then vote or average |
| Bootstrap sample | n rows drawn with replacement from n rows |
| Unique rows per sample | about 63% |
| OOB rows | the ~37% a given tree never saw |
| OOB score | each row judged only by trees that didn't train on it |
| Combine | majority vote (classification), average (regression) |
| Reduces | variance, not bias |
| sklearn | `BaggingClassifier(DecisionTreeClassifier(), n_estimators=..., oob_score=True)` |
