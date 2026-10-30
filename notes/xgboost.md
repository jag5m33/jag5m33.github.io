# XGBoost

XGBoost builds many small decision trees one after another, each one focusing on the examples the previous trees got wrong, and adds their answers together to make a prediction.

> **What you need to know**
> - starts from the bagged-tree idea: train each tree on a new subset of the training set
> - the change: instead of every example having an equal $1/m$ chance of being picked, examples that **earlier trees misclassified** are made more likely to be picked
> - so the trees are built **one after another**, each fixing the last ones' mistakes (bagging and random forest build them independently)
> - it mainly reduces **bias**; bagging and random forest mainly reduce **variance**
>
> **Skip:** the maths XGBoost uses to score splits and leaves; `fit()` does it

## The idea

This is Andrew Ng's way of explaining boosting, from my notes:

1. Given a training set of $m$ examples, train a decision tree on a new subset, **as in bagging**.
2. "XGBoost changes something!" Instead of picking from all examples with an equal $1/m$ chance, **make it more likely to pick previously misclassified samples** (from the previous trees).
3. Repeat: each new tree has a higher chance of training on the examples the forest so far still gets wrong.

My red note: "XGBoost – based off bagged decision tree." Ng calls this **deliberate practice**: practise the bars of a piano piece you keep getting wrong.

> **Check:** this is the right intuition, and here is how XGBoost actually gets the same effect. It doesn't literally resample with higher probabilities. Instead, each new tree is trained on the **errors (residuals)** the previous trees left behind, where residual = actual − predicted. The examples the model gets most wrong have the biggest residuals, so they have the biggest pull on the next tree. And unlike bagging, the trees are built **one after another**, never independently: tree 2 needs tree 1's mistakes.

## Bagging vs boosting in one picture

![Left: bagging builds independent trees on bootstrap samples and lets them vote. Right: boosting builds trees in sequence, and each one fits the mistakes left by the trees before it.](figures/xgboost-bagboost.svg)

In the figure, the arrows on the right carry the mistakes forward from tree to tree. That is the whole difference: bagging averages independent trees; boosting chains dependent ones.

![The cat example from the trees chapter: each bar is one animal's error. As trees are added, the bars shrink towards 0.](figures/xgboost-residuals.svg)

In the figure, each bar is how wrong the model still is about one of the 10 animals from [Decision trees](#/t/trees). Teal bars are cats that still need a higher probability; wine bars are non-cats that still need a lower one. Each new tree fixes part of what is left, so the bars shrink.

XGBoost ("eXtreme Gradient Boosting") picks its splits the same way a decision tree does, "which split makes the groups purer?", but it scores purity with its own measure instead of entropy.

## Tree vs bagging vs random forest vs XGBoost

| | Decision tree | Bagging | Random forest | XGBoost |
|---|---|---|---|---|
| Rows each tree sees | all | bootstrap sample | bootstrap sample | all, but focused on the mistakes so far |
| Features per split | all | all | random subset | all (by default) |
| Trees built | one | independently | independently | one after another |
| Combine | – | vote | vote | add up the trees' answers |
| Mainly reduces | – | variance | variance | bias |
| Typical trees | one, tuned depth | deep | deep | small (depth 3–6) |

## Key settings in plain words

| Setting | Plain meaning | If overfitting |
|---|---|---|
| `n_estimators` | how many trees in the chain | fewer, or use early stopping |
| `learning_rate` | how big a step each tree takes (default 0.3) | lower it and add trees |
| `max_depth` | how deep each small tree is (default 6) | lower (3–6 is typical) |
| `early_stopping_rounds` | stop adding trees once validation stops improving | use it |

## Code

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1500, random_state=0)   # toy data
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0)

model = xgb.XGBClassifier(
    n_estimators=600,            # upper limit on trees
    learning_rate=0.1,           # small steps
    max_depth=4,                 # small trees
    early_stopping_rounds=50,    # stop when validation stops improving
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print("best round:", model.best_iteration)
```

## In my accidents project

In [US Accidents](#/p/acc) I trained XGBoost with default settings, using the same interface as my decision tree class ([src/xgboost.py](https://github.com/jag5m33/py_acc/blob/main/src/xgboost.py)). Validation results, same split and the same 0.5 threshold:

| Model | Precision | Recall | F1 | PR-AUC | Accuracy |
|---|---|---|---|---|---|
| Baseline | – | – | – | 0.340 | 0.660 |
| Decision tree (depth 10, min leaf 500) | 0.783 | 0.689 | 0.733 | 0.822 | 0.830 |
| XGBoost (default settings) | 0.821 | 0.763 | 0.791 | 0.871 | 0.863 |

- With **no tuning at all**, XGBoost beat the tree on every metric. The biggest gain was recall (+7.4 points): more severe accidents caught.
- Train and validation scores were within 0.3 points, so no sign of overfitting.

## Assumptions & caveats

- **No scaling needed.** Splits only use the order of values.
- **Overfits** with too many or too-deep trees, and with noisy labels it can end up chasing noise. Use early stopping.
- **Class imbalance:** use `scale_pos_weight` (extra weight on the rare class) and judge with precision/recall ([Classification metrics](#/t/clsmetrics)).
- **Leakage:** be suspicious of amazing scores.

## Use cases

- **Tabular data** (rows and columns): risk scoring, churn, fraud, demand forecasting.
- A strong benchmark to beat before trying anything more complex.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Often the most accurate model on tabular data | More settings to tune than a forest |
| Handles missing values itself | Overfits if trees or depth aren't controlled |
| No scaling; captures interactions | Hard to interpret (hundreds of trees) |
| Early stopping built in | Trees must be built one after another (slower to train) |

## Interview one-liners

- **Boosting in one line?** Small trees built one after another, each focusing on the mistakes of the trees before it.
- **Bagging vs boosting?** Bagging: independent trees, vote, lowers variance. Boosting: trees in a chain, lowers bias.
- **How does it "focus" on mistakes?** Each new tree is trained on the previous trees' errors (residuals), so the worst-predicted examples pull hardest.
- **How do you stop it overfitting?** Lower learning rate, shallower trees, early stopping on a validation set.

## Quick reference

| Term | Meaning |
|---|---|
| Idea (Ng) | like bagging, but previously misclassified samples are more likely to be picked |
| In practice | each new tree fits the previous trees' errors (residuals) |
| Trees built | one after another, not independently |
| Reduces | bias (bagging / random forest: variance) |
| Key settings | `n_estimators`, `learning_rate`, `max_depth`, `early_stopping_rounds` |
| My accidents result | F1 0.791 vs tree 0.733, PR-AUC 0.871 vs 0.822 |
