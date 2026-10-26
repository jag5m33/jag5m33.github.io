# Choosing the right metric

A metric is one number that says how good a model's predictions are; which one to use depends on the kind of problem and on which mistakes hurt most, much more than on which model you trained.

## Why we need metrics

> **What you need to know**
> - a model always gives *some* answer; a **metric** tells you how good those answers are
> - you need one to **compare** models (tree vs XGBoost), **tune** settings (`max_depth`, `alpha`) and **report** results to other people
> - different metrics reward different things, so the choice is a decision, not a default
>
> **Skip:** memorising every metric in sklearn; the ten or so in these chapters cover almost every junior interview

Let's say two spam filters both score 95%. One lets through a few spam emails; the other deletes your boss's email. Same number, very different problems. A good metric is one that matches **the mistake you care about**.

## Where you measure it: train, validation, test

> **What you need to know**
> - fit on **train**, choose models and settings on **validation**, report once on **test** ([Supervised learning](#/t/supintro))
> - a big gap between train and validation scores means overfitting ([Bias & variance](#/t/biasvar))

A score on the training data flatters the model, because it has already seen the answers.

## The chooser

![A visual chooser. First ask whether you have true labels. Regression leads to MAE, RMSE or R²; classification to accuracy, recall, precision, PR-AUC or macro F1 depending on the situation; no labels leads to silhouette and elbow for clusters, and stability checks or precision@k for anomalies.](figures/metricsintro-chooser.svg)

In the figure, start at the top question, pick the column for your problem, then read down to the line that matches your situation.

| Problem | Situation | Lead with | Chapter |
|---|---|---|---|
| Regression | typical miss in real units | MAE or RMSE | [Regression metrics](#/t/regmetrics) |
| Regression | outliers are noise, every £ costs the same | **MAE** | |
| Regression | big misses are much worse than small ones | **RMSE** | |
| Regression | "how much of the variation does it explain?" | **R²** | |
| Classification | classes roughly balanced | **accuracy** is fine (plus F1) | [Classification metrics](#/t/clsmetrics) |
| Classification | imbalanced classes | **precision, recall, F1, PR-AUC** | |
| Classification | missing a positive is costly (cancer, fraud) | **recall** | |
| Classification | false alarms are costly (spam filter) | **precision** | |
| Classification | many classes, all matter | **macro F1** | |
| Classification | comparing models' ranking, any threshold | **PR-AUC** (rare positives) or **ROC-AUC** (balanced) | |
| Clustering | no labels | **silhouette**, **elbow** (inertia) | [Clustering & anomaly metrics](#/t/unsupmetrics) |
| Anomaly detection | no labels | stability and common sense; **precision@k** if a few are labelled | |

> **Say it like this:** "I pick the metric from the problem and the cost of each mistake, then compare every model on that same metric on the validation set."

## Model → what it outputs → metrics that fit

> **What you need to know**
> - most metrics depend on the **problem**, not the model: any classifier can be scored with precision and recall, any regressor with RMSE
> - the exception: **ROC-AUC, PR-AUC and log loss need a score or probability** for each row, not just a class
> - so check what your model outputs (`predict` gives a class or number; `predict_proba` gives probabilities)

| Model | What it outputs | Metrics that fit |
|---|---|---|
| [Linear regression](#/t/linreg) | a number | MAE, RMSE, R² |
| [Logistic regression](#/t/logreg) | a probability (0 to 1), then a class by threshold | precision, recall, F1, PR-AUC, ROC-AUC, log loss |
| [Softmax regression](#/t/softmax) | one probability per class | accuracy, macro F1, log loss |
| [Naive Bayes](#/t/naivebayes) | a class, plus probabilities that are too extreme | precision, recall, F1; AUCs are fine (they only need the ranking); don't trust its log loss |
| [Decision tree](#/t/trees) | a class or a number; probabilities are just the leaf's class shares | precision, recall, F1 (classifier); MAE, RMSE, R² (regressor) |
| [Random forest](#/t/randomforest) / [bagging](#/t/bagging) | averaged votes, so smoother probabilities | all classification metrics incl. AUCs; or MAE, RMSE, R² |
| [XGBoost](#/t/xgboost) | probabilities (or numbers) | PR-AUC, F1, log loss; or MAE, RMSE, R² |
| [K-means](#/t/kmeans) | a cluster number for each point | inertia + elbow, silhouette |
| [Anomaly detection](#/t/anomaly) | an "unusualness" score per point | stability checks, precision@k if a few labels exist |

**Skip:** probability calibration (making predicted probabilities match real rates); not needed at junior level.

## Always compare against a baseline

> **What you need to know**
> - a **baseline** is the score of a "model" that doesn't learn anything
> - regression: always predict the **mean** (this is exactly what R² = 0 means)
> - classification: always predict the **majority class**; for PR-AUC the baseline is the **share of positives**
> - a score only means something **next to its baseline**

**In context: the US accidents project.** About 34% of accidents in the data are severe. A "model" that always says "not severe" is right 66% of the time:

| | Accuracy | PR-AUC |
|---|---|---|
| Baseline (always "not severe") | 0.66 | 0.34 |
| Decision tree | 0.83 | 0.822 |
| XGBoost | 0.863 | 0.871 |

The tree's 0.83 accuracy sounds good, but it is only 17 points above doing nothing. Its PR-AUC of 0.822 against a baseline of 0.34 is the much clearer win. That's why the [accidents project](#/p/acc) leads with precision, recall, F1 and PR-AUC, and reports accuracy for reference only.

> **Trap:** with 1,000 transactions and 5 frauds, "never fraud" scores 99.5% accuracy and catches nothing. High accuracy on imbalanced data means nothing on its own.

## Interview one-liners

- **Which metric would you use?** "It depends on the problem and which mistake costs more: recall if a miss is costly, precision if false alarms are, macro F1 if many classes all matter."
- **Why not accuracy?** "With imbalanced classes, predicting the majority class already scores high, so accuracy hides a useless model."
- **RMSE or MAE?** "RMSE if big misses are much worse; MAE if outliers are noise and every unit of error costs the same."
- **What does R² = 0 mean?** "No better than always predicting the mean."
- **Which metrics need probabilities?** "ROC-AUC, PR-AUC and log loss. Precision, recall and F1 only need the predicted class."
- **How do you evaluate clustering?** "There are no labels, so silhouette and the elbow of inertia, plus whether the groups make sense."
- **First thing you check?** "The baseline: the mean, or the majority class."

## Quick reference

| Term | Meaning |
|---|---|
| Metric | one number scoring a model's predictions |
| Validation set | where you compare models and choose settings |
| Test set | scored once, at the end |
| Baseline | score of a model that learns nothing (mean / majority class / positive rate) |
| Needs probabilities | ROC-AUC, PR-AUC, log loss |
| Needs only classes | accuracy, precision, recall, F1 |
| Regression | MAE, RMSE, R² |
| Clustering | inertia + elbow, silhouette |
