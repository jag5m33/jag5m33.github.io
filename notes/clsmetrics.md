# Classification metrics

Classification metrics count the ways a classifier is right and wrong, and the right one depends on which mistakes cost more and how rare the positive class is.

## The confusion matrix

Everything starts from four counts. Call the class you care about **positive** (1), the other **negative** (0).

- **TP (true positive)**: predicted 1, actually 1.
- **FP (false positive)**: predicted 1, actually 0. A false alarm (Type I error).
- **FN (false negative)**: predicted 0, actually 1. A miss (Type II error).
- **TN (true negative)**: predicted 0, actually 0.

![A 2 by 2 confusion matrix with 40 true positives, 10 false negatives, 20 false positives and 130 true negatives.](figures/clsmetrics-confusion.svg)

In the figure the teal cells are correct and the wine cells are mistakes. We'll use these numbers, 200 rows in total, 50 actual positives, for every worked example below.

> **Trap:** sklearn's `confusion_matrix` puts **actual** on the rows and **predicted** on the columns, with class 0 first, so its layout is `[[TN, FP], [FN, TP]]`, upside down compared with the figure. `.ravel()` gives `tn, fp, fn, tp`.

## The metrics, in calculation order

### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{40 + 130}{200} = 0.85
$$

The share of all predictions that are right.

### Precision

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{40}{40 + 20} = 0.667
$$

"Of everything I **called** positive, how much was?" Reads down the predicted-1 column. High precision = few false alarms.

### Recall (sensitivity, true positive rate)

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{40}{40 + 10} = 0.80
$$

"Of all the **real** positives, how many did I catch?" Reads across the actual-1 row. High recall = few misses.

### Specificity (true negative rate)

$$
\text{Specificity} = \frac{TN}{TN + FP} = \frac{130}{130 + 20} = 0.867
$$

Recall for the negative class. The **false positive rate** is $1 - \text{specificity} = 20/150 = 0.133$.

### F1 score

$$
F_1 = \frac{2 \cdot P \cdot R}{P + R} = \frac{2 \times 0.667 \times 0.80}{0.667 + 0.80} = 0.727
$$

The **harmonic mean** of precision and recall: one number that is only high when both are high.

Why harmonic and not the ordinary average? Let's say a model predicts positive for exactly one row and gets it right, missing the other 49: $P = 1$, $R = 0.02$.

- Arithmetic mean: $(1 + 0.02)/2 = 0.51$, which looks half-decent.
- Harmonic mean: $2 \times 1 \times 0.02 / 1.02 = 0.039$, which correctly says the model is useless.

In the limit $P = 1, R = 0$, the arithmetic mean is 0.5 but F1 is 0. The harmonic mean is dragged down to the weaker of the two.

| Metric | Formula | Worked value |
|---|---|---|
| Accuracy | (TP + TN) / all | 0.850 |
| Precision | TP / (TP + FP) | 0.667 |
| Recall | TP / (TP + FN) | 0.800 |
| Specificity | TN / (TN + FP) | 0.867 |
| False positive rate | FP / (FP + TN) | 0.133 |
| F1 | 2PR / (P + R) | 0.727 |

## Why accuracy fails under imbalance

Let's say 1,000 transactions contain 5 frauds (0.5% positives). A "model" that always says "not fraud" scores:

$$
\text{Accuracy} = \frac{995}{1000} = 0.995 \qquad \text{Recall} = \frac{0}{5} = 0
$$

99.5% accurate and it catches nothing. Accuracy is dominated by the big class.

His accidents project shows the same thing in a milder form: 34% of accidents are severe, so predicting "not severe" for every row already scores accuracy **0.66**. The decision tree's 0.83 accuracy has to be judged against 0.66, not against 0. That is why his [py_acc README](https://github.com/jag5m33/py_acc) reports accuracy "for reference only" and leads with precision, recall, F1 and PR-AUC.

> **Key idea:** always compare with the **baseline**: the score of predicting the majority class (for accuracy) or the positive rate (for PR-AUC).

## The threshold trade-off

Most classifiers output a probability (`predict_proba`). The class comes from a **threshold**: predict 1 if $p \ge t$. The default $t = 0.5$ is just a convention.

![Precision rises and recall falls as the threshold goes up; F1 peaks in between.](figures/clsmetrics-threshold.svg)

The figure uses simulated scores with about 29% positives. Moving right (higher threshold), the model says "positive" less often: precision (teal) rises because the remaining calls are surer, recall (wine) falls because more positives slip under the bar. The dotted line marks where F1 is highest.

Choose the threshold on the **validation** set based on cost:

- Missing a positive is expensive (cancer, a severe accident, fraud) → lower threshold, favour recall.
- False alarms are expensive (spam filter deleting real email, costly interventions) → higher threshold, favour precision.

## PR curve and PR-AUC

The **precision-recall (PR) curve** plots precision against recall for every possible threshold. **PR-AUC** is the area under it; sklearn's `average_precision_score` (AP) is the standard way to compute it.

- Perfect model: PR-AUC = 1.
- **Random guessing: PR-AUC = the positive rate**, because a random ranking has precision equal to the share of positives at every recall.

So the baseline depends on the data. In his accidents project the positive rate is 0.34, so the baseline PR-AUC is **0.34**; the decision tree reached 0.822 and XGBoost 0.871.

## ROC curve and ROC-AUC

The **ROC curve** plots the true positive rate (recall) against the false positive rate for every threshold. **ROC-AUC** is the area under it.

- Perfect: 1.0. **Random: 0.5**, whatever the class balance (the diagonal).
- Interpretation: the probability that a randomly chosen positive gets a higher score than a randomly chosen negative.

![Left: precision-recall curve with the random baseline at the positive rate. Right: ROC curve with the random diagonal.](figures/clsmetrics-prroc.svg)

Both panels use the same simulated scores. On the left, the wine dashed line is the positive rate (0.29), the PR-AUC of a random ranking. On the right, the diagonal is random (0.5).

### When PR-AUC is better than ROC-AUC

The false positive rate divides by **all negatives**. When negatives vastly outnumber positives, even thousands of false alarms are a small FPR, so ROC-AUC stays high while precision is poor.

Simulated example, 0.5% positives, 200,000 rows, a moderately good score: **ROC-AUC = 0.92**, but **PR-AUC = 0.21**. The ROC number sounds excellent; the PR number tells you that most of what the model flags is false alarms.

> **Key idea:** with rare positives and when you care about the positive class, report PR-AUC. ROC-AUC is fine when classes are roughly balanced or both classes matter equally.

## Log loss as a metric

Log loss scores the **probabilities**, not the labels:

$$
\text{Log loss} = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log p^{(i)} + (1-y^{(i)})\log(1-p^{(i)})\right]
$$

It is the cost function from [logistic regression](#/t/logreg) used as a score. Lower is better; always predicting 0.5 scores $\log 2 = 0.693$. Use it when the probabilities themselves will be used (risk scores, expected costs). It heavily punishes confident wrong answers, so one $p = 0.001$ on a true positive costs 6.9 on its own.

## Multiclass averaging: macro, micro, weighted

With K classes, compute precision, recall and F1 **per class** (each class vs the rest), then average. Three ways:

- **Macro**: plain mean over classes. Every class counts equally, however rare.
- **Micro**: pool all TP, FP and FN across classes, then compute once. Every **example** counts equally, so big classes dominate. For single-label multiclass, micro F1 = accuracy.
- **Weighted**: mean over classes, weighted by each class's number of true examples (support).

### Tiny worked example

12 examples, 3 classes. A has 8 and is always right. B has 2, both predicted as A. C has 2: one right, one predicted as B.

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| A | 8 | 8/10 = 0.80 | 8/8 = 1.00 | 0.889 |
| B | 2 | 0/1 = 0.00 | 0/2 = 0.00 | 0.000 |
| C | 2 | 1/1 = 1.00 | 1/2 = 0.50 | 0.667 |

| Average | Calculation | F1 |
|---|---|---|
| Macro | (0.889 + 0 + 0.667) / 3 | **0.519** |
| Weighted | (8 × 0.889 + 2 × 0 + 2 × 0.667) / 12 | 0.704 |
| Micro | 9 right out of 12 | 0.750 (= accuracy) |

Micro and weighted look fine because class A is big and perfect. Macro exposes that class B is never found.

### Why he used macro F1 on LEDGAR

LEDGAR has 100 clause types, from 3,167 examples (Governing Laws) down to 23 (Books), a 138 to 1 ratio. Micro F1 and accuracy (both 0.839 on validation) are carried by the big classes. Macro F1 (0.780) gives each clause type an equal vote, so a model that ignores rare clause types is penalised. The 11-point gap between train and validation macro F1 (0.887 vs 0.780), against a 6-point gap in accuracy, showed that the overfitting is mostly in the rare classes. See [Multiclass (softmax)](#/t/softmax).

## Which metric when

| Situation | Lead with | Why |
|---|---|---|
| Balanced classes, all errors cost the same | Accuracy, F1 | Simple and fair here |
| Rare positive class (fraud, disease, severe accident) | PR-AUC, recall, precision, F1 | Accuracy and ROC-AUC look good by default |
| Missing a positive is costly | Recall (at an acceptable precision) | Count the misses |
| False alarms are costly | Precision (at an acceptable recall) | Count the false alarms |
| Comparing rankers regardless of threshold | PR-AUC (imbalanced) or ROC-AUC (balanced) | Threshold-free |
| Probabilities will be used directly | Log loss (plus a calibration plot) | Scores the probabilities |
| Multiclass, rare classes matter | Macro F1 | Every class counts equally |
| Multiclass, overall throughput matters | Micro F1 / accuracy, weighted F1 | Every example counts equally |

## Code

```python
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score,
                             roc_auc_score, f1_score, log_loss)

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])                     # actual labels
probs  = np.array([0.9, 0.6, 0.3, 0.4, 0.2, 0.7, 0.1, 0.8, 0.3, 0.05]) # predict_proba[:, 1]
y_pred = (probs >= 0.5).astype(int)                                    # threshold at 0.5

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # sklearn order: tn, fp, fn, tp
print(tp, fp, fn, tn)

print(classification_report(y_true, y_pred, target_names=["neg", "pos"]))  # P, R, F1 per class

prec, rec, thr = precision_recall_curve(y_true, probs)   # one point per threshold
print("PR-AUC (AP):", average_precision_score(y_true, probs))  # baseline = y_true.mean()
print("ROC-AUC    :", roc_auc_score(y_true, probs))           # baseline = 0.5
print("log loss   :", log_loss(y_true, probs))

# multiclass averaging (the worked example above)
yt = ["A"]*8 + ["B", "B", "C", "C"]
yp = ["A"]*8 + ["A", "A", "C", "B"]
for avg in ["macro", "micro", "weighted"]:
    print(avg, round(f1_score(yt, yp, average=avg), 3))   # 0.519, 0.75, 0.704
```

His error-analysis classes build on exactly these calls: [py_acc's ErrorAnalysis](https://github.com/jag5m33/py_acc/blob/main/src/classification_error_analysis.py) unpacks `confusion_matrix(...).ravel()` into TP/FP/FN/TN and then breaks the error rate down by feature (missed vs false-alarm rate per road type), and [py_legal_data's version](https://github.com/jag5m33/py_legal_data/blob/main/src/classification_error_analysis.py) uses `multilabel_confusion_matrix` to get TP/FP/FN/TN for each of the 100 clause types.

## His results

| Project | Model | Precision | Recall | F1 | PR-AUC | Accuracy |
|---|---|---|---|---|---|---|
| US accidents (validation) | Baseline | - | - | - | 0.340 | 0.660 |
| US accidents (validation) | Decision tree | 0.783 | 0.689 | 0.733 | 0.822 | 0.830 |
| US accidents (validation) | XGBoost | 0.821 | 0.763 | 0.791 | 0.871 | 0.863 |

| Project | Split | Accuracy | Macro F1 | Micro F1 |
|---|---|---|---|---|
| LEDGAR | Train | 0.899 | 0.887 | 0.899 |
| LEDGAR | Validation | 0.839 | 0.780 | 0.839 |

Both projects used the default 0.5 / argmax decision rule and have not scored their test sets yet.

> **Say it like this:** "I start from the confusion matrix. Precision is how many of my alarms were real, recall is how many real cases I caught, and F1 balances them. With rare positives I report PR-AUC against the positive-rate baseline rather than accuracy, and for many classes I use macro F1 so small classes count."

## Quick reference

| Term | Formula / meaning |
|---|---|
| Accuracy | (TP + TN) / all |
| Precision | TP / (TP + FP) |
| Recall, TPR, sensitivity | TP / (TP + FN) |
| Specificity, TNR | TN / (TN + FP) |
| FPR | FP / (FP + TN) = 1 − specificity |
| F1 | 2PR / (P + R) |
| PR-AUC baseline | positive rate |
| ROC-AUC baseline | 0.5 |
| Log loss | −mean[y log p + (1 − y) log(1 − p)] |
| Macro | mean over classes |
| Micro | pooled counts; = accuracy for single-label |
| Weighted | mean over classes weighted by support |
| sklearn layout | [[TN, FP], [FN, TP]] |
