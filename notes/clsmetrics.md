# Classification metrics

Classification metrics count the ways a classifier is right and wrong, and the right one depends on which mistakes cost more and how rare the positive class is.

## The confusion matrix

Everything starts from four counts. Call the class you care about **positive** (1), the other **negative** (0).

- **TP (true positive)**: predicted 1, actually 1. A correct catch.
- **FP (false positive)**: predicted 1, actually 0. A false alarm.
- **FN (false negative)**: predicted 0, actually 1. A miss.
- **TN (true negative)**: predicted 0, actually 0. Correctly left alone.

![A 2 by 2 confusion matrix with 40 true positives, 10 false negatives, 20 false positives and 130 true negatives.](figures/clsmetrics-confusion.svg)

In the figure the teal cells are correct and the wine cells are mistakes. We'll use these numbers (200 rows, 50 actual positives) for every example below.

> **Trap:** sklearn's `confusion_matrix` puts **actual** on the rows and **predicted** on the columns, with class 0 first: `[[TN, FP], [FN, TP]]`. That is upside down compared with the figure. `.ravel()` gives `tn, fp, fn, tp`.

## The metrics, worked from one matrix

### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{\text{all}} = \frac{40 + 130}{200} = 0.85
$$

The share of all predictions that are right.

### Precision

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{40}{40 + 20} = 0.667
$$

"Of everything I **called** positive, how much really was?" High precision = few false alarms.

### Recall

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{40}{40 + 10} = 0.80
$$

"Of all the **real** positives, how many did I catch?" High recall = few misses. Also called sensitivity.

### F1 score

$$
F_1 = \frac{2 \times P \times R}{P + R} = \frac{2 \times 0.667 \times 0.80}{0.667 + 0.80} = 0.727
$$

$P$ is precision and $R$ is recall. F1 is one number that is only high when **both** are high.

Why not just the ordinary average? Let's say a model flags only one row, gets it right, and misses the other 49 positives: $P = 1$, $R = 0.02$. The ordinary average is 0.51, which looks half-decent. F1 is $2 \times 1 \times 0.02 / 1.02 = 0.039$, which correctly says the model is useless. F1 gets dragged down to the weaker of the two.

| Metric | Formula | Worked value |
|---|---|---|
| Accuracy | (TP + TN) / all | 0.850 |
| Precision | TP / (TP + FP) | 0.667 |
| Recall | TP / (TP + FN) | 0.800 |
| F1 | 2PR / (P + R) | 0.727 |

## Why accuracy fails with imbalance

Let's say 1,000 transactions contain 5 frauds. A "model" that always says "not fraud" gets 995 right: **accuracy 0.995**, but **recall 0**. It catches nothing. Accuracy is dominated by the big class.

His accidents project shows the same thing in a milder form: 34% of accidents are severe, so predicting "not severe" for every row already scores accuracy **0.66**. The decision tree's 0.83 has to be judged against 0.66, not against 0. That is why his [py_acc README](https://github.com/jag5m33/py_acc) reports accuracy "for reference only" and leads with precision, recall, F1 and PR-AUC.

> **Key idea:** always compare with the **baseline**: the score you'd get by always predicting the most common class.

## The threshold trade-off

Most classifiers output a probability (`predict_proba`). The class comes from a **threshold**: predict 1 if the probability is at least $t$. The default $t = 0.5$ is just a convention.

![Precision rises and recall falls as the threshold goes up; F1 peaks in between.](figures/clsmetrics-threshold.svg)

The figure uses simulated scores with about 29% positives. Moving right (higher threshold), the model says "positive" less often: precision (teal) rises because the remaining calls are surer; recall (wine) falls because more positives slip under the bar. The dotted line marks where F1 is highest.

Choose the threshold on the **validation** set, based on cost:

- Missing a positive is expensive (cancer, a severe accident, fraud) → lower threshold, favour recall.
- False alarms are expensive (spam filter deleting real email) → higher threshold, favour precision.

## PR-AUC and ROC-AUC, in words

Both curves try **every** threshold at once, so they judge how well the model **ranks** cases, not one particular cut-off. AUC means "area under the curve": 1 is perfect.

- **PR curve**: precision against recall at every threshold. **PR-AUC** is its area. A random model scores about the **positive rate** (34% positives → baseline 0.34). In his accidents project the decision tree reached 0.822 and XGBoost 0.871 against that 0.34 baseline.
- **ROC curve**: recall against the **false positive rate** (the share of real negatives that got flagged) at every threshold. A random model scores **0.5**, whatever the class balance.

![Left: precision-recall curve with the random baseline at the positive rate. Right: ROC curve with the random diagonal.](figures/clsmetrics-prroc.svg)

On the left, the wine dashed line is the positive rate (0.29), what a random model would score. On the right, the diagonal is random (0.5).

**Which to use?** When positives are rare, ROC-AUC can look excellent even when most of the model's alarms are false, because there are so many negatives that thousands of false alarms are still a small share. PR-AUC exposes this. So with rare positives, report PR-AUC; ROC-AUC is fine when classes are roughly balanced.

## Many classes: macro vs micro F1

With many classes, work out F1 **for each class** (that class vs the rest), then combine:

- **Macro F1**: a plain average over classes. Every class counts equally, however rare. If the model never finds a small class, macro F1 drops clearly.
- **Micro F1**: pool all the hits and mistakes together, then compute once. Every **example** counts equally, so big classes dominate. When each example has one label, micro F1 equals accuracy.
- **Weighted F1**: average over classes, weighted by how many examples each has. Sits in between.

### Why he used macro F1 on LEDGAR

LEDGAR has 100 clause types, from 3,167 examples (Governing Laws) down to 23 (Books). Accuracy and micro F1 (both 0.839 on validation) are carried by the big classes. Macro F1 (0.780) gives each clause type an equal vote, so a model that ignores rare clause types gets penalised. The 11-point drop in macro F1 from train to validation (0.887 to 0.780), against a 6-point drop in accuracy, showed the overfitting is mostly in the rare classes. See [Multiclass (softmax)](#/t/softmax).

## Which metric when

| Situation | Lead with |
|---|---|
| Balanced classes, all errors cost the same | Accuracy, F1 |
| Rare positive class (fraud, disease, severe accident) | PR-AUC, precision, recall, F1 |
| Missing a positive is costly | Recall |
| False alarms are costly | Precision |
| Many classes, rare ones matter | Macro F1 |

## Code

```python
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
                             average_precision_score, roc_auc_score)

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])                     # actual labels
probs  = np.array([0.9, 0.6, 0.3, 0.4, 0.2, 0.7, 0.1, 0.8, 0.3, 0.05]) # predict_proba[:, 1]
y_pred = (probs >= 0.5).astype(int)                                    # threshold at 0.5

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()   # sklearn order
print(classification_report(y_true, y_pred))                # precision, recall, F1
print("PR-AUC :", average_precision_score(y_true, probs))   # baseline = positive rate
print("ROC-AUC:", roc_auc_score(y_true, probs))             # baseline = 0.5
```

His error-analysis classes build on exactly these calls: [py_acc's ErrorAnalysis](https://github.com/jag5m33/py_acc/blob/main/src/classification_error_analysis.py) unpacks `confusion_matrix(...).ravel()` into TP/FP/FN/TN and then breaks the error rate down by feature (missed vs false-alarm rate per road type), and [py_legal_data's version](https://github.com/jag5m33/py_legal_data/blob/main/src/classification_error_analysis.py) gets TP/FP/FN/TN for each of the 100 clause types.

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

Both projects used the default 0.5 / top-class decision rule and have not scored their test sets yet.

> **Say it like this:** "I start from the confusion matrix. Precision is how many of my alarms were real, recall is how many real cases I caught, and F1 balances them. With rare positives I report PR-AUC against the positive-rate baseline rather than accuracy, and with many classes I use macro F1 so small classes count."

## Quick reference

| Term | Formula / meaning |
|---|---|
| Accuracy | (TP + TN) / all |
| Precision | TP / (TP + FP): how many alarms were real |
| Recall | TP / (TP + FN): how many real cases were caught |
| F1 | 2PR / (P + R) |
| PR-AUC baseline | positive rate |
| ROC-AUC baseline | 0.5 |
| Macro F1 | every class counts equally |
| Micro F1 | every example counts equally (= accuracy) |
| sklearn layout | [[TN, FP], [FN, TP]] |
