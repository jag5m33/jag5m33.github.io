# Classification metrics

Classification metrics count the ways a classifier is right and wrong, and the right one depends on which mistake costs more and on how rare the class you care about is.

## The running example: a disease test on 200 people

Let's say 200 people take a test for a disease. **50 are really ill** and 150 are healthy. "Ill" is the class we care about, so we call it **positive** (1); healthy is **negative** (0).

> **What you need to know**
> - **TP (true positive):** said ill, was ill. A correct catch.
> - **FN (false negative):** said healthy, was ill. A **miss**.
> - **FP (false positive):** said ill, was healthy. A **false alarm**.
> - **TN (true negative):** said healthy, was healthy. Correctly left alone.
> - every classification metric is built from these four counts

![The confusion matrix for the disease test: 40 true positives, 10 false negatives, 20 false positives and 130 true negatives.](figures/clsmetrics-confusion.svg)

In the figure, teal cells are correct and wine cells are mistakes: the test caught 40 of the 50 ill people, missed 10, and wrongly flagged 20 healthy people.

> **Trap:** sklearn's `confusion_matrix` puts **actual** on the rows and **predicted** on the columns, class 0 first: `[[TN, FP], [FN, TP]]`. That's upside down compared with the figure. `.ravel()` gives `tn, fp, fn, tp`.

**Best with (all the metrics in this section):** any classifier: [logistic regression](#/t/logreg), [Naive Bayes](#/t/naivebayes), [decision trees](#/t/trees), [random forest](#/t/randomforest), [XGBoost](#/t/xgboost). They only need the predicted class, so the choice depends on the **problem**, not the model.

## Accuracy

> **What you need to know**
> - **In plain words:** the share of all predictions that are right
> - **In context:** (40 + 130) / 200 = **0.85**
> - **Use it when:** classes are roughly balanced and both mistakes cost about the same (handwritten digits, cats vs dogs)
> - **Watch out:** on imbalanced data it flatters a useless model. Saying "healthy" to everyone already scores 150 / 200 = **0.75** here. With 5 frauds in 1,000 transactions, "never fraud" scores 0.995 and catches nothing

$$
\text{Accuracy} = \frac{TP + TN}{\text{all}}
$$

## Precision

> **What you need to know**
> - **In plain words:** of everyone I **flagged** as ill, how many really were? High precision = few false alarms
> - **In context:** 40 / (40 + 20) = **0.667**: one in three flagged people is actually healthy
> - **Use it when:** false alarms are costly: a **spam filter** (deleting a real email is worse than letting spam through), **approving loans** (each one you approve should really be safe), recommending products
> - **Watch out:** you can get high precision by flagging only the most obvious cases and missing most positives

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

## Recall

> **What you need to know**
> - **In plain words:** of all the people who **really** were ill, how many did I catch? High recall = few misses. Also called **sensitivity**
> - **In context:** 40 / (40 + 10) = **0.80**: the test catches 80% of ill people
> - **Use it when:** missing a positive is costly: **cancer screening**, **fraud detection**, severe accidents. A false alarm costs a follow-up check; a miss costs much more
> - **Watch out:** flagging everyone gives recall 1.0, so always look at precision alongside it

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

## F1 score

> **What you need to know**
> - **In plain words:** one number that is only high when precision **and** recall are both high
> - **In context:** 2 × 0.667 × 0.80 / (0.667 + 0.80) = **0.727**
> - **Use it when:** classes are imbalanced and you care about both mistakes; comparing models with one number (the accidents project)
> - **Watch out:** it treats a miss and a false alarm as equally bad; if they aren't, lead with recall or precision instead

$$
F_1 = \frac{2 \times P \times R}{P + R}
$$

$P$ is precision and $R$ is recall. Why not a plain average? Say a test flags just one person, correctly, and misses the other 49: $P = 1$, $R = 0.02$. The plain average is 0.51, which looks half-decent. F1 is **0.039**, which correctly says the test is useless. F1 is dragged down to the weaker of the two.

## The threshold trade-off

> **What you need to know**
> - most classifiers output a **probability** (`predict_proba`); a **threshold** turns it into a class: "ill" if the probability is at least $t$
> - the default $t = 0.5$ is just a convention
> - **lower** threshold → catch more (recall up), more false alarms (precision down); **higher** threshold → the opposite
> - choose it on the **validation** set, based on which mistake costs more

![Precision rises and recall falls as the threshold goes up; F1 peaks in between.](figures/clsmetrics-threshold.svg)

In the figure (simulated scores), moving right means saying "positive" less often: precision (teal) rises, recall (wine) falls, and the dotted line marks the best F1.

**In context:** for cancer screening you'd lower the threshold (catch more, accept more follow-ups); for a spam filter you'd raise it (only bin email you're very sure about).

## PR-AUC and ROC-AUC

> **What you need to know**
> - **In plain words:** both try **every** threshold at once and measure how well the model **ranks** positives above negatives. AUC = "area under the curve"; 1 is perfect
> - **PR-AUC:** area under the precision-against-recall curve. A random model scores the **share of positives** (0.25 in the disease test)
> - **ROC-AUC:** area under the curve of recall against the false-alarm rate (share of real negatives flagged). A random model scores **0.5**, whatever the balance
> - **Best with:** models that give probabilities or scores: logistic regression, random forest, XGBoost. Naive Bayes probabilities are too extreme to trust as real chances, but their **ranking** is fine, so AUCs still work
> - **Use it when:** comparing models before you've chosen a threshold; **PR-AUC when positives are rare** (fraud, disease, severe accidents), ROC-AUC when classes are roughly balanced
> - **Watch out:** with rare positives, ROC-AUC can look excellent while most alarms are false, because thousands of false alarms are still a small share of a huge negative class
>
> **Skip:** the formulas; drawing the curves by hand

![Left: precision-recall curve with the random baseline at the share of positives. Right: ROC curve with the random diagonal.](figures/clsmetrics-prroc.svg)

On the left, the wine dashed line is what a random model would score (the share of positives in this simulated data). On the right, the diagonal is random (0.5).

## Many classes: macro, micro and weighted F1

> **What you need to know**
> - with many classes, work out F1 **for each class** (that class vs the rest), then combine
> - **macro F1:** a plain average over classes; every class counts **equally**, however rare
> - **micro F1:** pool every row's hits and mistakes, then compute once; every **row** counts equally, so big classes dominate. With one label per row it **equals accuracy**
> - **weighted F1:** average over classes, weighted by class size; in between
> - **Best with:** any multi-class model: [softmax regression](#/t/softmax), Naive Bayes, random forest, XGBoost
> - **Use it when:** macro F1 for **many imbalanced classes where rare ones matter** (Jag's 100 LEDGAR clause types; rare diseases in a diagnosis model); micro/accuracy when only overall hit rate matters
>
> **Skip:** the per-class formulas; `f1_score(..., average="macro")` does it

**In context: why Jag used macro F1 on LEDGAR.** LEDGAR has 100 clause types, from 3,167 examples (Governing Laws) down to 23 (Books), a 138 : 1 imbalance. Accuracy and micro F1 (both 0.839 on validation) are carried by the big classes. Macro F1 (**0.780**) gives each clause type an equal vote, so ignoring rare clause types is penalised. Macro F1 fell 11 points from train to validation (0.887 → 0.780) against 6 for accuracy, so the overfitting is mostly in the rare classes. See [Softmax regression](#/t/softmax).

## Log loss, in one line

**Log loss** scores the predicted probabilities themselves: confident and wrong is punished hard, confident and right costs almost nothing. It's the cost [logistic regression](#/t/logreg) minimises and what XGBoost watches as `eval_metric="logloss"`. Lower is better. It needs real probabilities, so don't trust it for Naive Bayes.

## Jag's real results

| Accidents (validation) | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Baseline (34% severe) | - | - | - | 0.340 |
| [Decision tree](#/t/trees) | 0.783 | 0.689 | 0.733 | 0.822 |
| [XGBoost](#/t/xgboost) | 0.821 | 0.763 | 0.791 | 0.871 |

- About 34% of accidents are severe, so the PR-AUC baseline is 0.34; both models are far above it.
- XGBoost wins on every metric; the biggest gain is **recall** (+7.4 points): more severe accidents caught.
- **LEDGAR** ([Legal Clause NLP](#/p/legal)): validation **macro F1 0.780**, accuracy 0.839.
- Both used the default 0.5 / top-class rule; test sets not scored yet.

## Which one?

| Scenario | Lead with | Why |
|---|---|---|
| Balanced classes, both mistakes cost the same | accuracy, F1 | accuracy isn't misleading here |
| Cancer screening, fraud, severe accidents | **recall** (+ PR-AUC) | a miss is the expensive mistake |
| Spam filter, approving loans | **precision** | a false alarm is the expensive mistake |
| Imbalanced, both mistakes matter | **F1**, PR-AUC | accuracy hides a useless model |
| Comparing models before picking a threshold | **PR-AUC** (rare positives) / ROC-AUC (balanced) | judges the ranking at every threshold |
| Many classes, rare ones matter (LEDGAR) | **macro F1** | every class gets an equal vote |
| You need trustworthy probabilities | log loss | scores the probabilities themselves |

## Code

```python
import numpy as np
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])                      # actual: 1 = ill
probs  = np.array([.9, .6, .3, .4, .2, .7, .1, .8, .3, .05])            # predict_proba(X)[:, 1]
y_pred = (probs >= 0.5).astype(int)                                     # threshold at 0.5

print(classification_report(y_true, y_pred))           # precision, recall, F1, accuracy
print("PR-AUC :", average_precision_score(y_true, probs))   # baseline = share of positives
print("ROC-AUC:", roc_auc_score(y_true, probs))             # baseline = 0.5
```

His error-analysis classes build on these calls: [py_acc's ErrorAnalysis](https://github.com/jag5m33/py_acc/blob/main/src/classification_error_analysis.py) unpacks `confusion_matrix(...).ravel()` into TP/FP/FN/TN and breaks errors down by feature, and [py_legal_data's version](https://github.com/jag5m33/py_legal_data/blob/main/src/classification_error_analysis.py) does it for each of the 100 clause types.

## Interview one-liners

- **Precision vs recall?** "Precision: how many of my alarms were real. Recall: how many real cases I caught."
- **Why F1, not the average?** "F1 is only high when both are high; a plain average hides a model that catches almost nothing."
- **PR-AUC or ROC-AUC?** "PR-AUC when positives are rare, because ROC-AUC can look great while most alarms are false."
- **Macro vs micro F1?** "Macro gives every class an equal vote; micro gives every row one, so it equals accuracy for single-label data."
- **How do you choose the threshold?** "On the validation set, from the cost of a miss vs a false alarm."

## Quick reference

| Term | Formula / meaning | Disease test |
|---|---|---|
| Accuracy | (TP + TN) / all | 0.85 (baseline 0.75) |
| Precision | TP / (TP + FP): how many alarms were real | 0.667 |
| Recall | TP / (TP + FN): how many real cases were caught | 0.80 |
| F1 | 2PR / (P + R) | 0.727 |
| PR-AUC | ranking quality; baseline = share of positives | baseline 0.25 |
| ROC-AUC | ranking quality; baseline = 0.5 | |
| Macro / micro F1 | every class equal / every row equal (= accuracy) | |
| Log loss | punishes confident wrong probabilities | |
| sklearn layout | `[[TN, FP], [FN, TP]]` | |
