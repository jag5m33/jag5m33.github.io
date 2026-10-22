# What supervised means

Supervised learning means learning from examples that come with the right answer, so the model can predict that answer for new examples it has never seen.

## The core definitions

- **Supervised learning**: the model is given inputs together with the correct output for each one, and learns a mapping from input to output. "Supervised" because the right answers act like a teacher marking the work.
- **Feature (X)**: an input variable the model uses, such as house size, hour of day or a word's TF-IDF weight. All features together form the feature matrix $X$, one row per example and one column per feature.
- **Label / target (y)**: the thing we want to predict, such as price or "severe / not severe". Each row of $X$ has one $y$.
- **Training example**: one row, written $(x^{(i)}, y^{(i)})$. The superscript $(i)$ means "the $i$-th example", not a power. $m$ is the number of examples and $n$ the number of features.
- **Model**: a function $f(x)$ with adjustable numbers (parameters, e.g. weights $w$ and bias $b$). Training means choosing the parameters so $f(x^{(i)})$ is close to $y^{(i)}$ on the training rows.

> **Key idea:** supervised = you have $y$. If you only have $X$ and want to find structure in it, that is [unsupervised learning](#/t/unsupintro).

## The two subtypes

### Regression: predict a number

The target is a continuous quantity. The answer can be any value on a scale.

- House price from size and location.
- Tomorrow's electricity demand from today's demand and the temperature.
- Length of stay in hospital.

### Classification: predict a category

The target is one of a fixed set of classes. The model usually outputs a probability per class, then picks one.

- **Binary**: two classes, usually coded 0 and 1. Spam or not spam; severe or not severe accident (his [US accidents project](https://github.com/jag5m33/py_acc)).
- **Multiclass**: more than two classes, exactly one per example. Which of 100 clause types a contract clause is (his [LEDGAR project](https://github.com/jag5m33/py_legal_data)).
- **Multi-label**: each example can have several labels at once. A film tagged both "comedy" and "romance"; a clause that is both "Confidentiality" and "Notices". This is usually solved as one binary classifier per label.

![Left: regression fits a line through points and reads off a number. Right: classification draws a boundary between two groups and reads off a side.](figures/supintro-regcls.svg)

In the figure, the left panel's gold line gives a numeric prediction for any size; the right panel's dashed boundary only tells you which side (which class) a point falls on.

> **Trap:** the type of the **target** decides the task, not the type of the features. Predicting "1, 2, 3 or 4 stars" with numbers as labels is still classification if the classes are categories; predicting a probability does not make classification into regression.

## Do all supervised models use the same metrics?

No. A regression error is a distance ("off by £20k"); a classification error is a wrong category ("called it benign, it was malignant"). So the two families measure different things.

| Family | Metric | One-line maths | Read it as |
|---|---|---|---|
| Regression | MSE | mean of (y − ŷ)² | average squared miss; punishes big misses |
| Regression | RMSE | square root of MSE | typical miss, in the units of y |
| Regression | MAE | mean of the absolute miss | typical miss, robust to outliers |
| Regression | R² | 1 − SS_res / SS_tot | share of the variance in y explained |
| Regression | Adjusted R² | 1 − (1 − R²)(m − 1)/(m − n − 1) | R² with a penalty for extra features |
| Classification | Accuracy | (TP + TN) / all | share of predictions that are right |
| Classification | Precision | TP / (TP + FP) | of those called positive, how many were |
| Classification | Recall | TP / (TP + FN) | of the real positives, how many were caught |
| Classification | F1 | 2PR / (P + R) | harmonic mean of precision and recall |
| Classification | PR-AUC | area under precision vs recall | ranking quality, focused on positives |
| Classification | ROC-AUC | area under TPR vs FPR | chance a random positive outranks a random negative |
| Classification | Log loss | −mean of log(p of true class) | how good the probabilities are |

Full chapters: [Regression metrics](#/t/regmetrics) and [Classification metrics](#/t/clsmetrics).

## Train, validation and test

You cannot judge a model on the rows it learned from: it may have memorised them (overfitting, see [Bias & variance](#/t/biasvar)). So the data is split before anything is fitted.

- **Training set (about 60%)**: the model learns its parameters here.
- **Validation set (about 20%)**: used to compare models and tune settings (hyperparameters such as `C`, tree depth, the decision threshold).
- **Test set (about 20%)**: touched **once**, at the very end, to report an honest score. If you tune on it, it stops being an unseen estimate.

![The data is cut into three blocks: fit on train, tune on validation, score the test block once.](figures/supintro-split.svg)

The figure shows the order of work: the arrows go left to right only. Nothing learned from validation or test flows back into the training step, except your choice of settings from validation.

### The workflow, in order

1. **Split** into train / validation / test (before any fitting, scaling or imputing).
2. **Fit** preprocessing (scaler, imputer, TF-IDF vocabulary) and the model on **train only**.
3. **Apply** the fitted preprocessing to validation; score; change settings; repeat.
4. **Freeze** the final model, then score the **test** set once.

His projects follow this: py_acc uses 60/20/20 with median imputation fitted on train only; py_legal_data fits TF-IDF on train only and has not yet scored its 10,000-clause test set.

### Stratified splits for classification

A **stratified split** keeps the class proportions the same in every part. With 34% severe accidents overall, each split also gets about 34% severe. This matters most for rare classes: a random split could leave a rare class almost absent from validation (LEDGAR's "Books" class ended up with no validation examples at all).

> **Trap:** split first, then preprocess. Fitting a scaler or TF-IDF on all the data lets test information leak into training (data leakage), and the scores come out too good.

## Code: a 60/20/20 stratified split

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# toy data: 1,000 rows, 5 features, about 30% positives
X, y = make_classification(n_samples=1000, n_features=5,
                           weights=[0.7, 0.3], random_state=42)

# step 1: hold out 20% as the test set, keeping the class mix (stratify=y)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# step 2: split the remaining 80% into 60% train / 20% validation
# 0.25 of the 80% left = 20% of the original
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print(len(X_train), len(X_val), len(X_test))     # 600 200 200
print(y_train.mean(), y_val.mean(), y_test.mean())  # nearly identical positive rates
```

## Map of the supervised models on this site

| Model | Task | Typical metric |
|---|---|---|
| [Linear regression](#/t/linreg) | Regression | RMSE, MAE, R² |
| [Logistic regression](#/t/logreg) | Binary classification (despite the name) | F1, PR-AUC, log loss |
| [Multiclass (softmax)](#/t/softmax) | Multiclass classification | macro F1 |
| [Decision trees](#/t/trees) | Both (classifier or regressor) | depends on task |
| [XGBoost](#/t/xgboost) | Both (classifier or regressor) | depends on task |

Trees and XGBoost appear under classification here because that is how he used them on the accidents data, but `DecisionTreeRegressor` and `XGBRegressor` exist too.

## Supervised vs unsupervised

| | Supervised | [Unsupervised](#/t/unsupintro) |
|---|---|---|
| Data | X and y | X only |
| Goal | predict y for new rows | find groups, structure or oddities |
| Example | predict accident severity | cluster accidents into types |
| Scoring | compare predictions with known y | no right answer; internal measures |

> **Say it like this:** "Supervised learning fits a function from features to a known target. If the target is a number it's regression, if it's a category it's classification, and each has its own metrics. I split 60/20/20, fit on train, tune on validation and score the test set once."

## Quick reference

| Term | Meaning |
|---|---|
| X | feature matrix, m rows by n columns |
| y | target / label, one per row |
| (x⁽ⁱ⁾, y⁽ⁱ⁾) | the i-th training example |
| Regression | y is a number |
| Classification | y is a category: binary, multiclass or multi-label |
| Train / val / test | fit / tune / report once (60/20/20) |
| stratify=y | keep class proportions equal across splits |
| Leakage | information from val/test reaching training |
