# What supervised means

Supervised learning means learning from examples that come with the right answer, so the model can predict that answer for new examples it has never seen.

## The core words

- **Supervised learning**: the model sees inputs together with the correct output for each one, and learns to go from input to output. "Supervised" because the right answers act like a teacher marking the work.
- **Feature (X)**: an input column the model uses, such as house size or hour of day. All the features together are called $X$: one row per example, one column per feature.
- **Label / target (y)**: the thing we want to predict, such as price or "severe / not severe". Each row has one $y$.
- **Training example**: one row of data with its answer. Andrew Ng writes it $(x^{(i)}, y^{(i)})$, where $(i)$ just means "row number $i$", not a power.
- **Model**: a function $f(x)$ with adjustable numbers inside it (for example a weight $w$ and a bias $b$). **Training** means choosing those numbers so the predictions are close to the real $y$ values.

> **Key idea:** supervised = you have $y$. If you only have $X$ and want to find structure in it, that is [unsupervised learning](#/t/unsupintro).

## The two types

### Regression: predict a number

The target can be any value on a scale.

- House price from size and location.
- Tomorrow's electricity demand from today's demand and the temperature.

### Classification: predict a category

The target is one of a fixed set of classes. The model usually gives a probability for each class, then picks one.

- **Binary**: two classes, coded 0 and 1. Spam or not spam; severe or not severe accident (his [US accidents project](https://github.com/jag5m33/py_acc)).
- **Multiclass**: more than two classes, one per example. Which of 100 clause types a contract clause is (his [LEDGAR project](https://github.com/jag5m33/py_legal_data)).
- **Multi-label**: one example can have several labels at once, like a film tagged both "comedy" and "romance".

![Left: regression fits a line through points and reads off a number. Right: classification draws a boundary between two groups and reads off a side.](figures/supintro-regcls.svg)

In the figure, the gold line on the left gives a number for any size. The dashed boundary on the right only tells you which side (which class) a point falls on.

> **Trap:** the type of the **target** decides the task, not the type of the features. Predicting "1, 2, 3 or 4 stars" is still classification if the stars are categories.

## Do they use the same metrics?

No. A regression error is a distance ("off by £20k"). A classification error is a wrong category ("called it benign, it was malignant"). So each family has its own metrics.

The metrics have their own section of the notes: start with [Choosing the right metric](#/t/metricsintro), then [Regression metrics](#/t/regmetrics) and [Classification metrics](#/t/clsmetrics).

## Train, validation and test

You can't judge a model on the rows it learned from, because it may have just memorised them (overfitting, see [Bias & variance](#/t/biasvar)). So you split the data **before** fitting anything.

- **Training set (about 60%)**: the model learns here.
- **Validation set (about 20%)**: used to compare models and tune settings (called **hyperparameters**, e.g. tree depth or `C`).
- **Test set (about 20%)**: used **once**, at the very end, for an honest score.

![The data is cut into three blocks: fit on train, tune on validation, score the test block once.](figures/supintro-split.svg)

In the figure the arrows only go left to right: nothing from the test set flows back into training.

### The workflow, in order

1. **Split** into train / validation / test.
2. **Fit** any preprocessing (scaler, imputer, TF-IDF) and the model on **train only**.
3. **Score** on validation, change settings, repeat.
4. **Score the test set once** with the final model.

His projects follow this: py_acc uses a 60/20/20 split with median imputation fitted on train only; py_legal_data fits TF-IDF on train only.

### Stratified splits

A **stratified split** keeps the class mix the same in every part. If 34% of accidents are severe overall, each split also gets about 34% severe. This matters most for rare classes: a random split once left LEDGAR's "Books" class with no validation examples at all.

> **Trap:** split first, then preprocess. Fitting a scaler on all the data lets test information sneak into training (**data leakage**), and your scores come out too good.

## Code: a 60/20/20 stratified split

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# toy data: 1,000 rows, 5 features, about 30% positives
X, y = make_classification(n_samples=1000, n_features=5,
                           weights=[0.7, 0.3], random_state=42)

# step 1: keep 20% back as the test set (stratify keeps the class mix)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# step 2: split the other 80% into train and validation (0.25 of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print(len(X_train), len(X_val), len(X_test))   # 600 200 200
```

## Supervised models on this site

| Model | Task |
|---|---|
| [Linear regression](#/t/linreg) | Regression |
| [Logistic regression](#/t/logreg) | Binary classification (despite the name) |
| [Multiclass (softmax)](#/t/softmax) | Multiclass classification |
| [Decision trees](#/t/trees) | Both |
| [XGBoost](#/t/xgboost) | Both |

> **Say it like this:** "Supervised learning learns from examples with known answers. If the answer is a number it's regression, if it's a category it's classification, and each has its own metrics. I split 60/20/20, fit on train, tune on validation and score the test set once."

## Quick reference

| Term | Meaning |
|---|---|
| X | the features (inputs), one row per example |
| y | the target (answer), one per row |
| Regression | y is a number |
| Classification | y is a category: binary, multiclass or multi-label |
| Train / val / test | learn / tune / report once (60/20/20) |
| stratify=y | keep the class mix the same in every split |
| Leakage | information from val/test reaching training |
