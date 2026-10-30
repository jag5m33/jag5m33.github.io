# Bias & variance

Bias and variance are the two ways a model can be wrong: bias is being consistently off target because the model is too simple, variance is being jumpy because the model memorised its particular training set.

## 1. The words in plain English

- **Training error:** the model's error on the data it learned from.
- **Validation error:** its error on held-out data it never saw while learning.
- **Baseline error:** the best error you could realistically hope for, e.g. human-level performance. Some error can't be avoided; nobody predicts house prices to the penny.

### Bias and underfitting (Jag's notes)

- **Bias** "refers to the training dataset error" → **high bias = high training error** when the model is fit.
- **Underfitting = high bias:** the model "misses true relationships despite the data"; the model is **too simple**.
- Picture: a flat straight line through scattered points that clearly curve.

> **Check:** next to underfitting Jag's notes say "(high var.)". Underfitting is usually high bias with **low** variance: a straight line barely changes if you retrain it on a new sample. Overfitting is the high-variance one, as his next section says.

### Variance and overfitting (Jag's notes)

- **Variance** "refers to the validation dataset error" → **high variance = high validation error** when the model predicts.
- **Overfitting = high variance (low bias):**
  - a wiggly curve through every training point; the model is **too complex**
  - **low training error**
  - it has "memorised noise, no patterns learned"
  - it "cannot generalise to new points"

> **Trap:** "bias" here is not the $b$ in $wx + b$. Same word, different meaning.

## 2. How to read them

Jag's anchor: **bias = training error, variance = validation error.** A good memory hook, with one small precision:

- **Bias:** look at the **training error** compared with the baseline. Training error far above baseline = high bias.
- **Variance:** look at the **gap** between training and validation error. Validation much worse than training = high variance.

Why? A high validation error on its own could be either problem. If training error is **also** high, it's bias. Variance is the **extra** error that appears only on new data.

**Example** (baseline error 2%):

| Model | Train error | Val error | Bias (train − baseline) | Variance (val − train) | Diagnosis |
|---|---|---|---|---|---|
| A | 3% | 4% | 1 | 1 | Both low: good |
| B | 3% | 15% | 1 | **12** | High variance |
| C | 15% | 16% | **13** | 1 | High bias |
| D | 15% | 30% | **13** | **15** | Both high |

B and C have nearly the same validation error but **opposite** problems and opposite fixes. Only the training error tells them apart.

> **Say it like this:** "I read bias from the training error against a baseline, and variance from the gap between training and validation."

## 3. The four combinations

| Combination | Train error | Val error | Picture | Fix |
|---|---|---|---|---|
| **High bias, low variance** | High | High, close to train | A straight line through a curve | Bigger model, more features, less regularisation |
| **Low bias, high variance** | Low | Much higher | A wiggly curve through every point | More data, more regularisation, simpler model |
| **Low bias, low variance** | Low | Low, close to train | Smooth curve following the trend | The goal |
| **High bias, high variance** | High | Even higher | Wrong shape **and** jumpy | Fix bias first, then variance |

The last one happens, for example, when a model fits most of the data badly but memorises a few rare classes, or when there is very little data.

## 4. The dartboard

![Four dartboards. The bullseye is the true answer; each dot is a model trained on a different training set; the cross marks the average dot. Top left: tight cluster on the bullseye (low bias, low variance). Top right: centred on the bullseye but widely scattered (low bias, high variance). Bottom left: tight cluster, but off to one side (high bias, low variance). Bottom right: scattered and off to one side (high bias, high variance).](figures/biasvar-dartboard.svg)

Imagine training the same model on many different training sets. Each dot is one of those models.

- **Bias** = where the **cross** (the average dot) is compared with the bullseye.
- **Variance** = how **spread out** the dots are.

Top right is the overfit model: right on average, but any single training set could throw it far off.

## 5. Underfit, good, overfit

Let's say we have 15 training points from a curved pattern plus noise, and we fit curves of increasing flexibility (polynomials of degree 1, 4 and 12):

![Three models fitted to the same 15 training points. Degree 1 (left) is a straight line that misses the curve: train and validation errors both high. Degree 4 (middle) follows the true dotted curve closely: both errors low. Degree 12 (right) passes almost exactly through every training point but swings wildly between them: training error near zero, validation error enormous.](figures/biasvar-fits.svg)

- **Left (straight line):** train error 0.30, val 0.20. Both high → **high bias**. The line can't bend.
- **Middle (degree 4):** train 0.07, val 0.06. Both low → **good**.
- **Right (degree 12):** train 0.01, val enormous. Huge gap → **high variance**. It threads through every noisy point.

## 6. Error vs model complexity: the trade-off graph

Jag's sketch: error against model complexity, **bias falling**, **variance rising**, total error **U-shaped**, with the **"sweet spot"** at the bottom. Here it is with real numbers, fitting every degree from 0 to 12:

![Error against polynomial degree. Training error (teal) falls steadily. Validation error (red) is U-shaped: high on the left, lowest around degree 3, rising on the right. The dashed blue bias curve falls quickly to near zero; the dashed orange variance curve starts near zero and climbs. The left shaded region is the underfitting zone, the right is the overfitting zone, and an arrow marks the sweet spot at the bottom of the validation U.](figures/biasvar-complexity.svg)

Read the figure left to right:

- **Underfitting zone (left):** bias (blue dashed) is high. Training and validation errors are both high.
- **Sweet spot (arrow):** the bottom of the red U, where bias and variance balance. **This is the model you pick**, using the validation set.
- **Overfitting zone (right):** training error keeps falling, but variance (orange dashed) rises and drags validation error up. The double-headed arrow is **the gap**.

> **Key idea:** training error **always** falls as the model gets more complex. So never choose complexity using training error; use validation error.

This is the **bias–variance trade-off**: making the model more flexible lowers bias but raises variance, and the other way round. With lots of data you can often lower both (Jag's NLP notes: "today it's less strict").

## 7. Fixes (Andrew Ng's list)

| If you have... | Try |
|---|---|
| **High bias** | Bigger / more flexible model (higher degree, deeper trees) |
| | More or better features |
| | **Less** regularisation (smaller λ, bigger C) |
| **High variance** | **More training data** |
| | **More** regularisation (bigger λ, smaller C) |
| | Fewer features, simpler model |

> **Trap:** more data does **not** fix bias. A straight line through a curve stays wrong with 100 or 100,000 points.

Regularisation, the main dial for this, has its own chapter: [Regularisation](#/t/regularisation).

## 8. Jag's projects as real examples

**High variance: LEDGAR clause classification** ([py_legal_data](https://github.com/jag5m33/py_legal_data/blob/main/README.md)), TF-IDF + logistic regression, 100 clause types:

| Split | Macro F1 |
|---|---|
| Train | 0.887 |
| Validation | 0.780 |

- The **11-point gap** is the variance signal. Error analysis showed it comes mostly from **rare classes** with few examples, which the model memorises.
- Levers from his notes: lower C (stronger regularisation), `class_weight="balanced"`, more examples for rare types.
- (F1 is "higher is better", so the gap is train minus validation.)

**Low variance: US accidents severity** ([py_acc](https://github.com/jag5m33/py_acc/blob/main/README.md)), decision tree and XGBoost:

- Train and validation scores are within 0.3 points of each other → **no overfitting**.
- The open question is bias: is a validation F1 of 0.79 (XGBoost) close to the best possible with these features?

## Quick reference

| Term | Meaning | Read it from |
|---|---|---|
| Bias | Too simple; consistently off target | Training error vs baseline |
| Variance | Too sensitive; memorised the training set | Gap: validation − training |
| Underfitting | High bias | Both errors high and close |
| Overfitting | High variance | Train low, validation much higher |
| Sweet spot | Bottom of the validation U | Validation error |
| High bias fix | Bigger model, more features, less regularisation | More data won't help |
| High variance fix | More data, more regularisation, simpler model | |
