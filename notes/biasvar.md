# Bias & variance

Bias and variance are the two ways a model can be wrong: bias is being consistently off target because the model is too simple, variance is being jumpy because the model memorised its particular training set.

## 1. Definitions in plain words

- **Training error:** the model's error on the data it learned from.
- **Validation error:** the model's error on held-out data it never saw while learning. Used to choose settings. (The **test** set is scored once, at the very end.)
- **Baseline error:** the best error you could reasonably hope for, e.g. **human-level performance** or the noise in the labels. Some error is unavoidable; nobody predicts house prices to the penny.
- **Bias:** error from **wrong assumptions**. The model is too simple to capture the real pattern, so it misses even on the training data. Also called **underfitting**.
- **Variance:** error from **over-sensitivity to the particular training set**. The model learns the noise as if it were pattern, so a different training sample would give a noticeably different model. Also called **overfitting**.

> **Trap:** "Bias" here is not the $b$ in $wx + b$, and not social or sampling bias. Same word, different meanings.

## 2. How to read them (Jag's anchor, made precise)

Jag's anchor: **"bias refers to the training set always, variance refers to the validation set always."** That is a good memory hook, with one correction to the second half.

| | Where you **read** it | How |
|---|---|---|
| **Bias** | **Training error**, compared with the **baseline** | Training error far above baseline = high bias |
| **Variance** | **The gap** between training and validation error | Validation error far above training error = high variance |

Why the correction matters: validation error **contains both** problems.

$$
\text{validation error} = \underbrace{\text{baseline}}_{\text{unavoidable}} + \underbrace{(\text{train} - \text{baseline})}_{\text{bias}} + \underbrace{(\text{val} - \text{train})}_{\text{variance}}
$$

So a high validation error on its own doesn't mean high variance. If training error is *also* high and the two are close, the validation error is high because of **bias**. Variance is the **extra** error that appears only when you move from training to validation data.

**Example** (baseline 2% error):

| | Train error | Val error | Bias (train − baseline) | Variance (val − train) | Diagnosis |
|---|---|---|---|---|---|
| A | 3% | 4% | 1 | 1 | Both low: good |
| B | 3% | 15% | 1 | **12** | High variance |
| C | 15% | 16% | **13** | 1 | High bias |
| D | 15% | 30% | **13** | **15** | Both high |

Models C and B have similar-looking validation errors (16% vs 15%) but opposite problems and opposite fixes. Only the training error tells them apart.

> **Say it like this:** "I read bias off the training error against a baseline, and variance off the gap between training and validation."

## 3. The four combinations

| Combination | Train error | Val error | Picture | What it means | Fix |
|---|---|---|---|---|---|
| **High bias, low variance** | High | High, **close** to train | A straight line through a curve | Too simple; consistently wrong in the same way | Bigger / more flexible model, more features, less regularisation |
| **Low bias, high variance** | Low | Much **higher** than train | A wiggly curve through every point | Memorised the training set, including its noise | More data, more regularisation, fewer features, simpler model |
| **Low bias, low variance** | Low (near baseline) | Low, close to train | Smooth curve following the trend | The goal | Ship it (after the test set) |
| **High bias, high variance** | High | Even higher | Wrong shape **and** jumpy | Too simple in some regions, too flexible in others; or very little data | Fix bias first, then variance |

The last row sounds contradictory. It happens, e.g. a model that's a poor fit overall but has rare classes it memorises (rare clause types in a text classifier), or a model trained on so little data that it's both wrong and unstable.

### Each one on its own

- **High bias** = the model's **average** guess is off target. Retrain it on many training sets and the predictions would all be wrong in the same direction.
- **Low bias** = on average, the model gets it right.
- **High variance** = retrain on a different training set and you get a **very different** model. Predictions scatter.
- **Low variance** = retrain on new data and you get almost the same model. Predictions are stable.

## 4. The dartboard

![Four dartboards. The bullseye is the true answer; each dot is a model trained on a different training set; the cross marks the average dot. Top left: tight cluster on the bullseye (low bias, low variance). Top right: centred on the bullseye but widely scattered (low bias, high variance). Bottom left: tight cluster, but off to one side (high bias, low variance). Bottom right: scattered and off to one side (high bias, high variance).](figures/biasvar-dartboard.svg)

**Read the figure:** bias is **where the cross is** relative to the bullseye (the average aim). Variance is **how spread out the dots are** (consistency). The two are independent: you can have either without the other. Top right is the overfit model: right on average, but any one training set could throw it far off.

## 5. Underfit, good, overfit on the same data

Fifteen training points (black) and fifteen validation points (orange circles), from a curved pattern plus noise. Fit polynomials of increasing degree:

![Three models fitted to the same 15 training points. Degree 1 (left) is a straight line that misses the curve: train and validation errors both high. Degree 4 (middle) follows the true dotted curve closely: both errors low. Degree 12 (right) passes almost exactly through every training point but swings wildly between them: training error near zero, validation error enormous.](figures/biasvar-fits.svg)

**Read the figure:** check the numbers in each panel's corner.

- **Left, degree 1:** train MSE 0.30, val MSE 0.20. Both high, close together → **high bias**. The line can't bend.
- **Middle, degree 4:** train 0.07, val 0.06. Both near the noise level (0.09) → **good**.
- **Right, degree 12:** train 0.01, val in the tens of thousands. Tiny training error, huge gap → **high variance**. It used its 13 parameters to thread through 15 noisy points.

(Validation can come out slightly below training by chance on small samples, as in the middle panel. It's the size of the gap that matters, not its sign when it's tiny.)

## 6. Error vs model complexity

Now repeat the experiment 400 times with fresh noise, for every degree from 0 to 12, and average.

![Error against polynomial degree. Training error (teal) falls steadily. Validation error (red) is U-shaped: high on the left, lowest around degree 3, rising on the right. The dashed blue bias² curve falls quickly to near zero; the dashed orange variance curve starts near zero and climbs. The left shaded region is the underfitting zone, the right is the overfitting zone, and an arrow marks the sweet spot at the bottom of the validation U.](figures/biasvar-complexity.svg)

**Read the figure, left to right:**

- **Underfitting zone (left, gold):** bias² (blue dashed) is large. Training and validation errors are both high and close together.
- **Sweet spot:** the bottom of the red U. Bias has dropped to nearly nothing, variance hasn't grown much yet. **This is the model you pick**, and you find it with the **validation** set.
- **Overfitting zone (right, pink):** training error keeps falling, but variance (orange dashed) rises and drags validation error up. The double-headed arrow is **the gap**: the variance signal from section 2.
- **Dotted grey line:** irreducible noise. No model goes below it on validation data.

> **Key idea:** Training error **always** falls as complexity rises. That's why you can never choose model complexity using training error. Use validation error.

## 7. The trade-off

Most single changes move bias and variance in **opposite directions**:

- More flexible model / more features → less bias, more variance.
- More regularisation / simpler model → less variance, more bias.

That is the **bias–variance trade-off**: the U-shape above is the **sum** of a falling bias² curve and a rising variance curve. Its bottom is where one stops falling faster than the other rises, not where they cross.

From Jag's NLP notes (after Andrew Ng's *Machine Learning Yearning*): **"Today it's less strict"**: with lots of data you can often lower both, e.g. a bigger model plus tuned regularisation, or more data. A large neural network trained on enough data with good regularisation can sit in the "low, low" corner.

## 8. The maths (optional)

For squared error, the expected error of a model at a point $x$, averaged over all the training sets you might have drawn, splits exactly into three parts:

$$
\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\right)^2\right]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}
$$

- $f(x)$: the true pattern. $\hat{f}(x)$: your model's prediction. $\mathbb{E}[\hat{f}(x)]$: the **average** prediction over many training sets (the cross on the dartboard).
- **Bias²**: how far the average prediction is from the truth, squared.
- **Variance**: how much individual predictions scatter around their own average.
- $\sigma^2$: noise in $y$ itself. No model can remove it.

This is exactly how the dashed curves in the complexity figure were computed: train on 400 noisy training sets, average the predictions (bias²), measure their spread (variance).

## 9. Learning curves: what more data does

A **learning curve** plots training and validation error against the **training set size** $m$.

![Learning curves. Left, a high-bias model (straight line): training error rises and validation error falls, and they meet at a high plateau well above the noise floor; more data doesn't lower it. Right, a high-variance model (degree 10 polynomial): training error starts near zero and validation error starts very high, with a big gap that shrinks as the training set grows.](figures/biasvar-learning.svg)

**Read the figure:**

- **Why training error rises with m:** with 8 points almost any model can fit them; with 500 it can't fit them all.
- **Left, high bias:** the two curves **meet quickly and flatten at a high level**. Adding data just adds more points the line also misses. **More data won't help.** Change the model.
- **Right, high variance:** a **large gap** that **narrows as m grows**. More data makes it harder to memorise noise. **More data helps**, and so does regularisation.

> **Key idea:** Before spending weeks collecting data, plot a learning curve. If it looks like the left panel, extra data is wasted effort.

## 10. Fixes (Andrew Ng's table)

| If you have... | Try | Why |
|---|---|---|
| **High bias** | Bigger / more flexible model (more layers, higher degree, deeper trees) | Lets the model bend to the real pattern |
| | Add features (from error analysis) | Gives it the information it was missing |
| | Add polynomial / interaction features | Lets a linear model curve |
| | **Decrease** λ (regularisation) / **increase** C | Removes the brake that was holding it back |
| | Train longer / better optimiser | Make sure it actually reached the minimum |
| **High variance** | **More training data** | Harder to memorise; noise averages out |
| | **Increase** λ / **decrease** C | Penalises big weights, smooths the model |
| | Fewer features (feature selection) | Fewer knobs to fit noise with |
| | Simpler model (lower degree, shallower trees, early stopping, dropout) | Less capacity to memorise |

> **Trap:** More data **doesn't fix bias**. A straight line through a curve stays wrong whether you have 100 points or 100,000.

Regularisation, the main dial for this trade-off, has its own chapter: [Regularisation](#/t/regularisation).

## 11. Jag's projects as real examples

**High variance: LEDGAR clause classification** ([py_legal_data](https://github.com/jag5m33/py_legal_data/blob/main/README.md)), TF-IDF + multinomial logistic regression, 100 clause types:

| Split | Macro F1 |
|---|---|
| Train | 0.887 |
| Validation | 0.780 |

- **Bias check:** train macro F1 0.887 is good but not perfect. Some clause types genuinely overlap (Governing Laws vs Applicable Laws), so the baseline itself isn't 1.0. Some bias, partly unavoidable.
- **Variance check:** the **11-point gap** between train and validation is the variance signal. Error analysis showed it is concentrated in **rare classes** with few training examples, where the model memorises the handful it has.
- **Levers** (from his notes): lower C (stronger regularisation), `class_weight="balanced"`, more examples or features for the rare types.

(Macro F1 is a score where higher is better, so the "gap" is train minus validation. With an error metric it's validation minus train. Same idea.)

**Low variance: US accidents severity** ([py_acc](https://github.com/jag5m33/py_acc/blob/main/README.md)), decision tree and XGBoost:

- Train and validation scores are **within 0.3 points** of each other for both models → **no overfitting**: variance is low.
- The remaining question is bias: is validation F1 of 0.79 (XGBoost) close to what's achievable from these features? That's answered by comparing with a baseline or a more flexible model, not by the gap.

## Quick reference

| Term | Meaning | Read it from |
|---|---|---|
| Bias | Systematically off target; too simple | Training error vs baseline |
| Variance | Jumpy; memorised the training set | Gap: validation − training error |
| Underfitting | High bias | Both errors high and close |
| Overfitting | High variance | Train low, validation much higher |
| Baseline | Best achievable error (human-level, label noise) | Domain knowledge, a simple model |
| Decomposition | Expected error = bias² + variance + σ² | Simulation / theory |
| Learning curve | Error vs training set size | Flat high plateau = bias; closing gap = variance |
| High bias fix | Bigger model, more features, less regularisation | More data won't help |
| High variance fix | More data, more regularisation, fewer features | |
