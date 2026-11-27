# Loss functions

How wrong a prediction is: different problems need different loss functions.

**On this page**

- Loss vs cost
- Regression: MSE, MAE, Huber loss
- Classification: binary and categorical cross entropy
- Softmax

## Which loss function?

Deep learning (ANN: artificial neural network) has two kinds of problem:

| | Output | Loss functions |
|---|---|---|
| **Regression** | continuous | MSE, MAE, Huber loss |
| **Classification** | a class | cross entropy: binary CE, categorical CE |

Depending on the type of problem, you have a different loss function.

## Loss vs cost

Dataset = 100 records:

1. Forward propagation → $\hat y$
2. Calculate the loss: $\tfrac{1}{2}(y - \hat y)^2$
3. Back propagation

Cycle through 1 → 3 for 10 records: the **batch size**. When you've done 10 records, calculate the loss for them:

$$\text{cost} = \frac{1}{2}\sum_{i=1}^{n} (y_i - \hat y_i)^2$$

> **Definition:** **loss** = the error for **one** data point. **Cost** = the error for a **batch** of data points.

> **Check:** the cost is usually written as an **average**, $\frac{1}{n}\sum$, rather than $\frac{1}{2}\sum$. The ½ is only there to make the derivative neater.

![Error against loss for each regression loss: MSE is a U-shaped curve, MAE a V, Huber a U in the middle that turns into straight lines past δ.](figures/dl-losses.svg)

## 1) MSE (mean squared error)

$$\text{loss} = \frac{1}{2}(y - \hat y)^2 \qquad \text{cost} = \frac{1}{2}\sum_{i=1}^{n}(y_i - \hat y_i)^2$$

It's a **quadratic equation**: $(a + b)^2 = a^2 + b^2 + 2ab$, so

$$(y - \hat y)^2 = y^2 + \hat y^2 - 2\hat y y$$

Quadratic → a U-shaped curve → gradient descent.

**Benefits:**

1. Differentiable
2. Has one local / global minimum
3. Converges faster

**Disadvantage: not robust to outliers.**

- The squaring **emphasises** the error: it penalises it.
- It forces the equation to accommodate the outlier points, as they've been squared to be more extreme than they actually are.
- → The best fit line has a **major shift** (line ① moves to the new line ② to reach the outliers).

## 2) MAE (mean absolute error)

$$\text{loss} = |y - \hat y| \qquad \text{cost} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat y_i|$$

- **+** Robust to outliers: not squaring, so more resistant to outliers appearing in the error, so the best fit line is less dramatically affected.
- The MAE curve is a V (**not** quadratic): a **subgradient** is needed to do gradient descent.
- **−** Time consuming.

> **Check:** in my notes the MAE loss is written $\frac{1}{2}(y - \hat y)$. MAE uses the **absolute** value, $|y - \hat y|$, with no ½ and no square. Without the absolute value, positive and negative errors would cancel out.

## 3) Huber loss

Combines MSE and MAE:

$$\text{loss} = \begin{cases} \tfrac{1}{2}(y - \hat y)^2 & \text{if } |y - \hat y| \le \delta \quad\text{(quadratic)} \\[4pt] \delta\,|y - \hat y| - \tfrac{1}{2}\delta^2 & \text{otherwise} \quad\text{(linear)} \end{cases}$$

- $\delta$ = a **hyperparameter** (you set it).
- If the error is less than the hyperparameter value, use the quadratic: no outliers, not much error.
- Otherwise, when $|y - \hat y| > \delta$, use the linear part.
- It's a **piecewise function**: quadratic → linear.

**Extra note:** $\tfrac{1}{2}(y - \hat y)^2 = y^2 + \hat y^2 - 2\hat y y$ is differentiable; $\delta|y - \hat y| - \tfrac{1}{2}\delta^2$ is linear. So it goes **from quadratic to linear**.

## Classification: cross entropy (CE)

| | Used for |
|---|---|
| **Binary CE** | two classes |
| **Categorical CE** | multiclass |

### Binary cross entropy

The logistic regression loss function:

$$\text{loss} = -y\log(\hat y) - (1 - y)\log(1 - \hat y)$$

$$\text{loss} = \begin{cases} -\log(1 - \hat y) & \text{if } y = 0 \\ -\log(\hat y) & \text{if } y = 1 \end{cases}$$

$\hat y$ = sigmoid (binary) $= \dfrac{1}{1 + e^{-x}}$

### Categorical cross entropy (multiclass)

Example dataset: features $f_1, f_2, f_3, f_4$ → output: good / bad / neutral.

**1) Encode the output into one hot encoded (OHE):**

| $f_1$ | $f_2$ | $f_3$ | Good | Bad | Neutral |
|---|---|---|---|---|---|
| | | | 1 | 0 | 0 |
| | | | 0 | 0 | 1 |
| | | | 0 | 1 | 0 |

**2) The loss:**

$$\text{loss}(x_i, y_i) = -\sum_{j=1}^{C} y_{ij} \times \ln(\hat y_{ij})$$

- $C$ = the number of categories
- $y_i = [y_{i1}, y_{i2}, y_{i3}, \dots, y_{iC}]$: $i$ = the row, $j$ = the column
- $y_{ij} = 1$ if the element is in that class, $0$ otherwise
- Because of OHE, most $y_{ij}$ are 0, and $0 \times \ln(\hat y) = 0$
- $\ln(\hat y_{ij})$ = for the highest class probability

So $-\sum y_{ij} \ln(\hat y_{ij})$ = **one value for that record**, since OHE chooses which class column counts.

### Softmax: how to get $\hat y$

Use the **softmax activation** for multiclass classification:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

- Example: output scores 10, 20, 30, 40 → the probability for the first is $\dfrac{e^{10}}{e^{10} + e^{20} + e^{30} + e^{40}}$.
- The probabilities **sum to 1**.
- Whichever class has the highest output = the classification.

$$\hat y_{ij} = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} = \text{the prediction with the highest probability}$$

More on softmax in [Softmax regression](#/t/softmax).

### A worked row

$y_{ij}$ = the true value for that class, e.g.

| Good | Bad | OK |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

For each row, softmax calculates the predicted $\hat y_{ij}$ = the probability that row $i$ belongs to each class (column $c$).

Take the third row (true class = OK), with softmax saying 0.3 for good and 0.7 for OK:

- Good: $y_{ij} = 0$, so $0 \times \ln(0.3) = 0$: not that class, it doesn't count.
- OK: $y_{ij} = 1$, so $1 \times \ln(0.7) = \ln(0.7)$: this is the term that counts.

$$\text{loss} = -\ln(0.7) = 0.36$$

- $\hat y$ close to 1 = good probability in the correct class = **low loss** = good.
- $\hat y$ low = **high loss** = bad.

> **Check:** in my notes this reads "loss value ↑". Remember the minus sign: the loss is $-\ln(0.7) = 0.36$, which is small, because 0.7 is a fairly confident correct answer. If softmax had given OK only 0.1, the loss would be $-\ln(0.1) = 2.30$.

## Quick reference

| Loss | Use for | In one line |
|---|---|---|
| MSE | regression | squares the error: smooth, fast, not robust to outliers |
| MAE | regression | absolute error: robust to outliers, needs a subgradient |
| Huber | regression | quadratic for small errors, linear past δ |
| Binary CE | two classes | $-y\log\hat y - (1-y)\log(1-\hat y)$, with sigmoid |
| Categorical CE | multiclass | $-\sum y_{ij}\ln\hat y_{ij}$, with OHE and softmax |
| Cost | a batch | the loss over many data points |
