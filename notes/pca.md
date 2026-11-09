# PCA (principal component analysis)

Principal component analysis is a way of doing **dimensionality reduction**.

## 1. The curse of dimensionality

Let's say we have a dataset with **500 features** to predict the **price of a house**: house size, no. of bedrooms, no. of bathrooms, etc. Each feature is a **dimension**.

Now train 6 ML models (M1 → M6) on more and more features:

| Model | Features | Accuracy |
|---|---|---|
| M1 | 3 | Acc 1 |
| M2 | 6 | Acc 2 ↑ |
| M3 | 15 | Acc 3 ↑↑ |
| M4 | 50 | Acc 4 ↓ |
| M5 | 100 | Acc 5 ↓↓ |
| M6 | 500 | Acc 6 ↓↓↓ |

![M1 to M3: accuracy rises as features go from 3 to 15. M4 to M6: accuracy falls once there are 50 or more features. The shape is illustrative.](figures/pca-curse.svg)

- Accuracy ↑ for the models using 3 → 15 features.
- Accuracy ↓ for features ≥ 50 → **overfitting**: ↓ bias, ↑ variance.
  - Low training error: it fits.
  - ↑ higher validation error = overfit and unable to **generalise**.

> **Definition:** the **curse of dimensionality**: past a point, adding more features (dimensions) makes model performance **degrade**.

## 2. Reduce / remove the curse of dimensionality

1. **Feature selection**: keep the **important** features.
2. **Dimensionality reduction**: **feature extraction**. From $f_1, f_2, f_3$ (→ output), derive new features/dimensions $D_1, D_2$ (→ output) that keep the info from the features.

### Why is dimensionality reduction used?

1. **Prevent** the curse of dimensionality.
2. **Improve model performance**: if we have 100 features with a model, the model has a mathematical equation, and that equation is applied to **all** 100 features = **overfits** + a **long time to train**.
3. **Visualise** the data = understand the data.

## 3. Feature selection: covariance and correlation

Input $x$, output $y$.

![Effect 1: as x goes up, y goes up. Effect 2: as x goes up, y goes down.](figures/pca-cov.svg)

### Covariance

$$\text{cov}(x, y) = \sum_{i=1}^{n} \frac{(x_i - \bar x)(y_i - \bar y)}{n - 1}$$

The $n - 1$ is the **sample adjustment**.

- **+ve** = x ↑, y ↑ (effect 1)
- **−ve** = x ↓, y ↑ (effect 2)
- **0** = no relationship

If there is a covariance relationship between x and y, x is effective: we need it.

### Correlation (Pearson)

$$r = \frac{\text{cov}(x, y)}{\sigma_x \times \sigma_y}$$

$\sigma_x \times \sigma_y$ = STD(x) × STD(y). Dividing by them **scales** it to the range **−1 → +1**.

- **+1** ← +ve correlated
- **−1** ← −ve correlated
- **0** ← not correlated

> **Check:** covariance (and Pearson's r) of 0 means no **straight-line** relationship. There could still be a curved one. More in [Correlation](#/t/correlation).

### How correlation / covariance help with feature selection

Dataset (housing): **house size** and **fountain size** (independent features) → **price** (output).

![House size vs price: a linear relationship, covariance +ve and correlation close to +1, so keep it. Fountain size vs price: correlation around 0 to 0.25, so it's not an important feature: drop it.](figures/pca-select.svg)

## 4. Feature extraction

Features ① **house size** and ② **no. of rooms** → **price** (output).

**Goal:** reduce 2 features into 1.

There is a positive correlation between **both** independent features and price, so you **can't drop one** with feature selection.

Feature extraction:

1. Take the features.
2. Apply a transformation to extract a new feature.
3. Use that **new feature** to predict the output: no longer the previous 2 features, use the new one.

**PCA: 2D → 1D.**

### Directly projecting onto the x axis

Project all points onto the x axis (house size) → 1 dimension. The spread of the data points along it: **spread ↑ = variance ↑** (the wobble).

- This **keeps** the size info, but the **rooms info is lost**.
- **BUT** by projecting onto the x axis we neglect the y-axis spread / variance = **loss of information** from 2D → 1D, so the model may diminish in performance.

### What PCA does instead

PCA makes a **new axis**, **size′**, by performing an **eigen decomposition** on a matrix (a transformation; see [Eigenvalues & eigenvectors](#/t/eigen)). Perpendicular to it is **no. of rooms′**.

![Left: projecting onto the x axis keeps house size but throws away the rooms spread. Right: projecting onto PC1 (size′) keeps most of the spread; very little is left along PC2 (rooms′).](figures/pca-project.svg)

- **Max** variance / spread captured on **size′**.
- **No / little** variance / spread on the **no. of rooms′** axis: we lose this variance, so we make it small.
- 2D → 1D = **minimise** info loss, **maximise** info / spread / variance gained. Most of the variance is captured by size′.

> **Definition:** the **main aim of PCA**: create new axes that reduce information loss. The **final goal**: find the line / axis which **maximises variance**.

| New axis | Variance captured |
|---|---|
| **PC1** (size′) | max variance |
| **PC2** | less variance |
| **PC3** | even less variance |

$$\text{var}(PC_1) > \text{var}(PC_2)$$

## 5. Picking the principal components

2D → 1D: of the **2 best PCs**:

- **PC1**: we need to pick PC1 so that the spread and variance are **maximised**.
- **PC2**: the **2nd** max variance.

E.g. **3D → 1D**: there are PC1, PC2, PC3. We want 1 dimension, so project all points onto **PC1** and get 1D points with their max variance.

**3D → 2D**: use PC1 and PC2, following the same approach.

> **Check:** each PC is at **right angles** (perpendicular) to the ones before it, so PC2 picks up only the spread PC1 missed and the new features don't overlap.

## 6. Eigen projections: the maths behind PCA

How do we know the line for PC1 is the best? It **captures the most amount of variance**. Two keys:

1. **Projection**
2. A **cost function** related to variance

### Projection

$P_1$ = a data point at $(x_1, y_1)$; $\vec P_1 = \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$ is its vector. $\vec u$ = a **unit vector** along the candidate line.

![P₁ is a data point; u is a unit vector. Dropping P₁ straight down onto u's line gives P₁′, and the dot product P₁ · u is how far along u it lands.](figures/pca-dot.svg)

Projection of $P_1$ onto $\vec u$:

$$\text{proj}_{P_1}\vec u = \frac{\vec P_1 \cdot \vec u}{\lVert \vec u \rVert}$$

$\lVert \vec u \rVert$ = the magnitude of $\vec u$ = 1 (a unit vector is always 1). Therefore:

$$\text{proj}_{P_1}\vec u = \vec P_1 \cdot \vec u = \begin{bmatrix} x_{P_1} \\ y_{P_1} \end{bmatrix} \cdot \begin{bmatrix} x_u \\ y_u \end{bmatrix} = (x_{P_1} \cdot x_u) + (y_{P_1} \cdot y_u)$$

> **Check:** in my notes the second bracket says $x_{P_1} \cdot y_u$. It's $y_{P_1} \cdot y_u$: x times x, plus y times y.

- The dot product of data point $P_1$ and $\vec u$ (the point where the unit vector stops) = $P_1'$: the projected $P_1$ value on the $\vec u$ vector.
- After $P_1$ we do the rest of the points: $P_1', P_2', P_3', \dots$ = all the projected values of the data points onto $\vec u$.
- The result of a dot product is **always a scalar**; it lies along $\vec u$.
- That scalar = the **distance along the $\vec u$ axis**.

### Cost function

The scalar values are used to calculate the **variance**:

$$\text{var} = \sum_{i=1}^{n} \frac{(x_i - \bar x)^2}{n}$$

where the $x_i$ here are the projected values $P_1', P_2', \dots$

> **Definition:** the **goal** = find the unit vector with the **most amount of variance**. **Max of variance = the cost function.**

> **Check:** a "cost" is usually something you **minimise** (like error). Here it's an objective you **maximise**: pick the $\vec u$ whose projections have the biggest variance. The $n$ vs $n - 1$ difference doesn't change which line wins.

## 7. Eigenvectors and eigenvalues: how to find that unit vector

How to find the unit vector which maximises variance:

1. **Covariance matrix** between the features.
2. **Eigenvectors and eigenvalues** computed from (1): $A\vec v = \lambda\vec v$.
3. The eigenvector with the **largest eigenvalue** captures the **maximum variance**.

> **Say it like this:** "The best PC line captures the max variance: it's the eigenvector with the **highest eigenvalue**."

The full working, with a small example, is on [Eigenvalues & eigenvectors](#/t/eigen).

## Quick reference

| Term | In one line |
|---|---|
| Curse of dimensionality | Too many features → overfitting, accuracy falls |
| Feature selection | Keep the important features, drop the rest |
| Feature extraction | Make new features that combine the old ones |
| PC1 | The new axis with the most variance |
| PC2, PC3, … | Next most variance, each at right angles to the last |
| Projection | Dot product of a point with a unit vector: its distance along that axis |
| Eigenvalue | The variance captured along that eigenvector |

**Order:** (scale the features) → covariance matrix → eigenvectors & eigenvalues → sort by eigenvalue → keep the top PCs → project the data onto them.
