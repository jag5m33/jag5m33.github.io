# Eigenvalues & eigenvectors

How to find the unit vector which **maximises variance** for [PCA](#/t/pca).

> **Definition:** an **eigenvector** $\vec v$ of a matrix $A$ is a vector that $A$ only **scales**: it stays on its own line. The **eigenvalue** $\lambda$ is how much it's scaled by.

$$A\vec v = \lambda\vec v$$

![Left: v is an eigenvector, so Av lands on the same line as v, just 1.971 times longer. Right: w is not an eigenvector, so Aw is knocked off w's line.](figures/eigen-line.svg)

$\lambda \times \vec v$ = the resulting vector. Since $\vec v$ is the same, look at $\lambda$.

## The steps

1. **Covariance matrix** between the features.
2. **Eigenvectors and eigenvalues** computed from (1): $A\vec v = \lambda\vec v$, where $A$ is the linear transformation matrix.
3. The **"largest"** eigenvector, i.e. the one with the biggest **eigenvalue**, captures the **maximum variance**.

> **Say it like this:** "Best PC line = captures max variance = highest eigenvalue."

> **Check:** in my notes I wrote "largest eigenvector (magnitude of eigenvector) = eigenvalue". The eigenvectors themselves are all **unit length** (magnitude 1), so they're all the same size. What differs is the **eigenvalue**: it's the **variance of the data along that eigenvector**. So "largest" means largest eigenvalue, not a longer vector.

## Eigen decomposition of the covariance matrix

$$\underbrace{\begin{bmatrix} \; & \; \\ \; & \; \end{bmatrix}}_{A} \times \vec v = \underbrace{\lambda}_{\text{eigenvalue}} \times \vec v$$

Whichever eigenvector has the **↑ eigenvalue** = **WINNER**.

**Max eigenvalue = max magnitude = PC1.**

## To calculate the eigenvalues and eigenvectors

### 1. Covariance

Features $x$, $y$ (output $z$):

$$\text{cov}(x, y) = \sum_{i=1}^{n} \frac{(x_i - \bar x)(y_i - \bar y)}{n - 1}$$

### 2. Covariance matrix

| | $x$ | $y$ |
|---|---|---|
| $x$ | cov($x$) | cov($x$, $y$) |
| $y$ | cov($y$, $x$) | cov($y$) |

This table is the **covariance matrix** = $A$.

> **Check:** cov($x$) with itself is just the **variance** of $x$, so the diagonal holds the variances and the off-diagonal holds the covariances. The matrix is symmetric: cov($x$, $y$) = cov($y$, $x$).

### 3. Solve $A \cdot \vec v = \lambda \cdot \vec v$

If we have 2 features ($f_1$, $f_2$), the covariance matrix is 2 × 2, so there are 2 $\lambda$ values: $\lambda_1$, $\lambda_2$. These scale values = the magnitudes.

$\uparrow \lambda$ = ↑ magnitude = ↑ eigenvalue = **PC1**.

> **Check:** $\lambda_1$ and $\lambda_2$ don't belong to $f_1$ and $f_2$. Each one belongs to an **eigenvector**: a new direction that **mixes** $f_1$ and $f_2$. That mix is the new feature (like size′ on the PCA page).

## Worked example (added, not from my notes)

Six houses: size (m²) 50, 60, 70, 80, 90, 100 and rooms 2, 2, 3, 4, 4, 5. Scale both first (so neither wins just by being in bigger units), then:

$$A = \begin{bmatrix} 1 & 0.971 \\ 0.971 & 1 \end{bmatrix}$$

| | Eigenvector | Eigenvalue | Share of variance |
|---|---|---|---|
| **PC1** | $[0.707, \; 0.707]$ (size and rooms equally) | **1.971** | **98.6%** |
| PC2 | $[0.707, \; -0.707]$ | 0.029 | 1.4% |

Check $A\vec v = \lambda\vec v$ for PC1: $A \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix} = \begin{bmatrix} 1.971 \times 0.707 \\ 1.971 \times 0.707 \end{bmatrix} = 1.971 \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix}$ ✓

So one new feature, PC1 ("overall house bigness"), keeps 98.6% of the information in the two.

> **Where it goes wrong:** without scaling, size (variance 350) swamps rooms (variance 1.5), and PC1 is almost just "size". Standardise the features before PCA.

```python
import numpy as np
X = np.c_[[50, 60, 70, 80, 90, 100], [2, 2, 3, 4, 4, 5]]
Z = (X - X.mean(0)) / X.std(0, ddof=1)   # scale first
A = np.cov(Z.T)                           # covariance matrix
vals, vecs = np.linalg.eigh(A)            # eigenvalues (ascending) and eigenvectors (columns)
print(vals[::-1] / vals.sum())            # [0.986, 0.014]: PC1 keeps 98.6%
```

## Quick reference

| Term | In one line |
|---|---|
| Eigenvector | A direction the matrix only stretches, never turns |
| Eigenvalue (λ) | How much it's stretched; in PCA, the variance along that direction |
| Covariance matrix | Variances on the diagonal, covariances off it |
| PC1 | The eigenvector with the biggest eigenvalue |

**Order:** scale → covariance matrix → eigenvalues & eigenvectors → biggest λ = PC1.
