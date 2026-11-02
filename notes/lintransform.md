# Linear transformations & matrix-vector multiplication

## Linear transformations

**Transformation** → refers to a **function**: taking a value, spitting out another value.

Transformation suggests **movement**.

**Linear** transformation:

- → origin stays in place
- → straight lines only = no curved

Grid lines remain parallel & evenly spaced!

→ Transform via basis vectors.

$$\begin{bmatrix} -1 \\ 2 \end{bmatrix} = \vec v = -\hat\imath + 2\hat\jmath$$

Then transform the basis vectors ⇒ resulting new vector:

$$= -1(\text{transformed } \hat\imath) + 2(\text{transformed } \hat\jmath)$$

with transformed $\hat\imath = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$ and transformed $\hat\jmath = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$

$$= \begin{bmatrix} -1 \\ 2 \end{bmatrix} + \begin{bmatrix} 6 \\ 0 \end{bmatrix} = \begin{bmatrix} 5 \\ 2 \end{bmatrix} \rightarrow \text{new transformed vector tip}$$

### Basis vectors

All you need (& the scalars) to view the transformed linear 2D vector:

Where $\hat\imath$ & $\hat\jmath$ land (the new transformed basis vectors), i.e. ⇒ $\begin{bmatrix} \hat\imath \\ \hat\jmath \end{bmatrix}$ to the transformed version:

$$\begin{bmatrix} \text{transformed } \hat\imath & \text{transformed } \hat\jmath \end{bmatrix}$$

→ often no longer a vector but 2 vectors together.

New $\hat\imath = \begin{bmatrix} 3 \\ -2 \end{bmatrix}$, new $\hat\jmath = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

$$\therefore \text{vector} = \begin{bmatrix} 3 & 2 \\ -2 & 1 \end{bmatrix}$$

(left column: where $\hat\imath$ lands; right column: where $\hat\jmath$ lands)

> **Check:** this 2 × 2 grid is a **matrix** (two vectors side by side), not a vector.

If you have the old format vector $\begin{bmatrix} 5 \\ 7 \end{bmatrix}$ & the new matrix transformation $\begin{bmatrix} 3 & 2 \\ -2 & 1 \end{bmatrix}$:

$$\left(\begin{bmatrix} 3 \\ -2 \end{bmatrix} \times 5\right) + \left(7 \times \begin{bmatrix} 2 \\ 1 \end{bmatrix}\right) =$$

$$\begin{bmatrix} 15 \\ -10 \end{bmatrix} + \begin{bmatrix} 14 \\ 7 \end{bmatrix} = \begin{bmatrix} 29 \\ -3 \end{bmatrix}$$

↓ transformed vector from $\begin{bmatrix} 5 \\ 7 \end{bmatrix}$ (5 → $\hat\imath$, 7 → $\hat\jmath$)

## 2 × 2 matrix

→ Package of all the info needed to describe a linear transformation.

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

- $\begin{bmatrix} a \\ c \end{bmatrix}$ → first basis vector ($\hat\imath$)
- $\begin{bmatrix} b \\ d \end{bmatrix}$ → 2nd basis vector ($\hat\jmath$)

## Matrix-vector multiplication

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix}$$

## Summary

**Linear transformation**

- → ways to move around space
- → grid lines remain parallel & origin remains fixed.

**Matrix** = gives us a language to describe these transformations.

- → columns represent the new version, the transformed basis vectors $\hat\imath$ & $\hat\jmath$.

**Matrix-vector multiplication**

- → gives us a way to see the linear transformation & what it does to old vectors.
