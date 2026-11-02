# Matrix multiplication (2D & 3D)

## Matrix-matrix multiplication

We need to read it **backwards**.

$$\underset{M_2}{\begin{bmatrix} x & z \\ y & A \end{bmatrix}} \times \underset{M_1}{\begin{bmatrix} B & D \\ C & E \end{bmatrix}}$$

**stop** ← $M_2$ ←──── $M_1$ **start**

Transform from $M_1$, then from $M_1$ transform via $M_2$.

$M_2$ = the transformation (sequence) from $M_1$ to get the final setting.

### How to multiply

1. Start @ the new $M_1$ baseline $\hat\imath$ vector, $\hat\imath = \begin{bmatrix} B \\ C \end{bmatrix}$:

$$\begin{bmatrix} x & z \\ y & A \end{bmatrix} \begin{bmatrix} B \\ C \end{bmatrix} = B \times \begin{bmatrix} x \\ y \end{bmatrix} + C \times \begin{bmatrix} z \\ A \end{bmatrix} = \begin{bmatrix} Bx + Cz \\ By + CA \end{bmatrix}$$

↓ new $\hat\imath$ basis vector.

2. Likewise for $\hat\jmath$ (using $\begin{bmatrix} D \\ E \end{bmatrix}$).

> **Check:** in my notes I wrote the vector first, e.g. $\begin{bmatrix} B \\ C \end{bmatrix} \times M_2$. The usual way to write it is the matrix first, then the vector: $M_2 \begin{bmatrix} B \\ C \end{bmatrix}$. The working and the answers are the same.

### Use case

$$\underset{M_2}{\begin{bmatrix} 0 & 2 \\ 1 & 0 \end{bmatrix}} \times \underset{M_1}{\begin{bmatrix} 1 & -2 \\ 1 & 0 \end{bmatrix}}$$

Old $M_1$ basis vector $\hat\imath$:

$$\begin{bmatrix} 0 & 2 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \times \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 1 \times \begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix} \rightarrow \text{transformed } \hat\imath$$

Old $M_1$ basis vector $\hat\jmath$:

$$\begin{bmatrix} 0 & 2 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} -2 \\ 0 \end{bmatrix} = -2 \times \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0 \times \begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ -2 \end{bmatrix} \rightarrow \text{transformed } \hat\jmath$$

$$\text{Result} = \begin{bmatrix} 2 & 0 \\ 1 & -2 \end{bmatrix}$$

### ORDER matters

Think of the vector like a sequence of transformations: go through them like a sequence. Where do you end up?

## 3D linear transformations as matrix-matrix multiplication

Use 3D inputs & outputs for linear transformations ⇒

$$\underset{M_2}{\begin{bmatrix} -1 & 2 & -1 \\ 3 & 4 & -2 \\ 2 & 5 & -3 \end{bmatrix}} \times \begin{bmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{bmatrix} =$$

The columns of $M_2$ are the coordinates of the 3 basis vectors: $\hat\imath$, $\hat\jmath$, $\hat k$.

$\hat k$ → the unit vector for the $\hat k$ dimension (the 3rd direction).

**New $\hat\imath$ vector** ⇒

$$1 \times \begin{bmatrix} -1 \\ 3 \\ 2 \end{bmatrix} + 2 \times \begin{bmatrix} 2 \\ 4 \\ 5 \end{bmatrix} + 3 \times \begin{bmatrix} -1 \\ -2 \\ -3 \end{bmatrix} = \begin{bmatrix} -1 \\ 3 \\ 2 \end{bmatrix} + \begin{bmatrix} 4 \\ 8 \\ 10 \end{bmatrix} + \begin{bmatrix} -3 \\ -6 \\ -9 \end{bmatrix}$$

$$\hat\imath = \begin{bmatrix} 0 \\ 5 \\ 3 \end{bmatrix}$$

**New $\hat\jmath$ vector** =

$$\left(\begin{bmatrix} -1 \\ 3 \\ 2 \end{bmatrix} \times 4\right) + \left(\begin{bmatrix} 2 \\ 4 \\ 5 \end{bmatrix} \times 5\right) + \left(6 \times \begin{bmatrix} -1 \\ -2 \\ -3 \end{bmatrix}\right) =$$

$$\begin{bmatrix} -4 + 10 - 6 \\ 12 + 20 - 12 \\ 8 + 25 - 18 \end{bmatrix} = \begin{bmatrix} 0 \\ 20 \\ 15 \end{bmatrix} = \hat\jmath$$

**New $\hat k$** =

$$7 \times \begin{bmatrix} -1 \\ 3 \\ 2 \end{bmatrix} + 8 \times \begin{bmatrix} 2 \\ 4 \\ 5 \end{bmatrix} + 9 \times \begin{bmatrix} -1 \\ -2 \\ -3 \end{bmatrix} =$$

$$\begin{bmatrix} -7 + 16 - 9 \\ 21 + 32 - 18 \\ 14 + 40 - 27 \end{bmatrix} = \begin{bmatrix} 0 \\ 35 \\ 27 \end{bmatrix} = \hat k$$

**New transformed matrix ($M_3$)** =

$$\begin{bmatrix} 0 & 0 & 0 \\ 5 & 20 & 35 \\ 3 & 15 & 27 \end{bmatrix}$$
