# Inverse matrices, column space & null space

Linear systems of equations, seen geometrically.

## A system of equations as a matrix

$$\underbrace{\begin{bmatrix} 2 & 5 & 3 \\ 4 & 0 & 8 \\ 1 & 3 & 0 \end{bmatrix}}_{M} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} -3 \\ 0 \\ 2 \end{bmatrix}$$

- $M$ = the **constant coefficients**
- $\begin{bmatrix} x \\ y \\ z \end{bmatrix}$ = the **variables**
- $\begin{bmatrix} -3 \\ 0 \\ 2 \end{bmatrix}$ = the matrix-vector product, also a **constant**

Multiple variables interact and intermingle with one another, and this is visualised by **morphing space**.

**Another example.** List of variables (a vector) and the matrix transform (the list of equations):

$$\begin{bmatrix} 6 & -3 & 2 \\ 1 & 2 & 5 \\ 2 & -8 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 7 \\ 0 \\ -2 \end{bmatrix}$$

$$\begin{aligned} 6x - 3y + 2z &= 7 \\ x + 2y + 5z &= 0 \\ 2x - 8y - z &= -2 \end{aligned}$$

That's the **system of equations**; $\begin{bmatrix} 7 \\ 0 \\ -2 \end{bmatrix}$ is the **resulting vector**.

- Each vector value is **scaled** by a constant (the matrix, showing the linear transformation).
- The resulting vector is the product of the vector $\begin{bmatrix} x \\ y \\ z \end{bmatrix}$ × the matrix (the transformation).

> **Check:** the third equation starts with $2x$ (in my notes it's written $2z$), and it's $-8y$, matching the $-8$ in the matrix. Also, $\begin{bmatrix} x \\ y \\ z \end{bmatrix}$ is the **variables** vector, not the basis vectors: the result is $x \times$ column 1 $+ \; y \times$ column 2 $+ \; z \times$ column 3, and the columns are where the basis vectors land.

## Inverse matrix

> **Definition:** $A^{-1}$ = the **inverse matrix**: it transforms back to the original.

$$A^{-1} \times A = \text{nothing} = \text{stays the same} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

## Solving $A\vec x = \vec v$

$$\underset{\text{matrix coefficients}}{A} \; \underset{\text{var.}}{\vec x} = \underset{\text{result}}{\vec v}$$

Multiply both sides by $A^{-1}$. $A^{-1} \times A$ does nothing ($= 1$), so:

$$\vec x = A^{-1} \times \vec v$$

- $A^{-1}$ = transform **in reverse**, following $\vec v$.
- This gives us $\vec x$: the single set of variables that works for this coefficient matrix and resulting vector.
- Multiply the reverse matrix transformation by $\vec v$ to **trace back the steps** to $\vec x$ (the vector that went through the $A$ transformation).

This works **when $\det(A) \neq 0$**.

> **Check:** $\vec x$ isn't the "best" set of variables, it's the **exact** answer: when $\det(A) \neq 0$ there is exactly one $\vec x$ that $A$ moves onto $\vec v$.

**When $\det(A) = 0$:**

- space is squished: it's a single line
- no area
- can't transform back: **no inverse**
- **unless** the vector $\vec v$ (the result) is also on that line: then a solution exists

> **Check:** in my notes I wrote "det(A) = 0 → no change, det(A) ≠ 0 → change". It's the other way round from what that suggests: det = 0 means the transformation **has** squished space flat (so it can't be undone), and det ≠ 0 means it can be undone with $A^{-1}$. "No change" is the identity matrix, whose det is 1. And when $\vec v$ is on the line, there are many solutions, not one.

## Rank

- Output of the transformation is on a **line** (1D) → **rank 1**
- Output of the transformation is on a **2D plane** → **rank 2**

> **Definition:** **rank** = the number of dimensions of the output of $A\vec x = \vec v$.

- A 2D matrix can have rank **2 or 1**.
- A 3D matrix can have rank **3, 2 or 1**.

> **Check:** in my notes I wrote $\vec x \times A = \vec v$. The usual way is the matrix first: $A\vec x = \vec v$.

## Column space

> **Definition:** the set of all possible outputs $A\vec v$ is the **column space** of the matrix: the range of places a vector can land after the transformation.

**Span of the columns = column space.**

- **Rank** = the number of dimensions in the column space.
- **Full rank** = rank equal to the number of columns.

## Why it matters

- When a linear transformation has an **inverse**, you can use it to **solve the system**.
- The idea of **column space** lets us identify whether a solution **even exists**: there's a solution only if $\vec v$ is in the column space.

> **Check:** null space isn't in these notes yet. In short: the **null space** is the set of all vectors that land on zero, $A\vec x = \vec 0$. If the matrix squishes space (det = 0), a whole line or plane of vectors lands on zero.
