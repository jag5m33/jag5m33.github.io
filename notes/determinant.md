# The determinant

Linear transformations appear to stretch or squish space. One thing that helps us understand them is **how much** they stretch or squish.

> **Definition:** the **determinant** is the factor by which the area of a given region increases or decreases: the scale by which a linear transformation changes area.

## What it measures

The basis vectors $\hat\imath$ and $\hat\jmath$ make a square with **area 1**. How much does the matrix change it? What scale?

$$\det \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = -2.0$$

![Before: î and ĵ make a square of area 1. After the matrix: î lands on [1, 3] and ĵ on [2, 4], making a parallelogram of area 2. The −2 means it's also flipped.](figures/la-det.svg)

How is an area factor negative?

- The **−** refers to an **orientation flip** (space has been flipped over).
- The **absolute value** is what the area has been scaled by, though. Here the area is scaled by 2.

## How to compute the determinant (2D)

$$\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = (a \times d) - (c \times b)$$

Multiply along the two diagonals, then subtract.

![If b and c are 0, î only stretches across by a and ĵ only up by d, so the area is the rectangle a × d.](figures/la-det-ad.svg)

$a \times d$ = the area **if** the corresponding $c$ and $b$ were 0. They usually aren't, so you need to subtract: $(a \times d) - (c \times b)$.

**Example:** $(1 \times 4) - (3 \times 2) = 4 - 6 = -2$.

## When the determinant is 0

When $\hat\imath$ lines up with $\hat\jmath$, the determinant is **0**: then it's a straight line, with no area.

![î lands on [1, 2] and ĵ on [2, 4]: both on the same line, so the whole plane is squished onto that line and the area is 0.](figures/la-det0.svg)

## 3D

Same for 3D, with $\hat\imath$, $\hat\jmath$, $\hat k$:

- The cube (initial) is transformed with the matrix.
- **Parallelepiped** = the name of the new shape (a slanted box).
- New size = $\det \times A$, where $A$ = the cube made by all the basis vectors.

> **Check:** in 3D the determinant scales **volume**, not area: the 1 × 1 × 1 cube becomes a parallelepiped with volume = det. And det = 0 in 3D means space is squished flat onto a plane, a line or a point.

## Note: multiplying determinants

$$\det(M_1 \times M_2) = \det(M_1) \times \det(M_2)$$

The determinants are multiplied as we go from the area of one to the area of the other.

> **Check:** using the matrices from [Matrix multiplication](#/t/matmul): $\det(M_2) = -2$ and $\det(M_1) = 2$, and their product $\begin{bmatrix} 2 & 0 \\ 1 & -2 \end{bmatrix}$ has det $= -4 = -2 \times 2$. ✓
