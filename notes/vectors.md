# Vectors & scalars

## Linear algebra

- Vector rooted @ origin. Important to remember when approaching problems.

$$\begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

![The vector [2, 1]: 2 across, 1 up, starting at the origin.](figures/la-vector.svg)

> **Check:** in my notes I labelled the 2 "row" and the 1 "col". The top number is the **x** coordinate (how far across) and the bottom number is the **y** coordinate (how far up). The whole thing is one column.

- Different from the point coordinate $(2, 1)$. Vector ⇒ line.

### Adding vectors

$\vec v + \vec w$ ⇒ move the start of the 2nd to the **tip** of the first, then draw a line in between the start & end.

→ Only time in algebra where vectors stray away from the origin.

![Adding v + w: move the start of w to the tip of v.](figures/la-add.svg)

**Why do we shift $\vec w$ to meet $\vec v$?**

It's simpler. Instead of x, y across & up/down ($\vec v$), THEN x, y across & up/down ($\vec w$), we just do:

- $x_1 + x_2$ ⇒ across ($\vec v$, $\vec w$)
- $y_1 + y_2$ ⇒ down/up ($\vec v$, $\vec w$)

Just easier if we group x together & y together.

$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 + 3 \\ 2 - 1 \end{bmatrix} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$$

$$\begin{bmatrix} x_1 \\ y_1 \end{bmatrix} + \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 + x_2 \\ y_1 + y_2 \end{bmatrix}$$

> **Check:** in my notes the bottom row says $y_1 + y_1$; it should be $y_1 + y_2$.

Match up terms & add together.

## Multiplication by vector

> **Check:** here the vector is multiplied by a **number** (a scalar), not by another vector.

$\vec v \times 2$ → stretch out the vector length in general.

Multiply by a −ve ⇒ flipped around vector, then squished / stretched, e.g. $\vec v \times (-1.8)$:

![v × 2 stretches; v × (−1.8) flips and stretches.](figures/la-scale.svg)

Stretch & flip (−1.8 multiple).

- Reverse direction of vector = scaling.
- → **SCALAR** (the number the vector is multiplied with).
