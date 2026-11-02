# Linear combinations, span & basis vectors

## Linear combinations, span, basis vectors

Vector ⇒ $\begin{bmatrix} \hat\imath \\ \hat\jmath \end{bmatrix}$ → "basis of coordinate system". Think of the numbers actually as **scalars** of this format.

e.g. $\begin{bmatrix} 3 \\ -2 \end{bmatrix}$ is actually $\begin{bmatrix} 3\hat\imath \\ -2\hat\jmath \end{bmatrix}$ ⇒ so technically 3 & −2 are **scalars**.

→ Scaled vector. Sum of scaled vectors $= (3)\hat\imath + (-2)\hat\jmath$

When you look at vectors, the numbers are those scalars **scaling** the basis vectors, e.g. ⇒

$$\begin{bmatrix} -5 \\ 2 \end{bmatrix} = \begin{bmatrix} -5\hat\imath \\ 2\hat\jmath \end{bmatrix}$$

**So why choose $\hat\imath$ & $\hat\jmath$?** They could have been any length, so why these ones?

→ By changing the basis vectors' fundamental length, all subsequent "scalars" would be different.

Any time we use a numerical vector, it relies on the implicit choice of what basis vectors we're using.

### Linear combination

Any time you're scaling & adding 2 vectors:

$$5\vec v + 2\vec w = \text{called a } \textbf{linear combination}$$

(5 and 2 → the scalars.)

They are scalars, they change. If one stays the same, the other moves across a linear line set by the constant vector, e.g.:

![3v + 2w: keep 2w constant and change the scalar on v; the tip moves along a straight line.](figures/la-combo.svg)

### Span

The **span** of $\vec v$ & $\vec w$ is the set of all their linear combinations, done only with vector addition & scalar multiplication.

$\vec v + \vec w$ = resulting vector whose **tip** is the focus.

Think of it like a scatter plot: a point (tip) where the arrow connected to the origin is there but doesn't get drawn = for simplicity.

- Span of most 2D vectors = the whole 2D space.
- If the vectors line up on top of each other, then it's the line itself & all possible points on it.

## Span in 3D

→ Their span ($\vec v$ & $\vec w$) in 3D = all possible linear combinations of both vectors.

→ All possible combinations you get by scaling each of the 2 in some way & getting the addition sum.

## 2D and 3D planes

2D vector span on a 3D plane = across a flat sheet → hyperplane.

![Two vectors in 3D: their span is a flat sheet through the origin.](figures/la-plane.svg)

→ They go on the hyperplane.

> **Check:** in 3D this flat sheet is usually just called a **plane**. "Hyperplane" is the general name for the same idea in any number of dimensions, so both are fine.

If all 3D vectors, all 3 change ⇒

Linear combination of 3 vectors is described the same way as it is for 2:

$$a\vec v + b\vec w + c\vec u$$

Choose 3 different scalars, scale each vector & add together.

The hyperplane moves along the 3rd vector; if it changes, it shifts across all of space.

### Linearly dependent

If vectors can be removed without reducing the span = they are **linearly dependent**.

$$3\vec v + 4\vec w + 9\vec z$$
$$2\vec v + 5\vec w + 9\vec z$$

→ same, no change. ↓ The 3D vectors scale across the hyperplane sheet.

Linearly dependent as $\vec z$ is the same = no change.

> **Check:** the clearer way to say it: $\vec z$ is linearly dependent when it already sits on the sheet that $\vec v$ and $\vec w$ span (for example $\vec z = 2\vec v + \vec w$). Then adding any amount of $\vec z$ keeps you on the same sheet, so removing it doesn't shrink the span.

**Basis vectors** = a set of linearly independent vectors that span the full space.
