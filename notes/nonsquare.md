# Non-square matrices

Non-square matrices as transformations between dimensions.

> **Definition:** **columns = the dimensions you start with** (input). **Rows = the dimensions you end up in** (output).

| | Tells you | Example: 3 rows × 2 columns |
|---|---|---|
| **Columns** | the dimensions you **start** with | 2 columns → start in **2D** |
| **Rows** | the dimensions you come **out** in | 3 rows → end in **3D** |

> **Say it like this:** "columns in, rows out." Read the shape as rows × columns = **out × in**, so a 3 × 2 matrix takes 2D in and gives 3D out.

## Start in 2D, go to 3D

$$\begin{bmatrix} 3 & 1 \\ 4 & 1 \\ 5 & 9 \end{bmatrix} \Rightarrow \text{2 columns (start in 2D)}, \text{ 3 rows (out in 3D)}$$

- But only **2 axes** are used.
- Column 1 = where $\hat\imath$ lands; column 2 = where $\hat\jmath$ lands.
- There's no $\hat k$ column: the outputs lie across a **plane** of values inside 3D.

![î lands on [3, 4, 5] and ĵ on [1, 1, 9]. Two input directions can only make a flat sheet, so every output sits on one plane through 3D.](figures/la-nonsquare.svg)

> **Check:** this matrix has rank 2: its column space is a 2D plane inside 3D space, so it's still full rank (2 columns, rank 2).

## Start in 3D, go to 2D

$$\begin{bmatrix} 3 & 4 & 5 \\ 1 & 1 & 9 \end{bmatrix} \Rightarrow \text{3 columns (start in 3D)}, \text{ 2 rows (out in 2D)}$$

Start in 3D ($\hat\imath$, $\hat\jmath$, $\hat k$), transform onto a 2D plane: only 2 coordinates, no 3rd.

**Example.** Input 3D, output 2D:

$$\begin{bmatrix} 4 & 4 & 5 \\ 1 & 1 & 9 \end{bmatrix} \begin{matrix} \leftarrow x \\ \leftarrow y \end{matrix} \qquad \text{3 columns in (3D)}, \text{ 2 rows out (2D)}$$

The columns are $\hat\imath$, $\hat\jmath$, $\hat k$.

How do 3 points go to 2D (2 coordinates)? The **rows** are the $(x, y)$ coordinates:

- read across the rows, $(4, 4, 5)$ and $(1, 1, 9)$ are 3D
- read down the columns, in 2D: $\hat\imath \to (4, 1)$, $\hat\jmath \to (4, 1)$, $\hat k \to (5, 9)$

> **Check:** in my notes the 3D → 2D rows are labelled "1D" and "2D". Each row is one **output coordinate**: row 1 gives the output $x$, row 2 gives the output $y$. Also, in the second example $\hat\imath$ and $\hat\jmath$ both land on $(4, 1)$, so they line up, but $\hat k$ at $(5, 9)$ points elsewhere, so the outputs still fill the whole 2D plane (rank 2).
