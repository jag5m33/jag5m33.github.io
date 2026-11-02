# Correlation

Correlation answers one question: **when one variable goes up, does the other tend to go up (or down) too, and how tightly?**

**On this page**

- r: one number for strength and direction
- Pearson vs Spearman: raw values or ranks
- r = 0 doesn't mean "no link"
- r²: the one to phrase carefully
- Adjusted R²: the noise tax
- Correlation is not causation

## r, the correlation coefficient

> **Definition:** **r** (Pearson's correlation) is a number from −1 to +1 that captures the **strength and direction** of a straight-line link between two variables.

- **+1:** a perfect upward line.
- **0:** no straight-line link, a shapeless cloud.
- **−1:** a perfect downward line.

The sign is the direction. The size is how tightly the points hug a line.

![Five clouds, r from −0.9 to +0.9. The further r is from 0, the tighter the line.](figures/correlation-grid.svg)

**How it works, in words:** points above average on both variables (or below on both) push r up. Points above on one and below on the other pull it down. The total is scaled to land between −1 and +1.

**Worked example.** Six students: hours studied 1 to 6, exam scores 52, 55, 61, 60, 68, 72.

![Dotted lines mark the averages. Five teal points push r up; the one red point pulls it down a little.](figures/correlation-worked.svg)

Almost every point is above average on both or below on both, so **r = 0.97**: a strong upward link.

> **Where it goes wrong:** r is **not** the slope. A shallow line and a steep line can both have r = 0.9. r measures tightness to a line, not steepness.

## Pearson vs Spearman

> **Definition:** **Pearson** works on the raw values. **Spearman** works on the **ranks** (smallest = 1, next = 2, and so on). Spearman is just Pearson run on the ranks instead of the raw numbers.

Reach for Spearman for **curves that keep rising**, **outliers** (a rank can't be more extreme than "the biggest") and **ratings** like 1 to 5 stars.

Students again: the scores rank 1, 2, 4, 3, 5, 6. Only two swap, so **Spearman = 0.94**.

![Left: a clean line, Pearson 0.98, Spearman 0.97. Right: one reading mistyped as 60. Pearson drops to 0.67; Spearman stays at 0.97.](figures/correlation-outlier.svg)

> **Where it goes wrong:** it is **not** "Pearson linear, Spearman non-linear". Both measure a linear-ish link. Spearman just copes with curves that keep going one way, and with outliers.

## r = 0 doesn't mean "no link"

> **Definition:** **r = 0** means no **straight-line** link. There can still be a strong link of another shape.

![A U-shape: y depends completely on x, yet Pearson and Spearman are both about 0.](figures/correlation-curve.svg)

Neither Pearson nor Spearman catches a relationship that turns back on itself. **Always plot the data first.**

## r², the one to phrase carefully

> **Definition:** **r²** is r squared: **the share of the variation in the outcome that the other variable explains**.

Pick one variable as the outcome. r² is the slice of its wobble that the other one explains.

**Height and weight.** r = 0.7 → r² = 0.49 → height accounts for about **49% of why people's weights differ**. The other 51% is diet, genes and noise.

> **Where it goes wrong:** r = 0.7 is **not** "70% related", and r² is not "the share of the correlation". Also, halving r quarters r²: r = 0.5 → r² = 0.25, so 25% explained and 75% down to everything else.

## Adjusted R², the noise tax

> **Definition:** **adjusted R²** is R² with a **noise tax** charged for every predictor in the model.

Plain R² only ever goes up: even a pure-noise column fits a scrap of noise by luck.

A junk column gains about the tax, so adjusted R² stays flat. A real column gains far more, so it still rises. More in [Regression metrics](#/t/regmetrics).

| Junk columns added | Plain R² | Adjusted R² |
|---|---|---|
| 0 | 0.494 | 0.483 |
| 4 | 0.548 | 0.497 |
| 8 | 0.587 | 0.494 |

Plain R² climbed on nothing but noise. Adjusted R² never moved.

> **Say it like this:** "Plain R² asks how well did I fit. Adjusted R² asks **how well did I fit, given how many chances I gave myself**."

## Correlation is not causation

> **Definition:** a **confounder** is a hidden third variable that drives both things you measured, making them correlate.

A strong r says two things move together, not that one causes. Ice-cream sales and drownings correlate because hot weather drives both. The arrow might also point the other way, or it might be chance.

**Simpson's paradox:** a trend inside every group flips when the groups are pooled.

![Each coloured group slopes down; pooled together (dashed line), they slope up.](figures/correlation-simpson.svg)

You call these, you don't memorise them. In his US Accidents EDA, Jag used `.corr()` to spot that wind chill and temperature were near-duplicates ([dcdm.ipynb](https://github.com/jag5m33/py_acc/blob/main/notebooks/dcdm.ipynb)).

```python
from scipy import stats

hours  = [1, 2, 3, 4, 5, 6]
scores = [52, 55, 61, 60, 68, 72]
r, p = stats.pearsonr(hours, scores)      # 0.973
rho, _ = stats.spearmanr(hours, scores)   # 0.943, Pearson on the ranks
print(r ** 2)                             # 0.947, share of variation explained
```

> **Say it like this:** "Hours and score are strongly correlated, r = 0.97, so hours explain about 95% of the variation in scores here. That's an association, not proof that studying caused the higher scores."

## Quick reference

| Term | In one line |
|---|---|
| r | Strength and direction of a straight-line link, −1 to +1 |
| Spearman | Pearson on the ranks: curves that keep rising, outliers, ratings |
| r = 0 | No straight-line link; plot it |
| r² | Share of the variation in the outcome the other variable explains |
| Adjusted R² | R² minus a noise tax per predictor |
| Confounder | Hidden third variable driving both |

**Order:** plot it → r → r² → say it in words (association, not causation).
