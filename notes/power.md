# Power & sample size

Power is the chance your test catches a real effect, and power analysis is how you choose the sample size before you collect any data.

## 1. The two-world picture

Every test is a contest between two bell curves on the same axis.

- **Left curve — H₀ is true.** The world where there's no effect.
- **Right curve — H₁ is true.** The world where the effect is real.

They **overlap**, because a middling result is plausible in either world. That overlap is the whole problem: where they overlap, you genuinely can't tell which world produced your result.

You place one vertical line — the **cutoff**, fixed by α. Anything to its right, you call significant and reject H₀.

![Two bell curves, H0 in dark ink and H1 in teal, with a dashed cutoff. Red: alpha, the H0 curve right of the cutoff. Gold: beta, the H1 curve left of the cutoff. Teal fill: power.](figures/power-two-worlds.svg)

In the figure, look at the dashed line first: every area is measured from it. Red is the H₀ curve past the line, gold is the H₁ curve short of the line, and the teal fill is the rest of H₁.

## 2. Alpha, beta and power — the formal definitions

| Term | What it is | On the picture |
|---|---|---|
| **Alpha (α)** | P(reject H₀ ∣ H₀ true) — a **false positive**. The false-alarm rate you set, usually 0.05 | Sliver of the **H₀ curve** to the **right** of the cutoff |
| **Beta (β)** | P(fail to reject H₀ ∣ H₀ false) — a **false negative**, a miss | Slice of the **H₁ curve** to the **left** of the cutoff |
| **Power** | P(reject H₀ ∣ H₀ false) = **1 − β** — a **true positive**, correctly catching a real effect | Rest of the **H₁ curve**, to the **right** of the cutoff. Usual target 0.80 |

(The "∣" is read "given". These are the same α and β as the error table in [Hypothesis testing](#/t/hypothesis).)

> **Trap:** β is measured against the **cutoff line**, not against where the two curves happen to overlap. It's the area of the H₁ curve sitting left of the line — nothing to do with the overlap region itself.

**Alpha guards against false alarms; power guards against misses.** Two different mistakes in opposite directions: α is "saw something that wasn't there," β is "missed something that was."

### The z-scores behind them

- **$z_{\alpha/2}$** — how many standard errors the cutoff sits from the **H₀** centre. For two-sided α = 0.05, $z_{\alpha/2}$ = 1.96.
- **$z_\beta$** — how many standard errors the same cutoff sits from the **H₁** centre. For 80% power, β = 0.20, and $z_\beta$ = 0.84 (the point with 80% of the curve above it).

## 3. Watch power fill in

![Three panels with the same cutoff. As the gap between the curves grows from 1.5 to 2.5 to 3.5 standard errors, power rises from 0.44 to 0.80 to 0.97.](figures/power-panels.svg)

Same cutoff in all three panels (one-sided α = 0.05, so the line is at 1.645). As the effect grows — or the sample grows and slims the curves — the H₁ curve pulls away from the line, its left tail (β, the miss, gold) shrinks, and the right area (power, teal) fills in: from **0.44** to **0.80** to **0.97**.

## 4. Why sample size is the only escape

Sliding the cutoff just **trades** α against β — shrink one, grow the other. It never improves both.

**Raising the sample size makes both curves skinnier** (the standard error falls as √n). Skinnier curves overlap less, so α and β can **both** fall and power climbs — without sacrificing anything. That's the one lever that beats the trade-off.

![Left: wide curves with the cutoff halfway, alpha = beta = 0.159. Right: with four times the sample both curves are half as wide, and alpha = beta = 0.023.](figures/power-sample-size.svg)

In the figure, the gap between centres is identical in both panels, and so is the cutoff (placed halfway). Quadrupling n halves the SE, and both error areas drop from 0.159 to 0.023 at once.

## 5. Effect size is the gap between the curves

> **Key idea:** Effect size = the distance between the two curve centres. A big effect pulls them far apart (little overlap, high power); a small effect leaves them nearly on top of each other (big overlap, low power).

This is why you **cannot** ask "how many samples do I need?" without first saying "to detect an effect of what size?" A tiny effect needs a huge sample to separate the curves; a big effect needs only a small one. See [Effect size](#/t/effectsize) for Cohen's d.

## 6. Power analysis and the formula

**Power analysis** — planning your sample size up front so that, if a real effect of a size you care about exists, you have a high enough chance (usually 80%) of detecting it.

Four dials, locked together — fix any three and the fourth is set: **α**, **power**, **effect size**, **sample size** (usually the one you solve for).

For comparing two means (the A/B-testing case), sample size **per group**:

$$
n = \frac{2\,(z_{\alpha/2} + z_{\beta})^2\,\sigma^2}{\Delta^2}
$$

where Δ is the difference in means you want to detect.

| Piece | Meaning | Where |
|---|---|---|
| **$z_{\alpha/2}$** | Cutoff position from the false-positive rate. α = 0.05 → 1.96 | Top |
| **$z_\beta$** | Position from the desired power. 80% power → 0.84 | Top |
| **σ²** | Variance — how spread out individuals are | Top |
| **Δ²** | Effect size, **squared** | Bottom |

### Where it comes from

**A z-score is just a count of standard deviations from a curve's centre.** $z_{\alpha/2}$ = 1.96 says the cutoff is 1.96 SEs from the H₀ centre; $z_\beta$ = 0.84 says the same line is 0.84 SEs from the H₁ centre. You **add** them because their sum is the total separation between the two centres you're paying samples to buy:

$$
\Delta = (z_{\alpha/2} + z_\beta) \times SE, \qquad SE = \sigma\sqrt{\tfrac{2}{n}}
$$

(the SE of a difference between two group means, each of size n). Square both sides and solve for n, and you get the formula above. It is the two-world picture turned into arithmetic.

### Reading the shape — the whole point

- **Δ is on the bottom and squared** → halving the effect you want to catch **quadruples** the sample. Tiny effects are brutally expensive. The single most important thing the formula says.
- **Both z-scores are on top** → more power or a stricter α both push n up. Confidence costs samples.
- **σ² on top** → noisier data needs more samples.

### Worked example

An A/B test on a score with σ = 10. You want to detect a difference of Δ = 5 points, α = 0.05 (two-sided), power 80%.

| Step | Working | Result |
|---|---|---|
| 1. $z_{\alpha/2}$ | α = 0.05 two-sided | 1.960 |
| 2. $z_\beta$ | power 0.80 → β = 0.20 | 0.842 |
| 3. Sum, squared | (1.960 + 0.842)² | 7.849 |
| 4. Top | 2 × 7.849 × 10² | 1569.8 |
| 5. Divide by Δ² | 1569.8 / 5² | 62.8 |
| 6. Round up | | **n = 63 per group** |

Now halve the effect you want to catch, Δ = 2.5: n = 1569.8 / 6.25 = 251.2 → **252 per group**. Four times as many, exactly as the shape predicts.

![Required n per group against Cohen's d on a log scale, at alpha 0.05 and power 0.80: d = 0.2 needs 394, d = 0.5 needs 64, d = 0.8 needs 26.](figures/power-n-vs-d.svg)

In the figure, the curve shoots up on the left: going from a medium effect (d = 0.5) to a small one (d = 0.2) takes you from 64 to 394 per group.

### In code

In practice you never compute it by hand — statsmodels or G*Power does it. Owning the shape means when the tool says "6,000 per arm," you know why, and that halving the effect would blow it up fourfold.

```python
import math
from statsmodels.stats.power import NormalIndPower, TTestIndPower

d = 5 / 10        # Cohen's d = delta / sigma = 5 / 10 = 0.5 (statsmodels wants it standardised)

# z-based, matches the hand formula
n_z = NormalIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8)
print(math.ceil(n_z))     # 63 per group

# t-based, slightly larger because sigma is estimated from the data
n_t = TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8)
print(math.ceil(n_t))     # 64 per group

# leave a different dial blank to solve for it: what power do 30 per group give?
pw = TTestIndPower().solve_power(effect_size=d, nobs1=30, alpha=0.05)
print(round(pw, 2))       # 0.48 -- worse than a coin flip at catching a real effect
```

`solve_power` takes the four dials; whichever one you leave out (or set to `None`) is the one it solves for. `nobs1` is the size of group 1; by default group 2 is the same size.

## 7. P-hacking — abusing the machinery

α = 0.05 is a **1-in-20 false-positive rate per test**. Run 20 tests on pure noise and about one crosses the line by chance — the chance at least one does is 1 − 0.95²⁰ = **0.64**. Cherry-pick that one and you've manufactured a fake "significant" result. That's **p-hacking**: testing many things, or peeking repeatedly, then reporting only what crossed.

> **Key idea:** The defence is exactly why power analysis is done **before** you see the data: fix your test, your α and your sample size in advance, so significance can't be shopped for after the fact.

Multiple-testing corrections (Bonferroni, false discovery rate) are the formal fix when you genuinely need many tests.

## 8. Saying it in words

> **Say it like this:** "To detect a 5-point difference with 80% power at α = 0.05, given a standard deviation of 10, we need 63 users per group."

> **Say it like this:** "The test was underpowered — with 30 per group we only had about a 48% chance of detecting a medium effect, so a non-significant result tells us little."

## 9. Traps

| Wrong | Right |
|---|---|
| β is the overlap between the curves | β is the H₁ area left of the **cutoff** |
| Move the cutoff to cut both errors | Moving the cutoff trades α for β; only larger n cuts both |
| "How many samples do I need?" | "…to detect an effect of what size?" |
| Non-significant, so no effect | Maybe just low power — check it |
| Computing power after the study to explain a null result | Power analysis is a planning tool, done before |
| Peeking at results and stopping when p < 0.05 | Fix n in advance |

## Quick reference

| Term | One line |
|---|---|
| α | False-positive rate you set. H₀ curve, right of cutoff |
| β | False-negative rate. H₁ curve, left of cutoff |
| Power | 1 − β. H₁ curve, right of cutoff. True positive |
| $z_{\alpha/2}$, $z_\beta$ | Cutoff distance in SEs, from each curve's centre (1.96, 0.84) |
| Effect size | Gap between the curve centres |
| Sample size formula | n = 2($z_{\alpha/2}$ + $z_\beta$)² σ² / Δ² per group |
| Power analysis | Solve the four dials for n, before running |
| P-hacking | Manufacturing significance by testing many / peeking |

**The order of calculation:** choose α → choose power → decide the smallest effect worth catching → estimate σ → solve for n → round up.
