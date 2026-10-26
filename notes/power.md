# Power & sample size

Power is the chance your test catches a real effect, and power analysis is how you choose the sample size before you collect any data.

## The two-world picture

Every test is a contest between two bell curves on the same axis.

- **Left curve: H₀ is true.** The world where there's no effect.
- **Right curve: H₁ is true.** The world where the effect is real.

They **overlap**, because a middling result is plausible in either world. Where they overlap, you genuinely can't tell which world produced your result.

You place one vertical line, the **cutoff**, fixed by α. Anything to its right, you call significant.

![Two bell curves, H0 in dark ink and H1 in teal, with a dashed cutoff. Red: alpha, the H0 curve right of the cutoff. Gold: beta, the H1 curve left of the cutoff. Teal fill: power.](figures/power-two-worlds.svg)

In the figure, look at the dashed line first: every area is measured from it.

| Term | What it is | On the picture |
|---|---|---|
| **Alpha (α)** | A **false positive**: "saw something that wasn't there". You set it, usually 0.05 | Red: the **H₀ curve** right of the cutoff |
| **Beta (β)** | A **false negative**: "missed something that was there" | Gold: the **H₁ curve** left of the cutoff |
| **Power** | $1 - \beta$: correctly catching a real effect. Usual target 0.80 | Teal: the rest of the **H₁ curve**, right of the cutoff |

These are the same α and β as the error table in [Hypothesis testing](#/t/hypothesis).

> **Trap:** β is measured against the **cutoff line**, not the overlap region. It's the part of the H₁ curve sitting left of the line.

## Watch power fill in

![Three panels with the same cutoff. As the gap between the curves grows from 1.5 to 2.5 to 3.5 standard errors, power rises from 0.44 to 0.80 to 0.97.](figures/power-panels.svg)

Same cutoff in all three panels. As the H₁ curve pulls away from the line, the gold miss area shrinks and the teal power area fills in: from **0.44** to **0.80** to **0.97**.

## Why sample size is the only escape

Sliding the cutoff just **trades** α against β: shrink one, grow the other.

**A bigger sample makes both curves skinnier** (the standard error shrinks). Skinnier curves overlap less, so α and β can **both** fall. That's the one lever that beats the trade-off.

![Left: wide curves with the cutoff halfway, alpha = beta = 0.159. Right: with four times the sample both curves are half as wide, and alpha = beta = 0.023.](figures/power-sample-size.svg)

In the figure, the gap between centres and the cutoff are the same in both panels. Four times the sample halves the width, and both error areas drop from 0.159 to 0.023 at once.

## Effect size is the gap between the curves

> **Key idea:** Effect size = the distance between the two curve centres. A big effect pulls them far apart (high power); a small effect leaves them nearly on top of each other (low power).

So you **cannot** ask "how many samples do I need?" without first saying "to detect an effect of what size?" See [Effect size](#/t/effectsize) for Cohen's d.

## Power analysis

**Power analysis**: planning your sample size up front so that, if a real effect of a size you care about exists, you have a good chance (usually 80%) of catching it.

Four dials, locked together. Fix any three and the fourth is set:

1. **α** (usually 0.05)
2. **power** (usually 0.80)
3. **effect size** (the smallest difference worth catching)
4. **sample size** (usually the one you solve for)

**Worked example.** An A/B test on a score with SD 10. You want to catch a 5-point difference, so Cohen's d = 5 / 10 = 0.5. With α = 0.05 and 80% power, you need about **64 users per group**.

Now halve the difference you want to catch (d = 0.25): you need about **253 per group**. Roughly four times as many.

> **Key idea:** halving the effect you want to catch roughly **quadruples** the sample. Tiny effects are very expensive to detect.

![Required n per group against Cohen's d on a log scale, at alpha 0.05 and power 0.80: d = 0.2 needs 394, d = 0.5 needs 64, d = 0.8 needs 26.](figures/power-n-vs-d.svg)

In the figure, the curve shoots up on the left: going from a medium effect (d = 0.5) to a small one (d = 0.2) takes you from 64 to 394 per group.

You never do this by hand: statsmodels or G*Power does it.

```python
import math
from statsmodels.stats.power import TTestIndPower

d = 5 / 10                      # Cohen's d: difference / SD = 0.5

# Leave out sample size -> it solves for sample size
n = TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8)
print(math.ceil(n))             # 64 per group

# Leave out power -> what power do 30 per group give?
pw = TTestIndPower().solve_power(effect_size=d, nobs1=30, alpha=0.05)
print(round(pw, 2))             # 0.48: worse than a coin flip
```

## P-hacking

α = 0.05 is a **1-in-20 false-positive rate per test**. Run 20 tests on pure noise and the chance at least one crosses the line is about **64%**. Cherry-pick that one and you've manufactured a fake result. That's **p-hacking**: testing many things, or peeking repeatedly, then reporting only what crossed.

> **Key idea:** this is why power analysis is done **before** you see the data: fix your test, your α and your sample size in advance, so significance can't be shopped for afterwards.

## Saying it in words

> **Say it like this:** "To detect a 5-point difference with 80% power at α = 0.05, given a standard deviation of 10, we need 64 users per group."

> **Say it like this:** "The test was underpowered: with 30 per group we only had about a 48% chance of detecting a medium effect, so a non-significant result tells us little."

## Traps

| Wrong | Right |
|---|---|
| β is the overlap between the curves | β is the H₁ area left of the **cutoff** |
| Move the cutoff to cut both errors | That trades α for β; only a larger sample cuts both |
| "How many samples do I need?" | "...to detect an effect of what size?" |
| Non-significant, so no effect | Maybe just low power: check it |
| Peeking and stopping when p < 0.05 | Fix the sample size in advance |

## Quick reference

| Term | One line |
|---|---|
| α | False positive. H₀ curve, right of cutoff |
| β | False negative. H₁ curve, left of cutoff |
| Power | 1 − β. H₁ curve, right of cutoff |
| Effect size | Gap between the curve centres |
| Four dials | α, power, effect size, sample size: fix three, solve the fourth |
| Halve the effect | about 4× the sample |
| P-hacking | Manufacturing significance by testing many or peeking |

**Order of calculation:** choose α → choose power → decide the smallest effect worth catching → solve for n → round up.
