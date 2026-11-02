# Power & sample size

Power answers one question before you collect any data: **if the effect is real, how likely is my test to catch it?**

**On this page**

- The two-world picture: α, β and power
- Watch power fill in as the curves pull apart
- Why a bigger sample is the only escape
- Effect size is the gap between the curves
- Power analysis: four dials, solve for n
- P-hacking: why you fix it all in advance

## The two-world picture

> **Definition:** **power** is the chance your test catches an effect that is really there. Power = 1 − β, and the usual target is **80%**.

Every test is a contest between two bell curves on the same axis:

- **Left curve: H₀ is true.** The world with no effect.
- **Right curve: H₁ is true.** The world where the effect is real.

They overlap, because a middling result is plausible in either world: there, you can't tell which world produced it. You draw one line, the **cutoff**, fixed by α. Anything right of it, you call significant.

![Look at the dashed cutoff first: every area is measured from it. Red: α. Gold: β. Teal: power.](figures/power-two-worlds.svg)

| Term | The mistake | On the picture |
|---|---|---|
| **α** | False alarm: "saw something that wasn't there" | H₀ curve, right of the cutoff |
| **β** | Miss: "missed something that was there" | H₁ curve, left of the cutoff |
| **Power** | Not a mistake: a real effect, caught | H₁ curve, right of the cutoff |

Alpha guards against false alarms; power guards against misses. Same α and β as in [Hypothesis testing](#/t/hypothesis).

> **Where it goes wrong:** β is measured against the **cutoff line**, not against where the two curves overlap. It's the H₁ curve's area left of the line.

## Watch power fill in

> **Definition:** the further the H₁ curve sits from the cutoff, the **higher the power**.

![Same cutoff in all three panels. As the H₁ curve pulls away, the gold miss shrinks and teal power fills in: 0.44, 0.80, 0.97.](figures/power-panels.svg)

Two things pull it away: a **bigger effect**, or a **bigger sample**.

## Why sample size is the only escape

> **Definition:** α and β **trade off**: slide the cutoff and you shrink one only by growing the other.

A bigger sample makes **both curves skinnier** (the standard error shrinks). Skinnier curves overlap less, so α and β can **both** fall. That's the one lever that beats the trade-off.

![Same gap, same cutoff. Four times the sample halves the width, and α and β both drop from 0.159 to 0.023.](figures/power-sample-size.svg)

## Effect size is the gap

> **Definition:** in this picture, **effect size** is the distance between the two curve centres (Cohen's d from [Effect size](#/t/effectsize)).

- Big effect: curves far apart, little overlap, high power.
- Small effect: curves nearly on top of each other, low power.

So you **cannot** ask "how many samples do I need?" without first saying "to detect an effect of what size?"

## Power analysis

> **Definition:** **power analysis** is planning your sample size up front so that, if a real effect of a size you care about exists, you have a high enough chance (usually 80%) of catching it.

Four dials, locked together. Fix any three and the fourth is set:

1. **α** (usually 0.05)
2. **Power** (usually 0.80)
3. **Effect size**: the smallest difference worth catching
4. **Sample size**: usually the one you solve for

**Worked example.** An A/B test on a score with SD 10. You want to catch a 5-point difference, so d = 5 / 10 = 0.5. At α = 0.05 and 80% power you need **64 users per group**.

Halve the difference (d = 0.25) and you need **253 per group**: about four times as many.

Why four times? Half the gap means each curve must be half as wide to separate again, and width only shrinks with the square root of the sample. **Halve the effect → 4× the sample.** Tiny effects are very expensive.

![Sample per group needed at α = 0.05 and 80% power. The curve shoots up on the left: d = 0.8 needs 26, d = 0.5 needs 64, d = 0.2 needs 394.](figures/power-n-vs-d.svg)

In practice a tool does it (statsmodels, G*Power). Your job is knowing why.

```python
from statsmodels.stats.power import TTestIndPower

# leave out the dial to solve for
tool = TTestIndPower()
n = tool.solve_power(effect_size=0.5, alpha=0.05, power=0.8)   # 63.8 -> 64 per group
pw = tool.solve_power(effect_size=0.5, nobs1=30, alpha=0.05)   # 0.48 with 30 per group
```

> **Where it goes wrong:** "not significant, so no effect" ignores power. With 30 per group you'd catch a medium effect only 48% of the time, worse than a coin flip.

> **Say it like this:** "To detect a 5-point difference with 80% power at α = 0.05, given a standard deviation of 10, we need 64 users per group. Halving the difference would need about four times that."

## P-hacking

> **Definition:** **p-hacking** is manufacturing a "significant" result by testing many things, or peeking repeatedly, and reporting only what crossed the line.

α = 0.05 is a 1-in-20 false-alarm rate **per test**. Run 20 tests on pure noise and there's a **64%** chance at least one crosses by luck.

The defence is why power analysis comes **before** the data: fix your test, α and sample size in advance, so significance can't be shopped for afterwards.

## Quick reference

| Term | In one line |
|---|---|
| α | False alarm. H₀ curve, right of the cutoff |
| β | Miss. H₁ curve, left of the cutoff |
| Power | 1 − β. Chance of catching a real effect. Aim for 80% |
| Effect size | Gap between the curve centres |
| Four dials | α, power, effect size, sample size: fix three, solve the fourth |
| Halve the effect | 4× the sample |
| P-hacking | Testing many or peeking, then reporting the winner |

**Order:** choose α → choose power → decide the smallest effect worth catching → solve for n → round up.
