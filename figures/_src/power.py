import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats
cut = stats.norm.ppf(0.95)   # one-sided alpha = 0.05 -> 1.645

def panel(ax, shift, sd=1.0, cutoff=cut, title=None, xr=(-3.5, 6)):
    x = np.linspace(*xr, 700)
    h0, h1 = stats.norm.pdf(x, 0, sd), stats.norm.pdf(x, shift, sd)
    ax.plot(x, h0, color=INK, lw=1.8)
    ax.plot(x, h1, color=TEAL, lw=1.8)
    ax.fill_between(x[x >= cutoff], h1[x >= cutoff], color=LIGHT_TEAL)
    ax.fill_between(x[x <= cutoff], h1[x <= cutoff], color=LIGHT_GOLD)
    ax.fill_between(x[x >= cutoff], h0[x >= cutoff], color=WINE)
    ax.axvline(cutoff, color=MUTED, ls="--", lw=1.2)
    ax.set_yticks([])
    pw = stats.norm.sf(cutoff, shift, sd)
    a = stats.norm.sf(cutoff, 0, sd)
    return pw, a

# 1. two worlds, labelled
fig, ax = plt.subplots(figsize=(8.5, 4))
pw, _ = panel(ax, 2.5)
ax.text(-0.2, 0.41, "H₀ world\n(no effect)", ha="center", fontsize=11)
ax.text(2.7, 0.41, "H₁ world\n(real effect)", ha="center", fontsize=11, color=TEAL)
ax.annotate("α: H₀ curve\nright of cutoff", xy=(1.85, 0.03), xytext=(4.2, 0.28), color=WINE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.annotate("β: H₁ curve\nleft of cutoff", xy=(1.1, 0.12), xytext=(-3.2, 0.2), color="#8a6d12", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#8a6d12"))
ax.text(2.6, 0.25, f"power = {pw:.2f}", ha="center", color=TEAL, fontsize=11, weight="bold")
ax.text(cut, 1.0, "cutoff", color=MUTED, fontsize=10, ha="center", va="bottom", transform=ax.get_xaxis_transform())
ax.set_ylim(0, 0.5); ax.set_xlabel("test statistic")
ax.set_title("Two worlds on one axis, one cutoff", pad=22)
save(fig, "power-two-worlds")

# 2. three panels: power fills in as the effect grows
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, s in zip(axes, [1.5, 2.5, 3.5]):
    pw, _ = panel(ax, s)
    ax.set_title(f"gap = {s} SE  →  power = {pw:.2f}", fontsize=11.5)
    ax.set_xlabel("test statistic")
fig.suptitle("Same cutoff, bigger effect: β (gold) shrinks, power (teal) fills in", y=1.04, fontsize=13, weight="bold")
save(fig, "power-panels")

# 3. larger n slims both curves: alpha AND beta fall
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
d = 2.0  # raw gap between centres, fixed
for ax, (n, lab) in zip(axes, [(1, "small n"), (4, "4× the sample (SE halves)")]):
    sd = 1 / np.sqrt(n)
    c = d / 2       # put the cutoff midway so you can see both errors
    pw, a = panel(ax, d, sd=sd, cutoff=c, xr=(-3.5, 5.5))
    ax.set_title(f"{lab}: α = {a:.3f}, β = {1 - pw:.3f}", fontsize=11.5)
    ax.set_xlabel("estimate")
fig.suptitle("A bigger sample makes both curves skinnier, so α and β fall together", y=1.04, fontsize=13, weight="bold")
save(fig, "power-sample-size")

# 4. required n per group vs effect size
from statsmodels.stats.power import TTestIndPower
an = TTestIndPower()
ds = np.linspace(0.15, 1.0, 60)
ns = [an.solve_power(effect_size=x, alpha=0.05, power=0.8) for x in ds]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(ds, ns, color=TEAL)
for x in (0.2, 0.5, 0.8):
    n = an.solve_power(effect_size=x, alpha=0.05, power=0.8)
    ax.plot(x, n, "o", color=WINE)
    ax.annotate(f"d = {x}: {int(np.ceil(n))} per group", xy=(x, n), xytext=(x + 0.03, n * 1.25), fontsize=10.5)
ax.set_yscale("log")
from matplotlib.ticker import ScalarFormatter, NullFormatter
ax.set_yticks([20, 50, 100, 200, 500]); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("effect size you want to detect (Cohen's d)")
ax.set_ylabel("n per group (log scale)")
ax.set_title("Halve the effect, roughly quadruple the sample (α = 0.05, power = 0.80)")
save(fig, "power-n-vs-d")
