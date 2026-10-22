import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats

# 1. alpha, beta, power on two overlapping curves (one-sided, alpha = 0.05, shift = 2.5)
x = np.linspace(-6.5, 6.5, 900)
shift = 2.5
cut = stats.norm.ppf(0.95)
h0, h1 = stats.norm.pdf(x), stats.norm.pdf(x, shift)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(x, h0, color=INK, label="H₀ true (no effect)")
ax.plot(x, h1, color=TEAL, label="H₁ true (real effect)")
ax.fill_between(x[x >= cut], h1[x >= cut], color=LIGHT_TEAL, label=f"power = 1 − β = {stats.norm.sf(cut - shift):.2f}")
ax.fill_between(x[x <= cut], h1[x <= cut], color=LIGHT_GOLD, alpha=0.9, label=f"β (miss) = {stats.norm.cdf(cut - shift):.2f}")
ax.fill_between(x[x >= cut], h0[x >= cut], color=WINE, label="α (false alarm) = 0.05")
ax.axvline(cut, color=MUTED, ls="--", lw=1.4)
ax.text(cut - 0.1, 0.425, "cutoff (set by α)", ha="right", color=MUTED, fontsize=10)
ax.text(cut - 0.1, -0.13, "← fail to reject H₀", ha="right", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=10)
ax.text(cut + 0.1, -0.13, "reject H₀ →", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=10)
ax.set_yticks([]); ax.set_ylim(0, 0.46)
ax.set_xlabel("test statistic", labelpad=22)
ax.legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
ax.set_xlim(-6.5, 6.5)
ax.set_title("α, β and power: one cutoff, two possible worlds")
save(fig, "hypothesis-errors")

# 2. p-value as a tail area: recycling example, z = 2.83 two-sided
z = (0.60 - 0.5) / np.sqrt(0.25 / 200)
x = np.linspace(-4, 4, 600); y = stats.norm.pdf(x)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(x, y, color=INK)
ax.fill_between(x[x >= z], y[x >= z], color=WINE)
ax.fill_between(x[x <= -z], y[x <= -z], color=WINE)
ax.axvline(z, color=WINE, lw=1.2); ax.axvline(-z, color=WINE, lw=1.2, ls=":")
ax.annotate(f"observed z = {z:.2f}", xy=(z, 0.25), xytext=(z - 2.4, 0.33), color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE))
p = 2 * stats.norm.sf(z)
ax.annotate(f"both red tails together\n= p-value = {p:.4f}", xy=(-3.05, 0.004), xytext=(-3.9, 0.15),
            arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=10.5)
ax.set_yticks([]); ax.set_xlabel("z under H₀: p = 0.5")
ax.set_title("The p-value is the area beyond your result, in the H₀ world")
save(fig, "hypothesis-pvalue")
print(z, p)
