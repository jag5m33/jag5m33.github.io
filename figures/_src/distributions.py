import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats

# ---------- 1. PMF vs PDF vs CDF ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
k = np.arange(0, 11)
ax = axes[0]
ax.vlines(k, 0, stats.binom.pmf(k, 10, 0.2), color=TEAL, lw=3)
ax.scatter(k, stats.binom.pmf(k, 10, 0.2), color=TEAL, zorder=3)
ax.vlines(3, 0, stats.binom.pmf(3, 10, 0.2), color=WINE, lw=4)
ax.annotate("P(X = 3) = 0.201\n(a bar height IS\na probability)", xy=(3, 0.2), xytext=(5, 0.24), fontsize=9.5, color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.set_title("PMF (discrete)", fontsize=12); ax.set_xlabel("k"); ax.set_ylabel("probability")
ax.set_ylim(0, 0.36); ax.set_xticks(range(0, 11, 2))

ax = axes[1]
xs = np.linspace(-3.5, 3.5, 400)
ax.plot(xs, stats.norm.pdf(xs), color=BLUE)
m = (xs >= 0) & (xs <= 1)
ax.fill_between(xs[m], stats.norm.pdf(xs[m]), color=LIGHT_GOLD)
ax.text(0.5, 0.12, "area\n= 0.341", ha="center", fontsize=9.5)
ax.annotate("height is a density,\nnot a probability", xy=(-1.2, stats.norm.pdf(-1.2)), xytext=(-3.6, 0.42), fontsize=9.5, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_title("PDF (continuous)", fontsize=12); ax.set_xlabel("x"); ax.set_ylabel("density")
ax.set_ylim(0, 0.52)

ax = axes[2]
ax.plot(xs, stats.norm.cdf(xs), color=GOLD)
ax.scatter([1], [stats.norm.cdf(1)], color=WINE, zorder=3)
ax.annotate("F(1) = P(X ≤ 1)\n= 0.841", xy=(1, stats.norm.cdf(1)), xytext=(-3.3, 0.72), fontsize=9.5, color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.axhline(1, color=RULE, lw=1); ax.axhline(0, color=RULE, lw=1)
ax.set_title("CDF (either kind)", fontsize=12); ax.set_xlabel("x"); ax.set_ylabel("P(X ≤ x)")
fig.tight_layout()
save(fig, "distributions-pmf-pdf-cdf")

# ---------- 2. 68-95-99.7 ----------
fig, ax = plt.subplots(figsize=(9, 4))
xs = np.linspace(-4, 4, 600); pdf = stats.norm.pdf(xs)
for k_, col in [(3, LIGHT_WINE), (2, LIGHT_GOLD), (1, LIGHT_TEAL)]:
    m = np.abs(xs) <= k_
    ax.fill_between(xs[m], pdf[m], color=col)
ax.plot(xs, pdf, color=INK)
for k_, txt, y in [(1, "68.3%", 0.47), (2, "95.4%", 0.53), (3, "99.7%", 0.59)]:
    ax.annotate("", xy=(-k_, y), xytext=(k_, y), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(0, y + 0.008, txt, ha="center", va="bottom", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec="none"))
ax.set_xticks(range(-3, 4))
ax.set_xticklabels(["μ−3σ", "μ−2σ", "μ−σ", "μ", "μ+σ", "μ+2σ", "μ+3σ"])
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_ylim(0, 0.66)
ax.set_title("The 68–95–99.7 rule for any normal distribution")
save(fig, "distributions-empirical-rule")

# ---------- 3. binomial and Poisson examples ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
k = np.arange(0, 11)
p = stats.binom.pmf(k, 10, 0.2)
cols = [WINE if kk == 3 else TEAL for kk in k]
axes[0].bar(k, p, color=cols)
axes[0].set_title("Binomial(n = 10, p = 0.2)", fontsize=12)
axes[0].set_xlabel("number of successes k"); axes[0].set_ylabel("P(X = k)")
axes[0].text(3, p[3] + 0.01, "0.201", ha="center", color=WINE, fontsize=10)
axes[0].set_xticks(k)
k2 = np.arange(0, 13)
p2 = stats.poisson.pmf(k2, 3)
cols2 = [WINE if kk == 5 else BLUE for kk in k2]
axes[1].bar(k2, p2, color=cols2)
axes[1].set_title("Poisson(λ = 3)", fontsize=12)
axes[1].set_xlabel("number of events k"); axes[1].set_ylabel("P(X = k)")
axes[1].text(5, p2[5] + 0.01, "0.101", ha="center", color=WINE, fontsize=10)
axes[1].set_xticks(k2)
fig.tight_layout()
save(fig, "distributions-binom-poisson")

# ---------- 4. CLT ----------
rng = np.random.default_rng(0)
pop = stats.expon(scale=2)        # mean 2, SD 2, strongly right-skewed
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
xs = np.linspace(0, 12, 400)
axes[0].fill_between(xs, pop.pdf(xs), color=LIGHT_WINE)
axes[0].plot(xs, pop.pdf(xs), color=WINE)
axes[0].set_title("Population\n(exponential, mean 2, SD 2)", fontsize=10.5)
axes[0].set_xlim(0, 10)
for ax, n in zip(axes[1:], [2, 5, 30]):
    means = rng.exponential(2, size=(4000, n)).mean(axis=1)
    ax.hist(means, bins=40, density=True, color=LIGHT_TEAL, edgecolor=TEAL, lw=0.4)
    se = 2 / np.sqrt(n)
    g = np.linspace(max(0, 2 - 4 * se), 2 + 4 * se, 300)
    ax.plot(g, stats.norm.pdf(g, 2, se), color=GOLD, lw=2)
    ax.set_title(f"Means of samples, n = {n}\nSE = 2/√{n} = {se:.2f}", fontsize=10.5)
    ax.set_xlim(0, 7 if n < 30 else 4)
for ax in axes:
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("value")
fig.suptitle("Central Limit Theorem: sample means turn normal (gold curve) as n grows", y=1.06, fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "distributions-clt")

# ---------- 5. SD vs SE ----------
fig, ax = plt.subplots(figsize=(9, 3.8))
xs = np.linspace(440, 560, 600)
ax.plot(xs, stats.norm.pdf(xs, 500, 10), color=MUTED, label="individual packets: SD σ = 10")
ax.fill_between(xs, stats.norm.pdf(xs, 500, 2), color=LIGHT_TEAL, alpha=0.8)
ax.plot(xs, stats.norm.pdf(xs, 500, 2), color=TEAL, label="means of 25 packets: SE = 10/√25 = 2")
ax.set_xlabel("grams"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.legend(loc="upper right", fontsize=10)
ax.set_title("Same population, two spreads: data (SD) vs the estimate (SE)")
save(fig, "distributions-sd-se")

# ---------- 6. test statistic as a z-score ----------
fig, ax = plt.subplots(figsize=(9, 3.8))
z = np.linspace(-4, 4, 600); pdf = stats.norm.pdf(z)
ax.plot(z, pdf, color=INK)
crit = stats.norm.ppf(0.975)
ax.fill_between(z, pdf, where=z >= crit, color=WINE, alpha=0.8)
ax.fill_between(z, pdf, where=z <= -crit, color=WINE, alpha=0.8)
for c in (-crit, crit):
    ax.axvline(c, color=WINE, ls="--", lw=1.4)
ax.text(crit - 0.08, 0.3, "+1.96", color=WINE, ha="right"); ax.text(-crit + 0.08, 0.3, "−1.96", color=WINE, ha="left")
ax.text(3.2, 0.06, "α/2 = 0.025", color=WINE, ha="center", fontsize=10)
ax.text(-3.2, 0.06, "α/2 = 0.025", color=WINE, ha="center", fontsize=10)
ax.axvline(2.0, color=GOLD, lw=2.6)
ax.annotate("our sample:\nz = (504 − 500) / 2 = 2.0", xy=(2.0, 0.2), xytext=(2.4, 0.36), color=GOLD, fontsize=10.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GOLD))
ax.text(0, 0.15, "95%: where z lands\nif H₀ is true", ha="center", fontsize=10, color=MUTED)
ax.set_xlabel("z (standard errors away from the hypothesised value)")
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_ylim(0, 0.46)
ax.set_title("A test statistic is a z-score on the sampling distribution")
save(fig, "distributions-test-stat")
