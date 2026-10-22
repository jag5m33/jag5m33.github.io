import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from scipy import stats

x = np.linspace(-4, 4, 600)
pdf = stats.norm.pdf(x)

# 1. 99% band and the two 0.005 tails
z = stats.norm.ppf(0.995)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, pdf, color=INK)
m = np.abs(x) <= z
ax.fill_between(x[m], pdf[m], color=LIGHT_TEAL)
ax.fill_between(x[x >= z], pdf[x >= z], color=WINE)
ax.fill_between(x[x <= -z], pdf[x <= -z], color=WINE)
for s in (-z, z):
    ax.axvline(s, ls="--", color=MUTED, lw=1.4)
ax.text(0, 0.17, "middle 99%\n(confidence level)", ha="center", color=TEAL, fontsize=12, weight="bold")
ax.annotate("α/2 = 0.005", xy=(2.85, 0.004), xytext=(3.0, 0.12), color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE), ha="center")
ax.annotate("α/2 = 0.005", xy=(-2.85, 0.004), xytext=(-3.0, 0.12), color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE), ha="center")
ax.set_xticks([-z, -1, 0, 1, z]); ax.set_xticklabels(["−2.576", "−1", "0", "1", "+2.576"])
ax.set_xlabel("z (standard errors away from the true parameter)")
ax.set_yticks([]); ax.set_ylim(0, 0.44)
ax.set_title("Sampling distribution: 99% in the middle, 0.005 in each tail")
save(fig, "ci-tails")

# 2. SD vs SE: individuals vs means of samples of 25
rng = np.random.default_rng(0)
pop = rng.normal(68, 10, 4000)
means = rng.normal(68, 10, (2000, 25)).mean(axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(30, 106, 77)
ax.hist(pop, bins=bins, density=True, color=LIGHT_GOLD, edgecolor="white", label="individual scores (SD ≈ 10)")
ax.hist(means, bins=bins, density=True, color=TEAL, alpha=0.8, edgecolor="white", label="sample means, n = 25 (SE ≈ 10/√25 = 2)")
ax.set_xlabel("score"); ax.set_ylabel("density"); ax.set_yticks([])
ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.0, 1.0))
ax.set_ylim(0, 0.26)
ax.set_title("SD describes people; SE describes the estimate")
save(fig, "ci-sd-vs-se")

# 3. z vs t
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, pdf, color=INK, label="Z (z* = 1.960)")
for df, c in [(2, WINE), (5, ORANGE), (24, TEAL)]:
    ax.plot(x, stats.t.pdf(x, df), color=c, lw=1.8, label=f"t, df = {df}  (t* = {stats.t.ppf(0.975, df):.3f})")
ax.annotate("fatter tails →\nbigger critical value", xy=(2.7, 0.03), xytext=(2.3, 0.2),
            arrowprops=dict(arrowstyle="->", color=MUTED), color=MUTED, fontsize=10)
ax.set_yticks([]); ax.set_xlabel("score")
ax.legend(fontsize=10, loc="upper left")
ax.set_title("t has fatter tails, and becomes Z as n grows")
save(fig, "ci-z-vs-t")

# 4. 50 simulated 95% intervals for a mean
rng = np.random.default_rng(0)
mu, sigma, n = 68, 10, 25
tstar = stats.t.ppf(0.975, n - 1)
fig, ax = plt.subplots(figsize=(8, 5.2))
miss = 0
for i in range(50):
    s = rng.normal(mu, sigma, n)
    xb, se = s.mean(), s.std(ddof=1) / np.sqrt(n)
    lo, hi = xb - tstar * se, xb + tstar * se
    hit = lo <= mu <= hi
    miss += (not hit)
    c = TEAL if hit else WINE
    ax.plot([lo, hi], [i, i], color=c, lw=2 if hit else 2.8)
    ax.plot(xb, i, "o", color=c, ms=3)
ax.axvline(mu, color=INK, lw=1.5, ls="--")
ax.text(mu + 0.4, 51.2, "← true mean μ = 68 (fixed)", ha="left", fontsize=11)
ax.set_yticks([]); ax.set_ylabel("50 separate samples, n = 25 each")
ax.set_xlabel("95% confidence interval for the mean")
ax.set_ylim(-1, 53)
ax.set_title(f"{50 - miss} of 50 intervals catch μ; {miss} miss (red)")
save(fig, "ci-coverage")
print("misses", miss)

# 5. MoE vs n
fig, ax = plt.subplots(figsize=(8, 3.8))
n = np.arange(50, 3001)
for conf, c in [(0.90, GOLD), (0.95, TEAL), (0.99, WINE)]:
    ax.plot(n, stats.norm.ppf(1 - (1 - conf) / 2) * np.sqrt(0.25 / n), color=c, label=f"{conf:.0%} confidence")
ax.axhline(0.03, color=MUTED, ls=":", lw=1.2)
ax.plot(1068, 0.03, "o", color=TEAL)
ax.annotate("MoE 0.03 at 95% → n = 1068", xy=(1068, 0.03), xytext=(1400, 0.07),
            arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=11)
ax.set_xlabel("sample size n"); ax.set_ylabel("margin of error (p̂ = 0.5)")
ax.legend(fontsize=10)
ax.set_title("MoE shrinks with √n, so gains get expensive")
save(fig, "ci-moe-n")
