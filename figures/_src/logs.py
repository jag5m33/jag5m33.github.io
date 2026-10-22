import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np

# ---------- 1. exponential growth and decay ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
x = np.linspace(-2, 3, 300)
ax = axes[0]
ax.plot(x, 2 ** x, color=TEAL, label="$2^x$")
ax.plot(x, np.exp(x), color=WINE, label="$e^x$")
ax.plot(x, 3 ** x, color=GOLD, label="$3^x$")
ax.scatter([0], [1], color=INK, zorder=3)
ax.annotate("every curve passes (0, 1)\nbecause $b^0 = 1$", xy=(0, 1), xytext=(-1.9, 6), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_ylim(0, 20); ax.legend(loc="upper left")
ax.set_title("Growth: base > 1", fontsize=12); ax.set_xlabel("x")
ax = axes[1]
t = np.arange(0, 6)
ax.plot(np.linspace(0, 5, 200), 100 * 0.5 ** np.linspace(0, 5, 200), color=BLUE)
ax.scatter(t, 100 * 0.5 ** t, color=BLUE, zorder=3)
for ti in t[:4]:
    ax.text(ti + 0.12, 100 * 0.5 ** ti + 3, f"{100 * 0.5 ** ti:g}", fontsize=10)
ax.set_title("Decay: halve every step, $100 \\times 0.5^t$", fontsize=12)
ax.set_xlabel("t (steps)"); ax.set_ylim(0, 110)
fig.tight_layout()
save(fig, "logs-growth-decay")

# ---------- 2. e^x and ln x mirror images ----------
fig, ax = plt.subplots(figsize=(6.2, 6.2))
x1 = np.linspace(-4, 2.2, 300)
x2 = np.linspace(0.02, 9, 400)
ax.plot(x1, np.exp(x1), color=WINE, label="$y = e^x$")
ax.plot(x2, np.log(x2), color=TEAL, label="$y = \\ln x$")
ax.plot([-4, 9], [-4, 9], color=MUTED, ls=":", lw=1.4, label="$y = x$ (the mirror)")
ax.axhline(0, color=RULE, lw=1); ax.axvline(0, color=RULE, lw=1)
ax.scatter([0, 1], [1, 0], color=INK, zorder=3)
ax.annotate("(0, 1): $e^0 = 1$", xy=(0, 1), xytext=(-3.8, 2.8), fontsize=10, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.annotate("(1, 0): $\\ln 1 = 0$", xy=(1, 0), xytext=(2.5, -2.2), fontsize=10, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.scatter([1, np.e], [np.e, 1], color=GOLD, zorder=3)
ax.annotate("", xy=(np.e, 1), xytext=(1, np.e), arrowprops=dict(arrowstyle="<->", color=GOLD, ls="--"))
ax.text(2.35, 2.45, "swap x and y", color=GOLD, fontsize=10, rotation=-45)
ax.fill_between([0, 1], -4, 0, color=LIGHT_WINE, alpha=0.5)
ax.text(0.5, -3.6, "0 < x < 1:\nln x < 0", ha="center", fontsize=9.5, color=WINE)
ax.set_xlim(-4, 9); ax.set_ylim(-4, 9); ax.set_aspect("equal")
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92), fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("$e^x$ and $\\ln x$ are reflections in $y = x$")
save(fig, "logs-mirror")

# ---------- 3. log transform before/after ----------
rng = np.random.default_rng(0)
prices = rng.lognormal(mean=np.log(300), sigma=0.6, size=1500)   # e.g. prices in £k
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
ax.hist(prices, bins=45, color=LIGHT_WINE, edgecolor=WINE, lw=0.4)
ax.axvline(np.mean(prices), color=GOLD, lw=2); ax.axvline(np.median(prices), color=BLUE, lw=2, ls="--")
ax.text(np.mean(prices) + 40, ax.get_ylim()[1] * 0.85, f"mean {np.mean(prices):.0f}", color=GOLD, fontsize=10, bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec="none"))
ax.text(np.mean(prices) + 40, ax.get_ylim()[1] * 0.72, f"median {np.median(prices):.0f}", color=BLUE, fontsize=10, bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec="none"))
import pandas as pd
ax.set_title(f"Raw: right-skewed (skew {pd.Series(prices).skew():.2f})", fontsize=12)
ax.set_xlabel("price (£ thousands)"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax = axes[1]
lp = np.log(prices)
ax.hist(lp, bins=45, color=LIGHT_TEAL, edgecolor=TEAL, lw=0.4)
ax.axvline(lp.mean(), color=GOLD, lw=2); ax.axvline(np.median(lp), color=BLUE, lw=2, ls="--")
ax.set_title(f"After ln: symmetric (skew {pd.Series(lp).skew():.2f})", fontsize=12)
ax.set_xlabel("ln(price)"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.text(lp.mean() + 0.35, ax.get_ylim()[1] * 0.92, "mean ≈ median", fontsize=10, bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec="none"))
fig.tight_layout()
save(fig, "logs-transform")
print("skews", pd.Series(prices).skew(), pd.Series(lp).skew(), prices.mean(), np.median(prices))

# ---------- 4. where you'll meet logs next: 2x2 ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax = axes[0, 0]
z = np.linspace(-7, 7, 300)
ax.plot(z, 1 / (1 + np.exp(-z)), color=TEAL)
ax.axhline(0.5, color=RULE, lw=1); ax.axvline(0, color=RULE, lw=1)
ax.scatter([0, 2], [0.5, 1 / (1 + np.exp(-2))], color=WINE, zorder=3)
ax.text(0.3, 0.42, "z = 0 → 0.5", fontsize=10)
ax.text(2.3, 0.8, "z = 2 → 0.881", fontsize=10)
ax.set_title("Sigmoid $\\sigma(z) = 1/(1+e^{-z})$", fontsize=12)
ax.set_xlabel("z"); ax.set_ylabel("probability")

ax = axes[0, 1]
p = np.linspace(0.005, 1, 300)
ax.plot(p, -np.log(p), color=WINE)
for pv in [0.9, 0.5, 0.01]:
    ax.scatter([pv], [-np.log(pv)], color=INK, zorder=3)
ax.text(0.9, 0.45, "p = 0.9\nloss 0.11", fontsize=9.5, ha="center")
ax.text(0.5, 1.05, "p = 0.5\nloss 0.69", fontsize=9.5, ha="center")
ax.text(0.05, 4.6, "p = 0.01 → loss 4.61", fontsize=9.5)
ax.set_ylim(0, 5.6)
ax.set_title("Log loss $-\\ln(p)$ for the true class", fontsize=12)
ax.set_xlabel("probability given to the right answer p"); ax.set_ylabel("loss")

ax = axes[1, 0]
p2 = np.linspace(0.01, 0.99, 300)
ax.plot(p2, np.log(p2 / (1 - p2)), color=BLUE)
ax.axhline(0, color=RULE, lw=1); ax.axvline(0.5, color=RULE, lw=1)
ax.scatter([0.8], [np.log(4)], color=WINE, zorder=3)
ax.text(0.55, 2.0, "p = 0.8 → odds 4\n→ ln 4 = 1.39", fontsize=10)
ax.set_title("Log-odds $\\ln(p/(1-p))$", fontsize=12)
ax.set_xlabel("p"); ax.set_ylabel("log-odds")

ax = axes[1, 1]
c = np.arange(1, 21)
ax.plot(c, c, color=MUTED, ls=":", label="raw count")
ax.plot(c, 1 + np.log(c), color=GOLD, marker="o", ms=4, label="1 + ln(count)")
ax.set_title("Sublinear TF: diminishing returns", fontsize=12)
ax.set_xlabel("times the word appears"); ax.set_ylabel("weight")
ax.set_ylim(0, 8); ax.legend(loc="upper right")
ax.text(19.5, 1 + np.log(20) + 0.35, f"{1 + np.log(20):.2f}", ha="center", fontsize=10, color=GOLD)
fig.tight_layout()
save(fig, "logs-next")

# ---------- 5. entropy ----------
fig, ax = plt.subplots(figsize=(7, 3.6))
p = np.linspace(0.001, 0.999, 300)
H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
ax.plot(p, H, color=TEAL)
ax.scatter([0.5, 0.9], [1, -(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))], color=WINE, zorder=3)
ax.text(0.5, 1.04, "50/50 → 1 bit", ha="center", fontsize=10)
ax.text(0.9, 0.53, "90/10 → 0.47 bits", ha="center", fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_xlabel("share of class 1 in a node"); ax.set_ylabel("entropy (bits)")
ax.set_title("Entropy with $\\log_2$: highest when the node is most mixed")
save(fig, "logs-entropy")
