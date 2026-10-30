import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
import warnings; warnings.simplefilter('ignore')
from numpy.polynomial import polynomial as P

def truth(x): return np.sin(2 * np.pi * x) * 1.0 + 0.5 * x
SIG = 0.3

# 1. dartboard 2x2
rng = np.random.default_rng(0)
fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5))
setups = [((0, 0), 0.12, "Low bias, low variance", "the goal", TEAL),
          ((0, 0), 0.45, "Low bias, high variance", "overfitting", ORANGE),
          ((0.62, 0.35), 0.10, "High bias, low variance", "underfitting", BLUE),
          ((0.6, 0.4), 0.45, "High bias, high variance", "worst of both", WINE)]
for ax, (c, s, t, sub, col) in zip(axs.flat, setups):
    for r, fc in zip([1.0, 0.75, 0.5, 0.25], ["#efe6cf", "#e6dab8", "#dccd9f", "#d1bf86"]):
        ax.add_patch(plt.Circle((0, 0), r, color=fc, ec=MUTED, lw=0.8))
    ax.add_patch(plt.Circle((0, 0), 0.06, color=WINE))
    z = rng.normal(0, 1, (14, 2)); z = (z - z.mean(0)) / z.std(0)
    pts = np.array(c) + s * 0.8 * z
    ax.scatter(pts[:, 0], pts[:, 1], s=34, color=col, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(*pts.mean(0), marker="x", s=80, color=INK, zorder=4)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{t}\n({sub})", fontsize=11)
fig.tight_layout(); save(fig, "biasvar-dartboard")

# 2. underfit / good / overfit
rng = np.random.default_rng(1)
xtr = np.sort(rng.uniform(0, 1, 15)); ytr = truth(xtr) + rng.normal(0, SIG, 15)
xva = np.sort(rng.uniform(0, 1, 15)); yva = truth(xva) + rng.normal(0, SIG, 15)
xs = np.linspace(0, 1, 300)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (d, t, col) in zip(axs, [(1, "Degree 1: underfit\n(high bias)", BLUE), (4, "Degree 4: about right", TEAL), (12, "Degree 12: overfit\n(high variance)", WINE)]):
    c = np.polyfit(xtr, ytr, d)
    tr = np.mean((np.polyval(c, xtr) - ytr) ** 2); va = np.mean((np.polyval(c, xva) - yva) ** 2)
    ax.plot(xs, truth(xs), color=MUTED, ls=":", lw=1.5, label="true pattern")
    ax.plot(xs, np.clip(np.polyval(c, xs), -3, 3), color=col, label="model")
    ax.scatter(xtr, ytr, s=25, color=INK, zorder=3, label="training points")
    ax.scatter(xva, yva, s=25, facecolor="none", edgecolor=ORANGE, zorder=3, label="validation points")
    ax.set_title(t, fontsize=11); ax.set_ylim(-2, 2.6); ax.set_xlabel("x")
    vs = f"{va:.2f}" if va < 10 else f"{va:.0f}"
    ax.text(0.03, 0.04, f"train MSE {tr:.2f}\nval MSE {vs}", transform=ax.transAxes, fontsize=10)
    print(d, round(tr, 3), round(va, 3))
axs[0].set_ylabel("y"); axs[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); save(fig, "biasvar-fits")

# 3. complexity curve: fixed x design, many noisy training sets per degree
rng = np.random.default_rng(2)
degs = np.arange(0, 13); ntr = 14; R = 400
x = np.linspace(0, 1, ntr)                       # same x every time; only the noise changes
xgrid = np.linspace(0.05, 0.95, 60)
b2, var, tre, vae = [], [], [], []
for d in degs:
    preds = []; trs = []; vas = []
    for r in range(R):
        y = truth(x) + rng.normal(0, SIG, ntr)
        c = np.polyfit(x, y, d)
        pg = np.polyval(c, xgrid); preds.append(pg)
        trs.append(np.mean((np.polyval(c, x) - y) ** 2))
        yv = truth(xgrid) + rng.normal(0, SIG, len(xgrid))
        vas.append(np.mean((pg - yv) ** 2))
    preds = np.array(preds)
    b2.append(np.mean((preds.mean(0) - truth(xgrid)) ** 2)); var.append(np.mean(preds.var(0)))
    tre.append(np.mean(trs)); vae.append(np.mean(vas))
b2, var, tre, vae = map(np.array, (b2, var, tre, vae))
best = degs[np.argmin(vae)]
print("best degree", best, "val", vae.round(3))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(degs, tre, color=TEAL, label="training error")
ax.plot(degs, vae, color=WINE, label="validation error")
ax.plot(degs, b2, color=BLUE, ls="--", lw=1.6, label="bias")
ax.plot(degs, var, color=ORANGE, ls="--", lw=1.6, label="variance")
ax.axhline(SIG ** 2, color=MUTED, ls=":", lw=1.2); ax.text(5.5, SIG ** 2 - 0.022, "unavoidable noise", ha="center", fontsize=9, color=MUTED)
ax.set_ylim(0, 0.5); ax.set_xlim(-0.3, 12.3)
ax.axvspan(-0.3, 2.5, color=LIGHT_GOLD, alpha=0.35); ax.axvspan(7.5, 12.3, color=LIGHT_WINE, alpha=0.3)
ax.text(1.1, 0.47, "underfitting zone\nhigh bias", ha="center", fontsize=10, va="top")
ax.text(9.9, 0.47, "overfitting zone\nhigh variance", ha="center", fontsize=10, va="top")
ax.annotate("sweet spot\n(lowest validation error)", (best, vae[best - 0] if best < len(vae) else 0), (best + 0.3, 0.3),
            arrowprops=dict(arrowstyle="->", color=INK), fontsize=10, ha="center")
ax.annotate("", (9.5, tre[9]), (9.5, vae[9]), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
ax.text(9.35, (tre[9] + vae[9]) / 2 + 0.01, "gap =\nvariance\nsignal", fontsize=9, va="center", ha="right")
ax.set_xlabel("model complexity (polynomial degree)"); ax.set_ylabel("mean squared error")
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2)
fig.tight_layout(); save(fig, "biasvar-complexity")
for d in [1, best, 10]: print(d, "tr", tre[d].round(3), "va", vae[d].round(3), "b2", b2[d].round(3), "var", var[d].round(3))

# 4. learning curves
rng = np.random.default_rng(4)
sizes = np.array([8, 12, 20, 30, 50, 80, 130, 200, 320, 500])
def lc(d, reps=60):
    tr, va = [], []
    for m in sizes:
        t, v = [], []
        for r in range(reps):
            x = rng.uniform(0, 1, m); y = truth(x) + rng.normal(0, SIG, m)
            c = np.polyfit(x, y, d)
            xv = rng.uniform(0.02, 0.98, 300); yv = truth(xv) + rng.normal(0, SIG, 300)
            t.append(np.mean((np.polyval(c, x) - y) ** 2)); v.append(np.mean((np.polyval(c, xv) - yv) ** 2))
        tr.append(np.mean(t)); va.append(np.median(v))
    return np.array(tr), np.array(va)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
for ax, (d, t) in zip(axs, [(1, "High bias (degree 1)"), (10, "High variance (degree 10)")]):
    tr, va = lc(d)
    ax.plot(sizes, tr, "-o", ms=3, color=TEAL, label="training error")
    ax.plot(sizes, va, "-o", ms=3, color=WINE, label="validation error")
    ax.axhline(SIG ** 2, color=MUTED, ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel("training set size m (log scale)"); ax.set_title(t, fontsize=11)
    ax.set_ylim(0, 0.6)
    print(d, tr.round(3), va.round(3))
axs[0].set_ylabel("mean squared error"); axs[0].legend(fontsize=9)
axs[0].text(9, 0.1, "baseline (noise floor)", fontsize=9, color=MUTED)
axs[0].annotate("both plateau HIGH and close\ntogether: more data won't help", (300, 0.27), (12, 0.48), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
axs[1].annotate("big gap that closes\nas m grows: more data helps", (40, 0.2), (60, 0.45), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
fig.tight_layout(); save(fig, "biasvar-learning")
