import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np

# worked-example data: x = house size (100 m^2), y = price (GBP 100k)
x = np.array([0.8, 1.2, 1.5, 2.0, 2.4, 3.0])
y = np.array([1.9, 2.4, 3.3, 3.8, 4.9, 5.6])
m = len(x)
w_best, b_best = np.polyfit(x, y, 1)
def J(w, b): return ((w * x + b - y) ** 2).sum() / (2 * m)

# 1. the model
fig, ax = plt.subplots(figsize=(8, 4.2))
xs = np.linspace(0, 3.3, 50)
ax.scatter(x, y, s=60, color=WINE, zorder=3, label="training examples $(x^{(i)}, y^{(i)})$")
ax.plot(xs, w_best * xs + b_best, color=TEAL, label=f"fitted line  $f = {w_best:.2f}x + {b_best:.2f}$")
ax.axhline(0, color=RULE, lw=1); ax.axvline(0, color=RULE, lw=1)
ax.scatter([0], [b_best], color=GOLD, s=70, zorder=4)
ax.annotate("intercept b = %.2f\n(prediction when x = 0)" % b_best, (0, b_best), (0.25, 2.6),
            arrowprops=dict(arrowstyle="->", color=INK), fontsize=11)
x1, x2 = 1.0, 2.0
ax.plot([x1, x2], [w_best*x1+b_best]*2, color=GOLD, lw=1.6, ls="--")
ax.plot([x2, x2], [w_best*x1+b_best, w_best*x2+b_best], color=GOLD, lw=1.6, ls="--")
ax.text(1.5, w_best*x1+b_best-0.35, "run = 1", ha="center", color=MUTED, fontsize=11)
ax.text(2.06, w_best*1.5+b_best, "rise = w = %.2f\n(slope: price added\nper extra 100 m²)" % w_best, color=MUTED, fontsize=11, va="center")
ax.set_xlabel("x: house size (100 m²)"); ax.set_ylabel("y: price (£100k)")
ax.set_xlim(-0.1, 3.4); ax.set_ylim(-0.2, 6.5); ax.legend(loc="upper left", fontsize=10)
ax.set_title("The model  f(x) = wx + b")
save(fig, "linreg-model")

# 2. residuals
w0, b0 = 1.0, 1.0
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, w0 * xs + b0, color=BLUE, label="a guess:  f = 1.0x + 1.0")
for xi, yi in zip(x, y):
    ax.plot([xi, xi], [w0*xi+b0, yi], color=WINE, lw=1.6)
ax.scatter(x, y, s=60, color=WINE, zorder=3, label="actual y")
ax.scatter(x, w0*x+b0, s=40, color=BLUE, marker="s", zorder=3, label="prediction ŷ on the line")
ax.annotate("residual = ŷ − y = −1.6\n(vertical gap, not the\nshortest distance)", (3.0, 4.8), (2.0, 1.2),
            arrowprops=dict(arrowstyle="->", color=INK), fontsize=11)
ax.set_xlabel("x: house size (100 m²)"); ax.set_ylabel("y: price (£100k)")
ax.legend(loc="upper left", fontsize=10); ax.set_title("Residuals: how far each prediction misses")
save(fig, "linreg-residuals")

# 3. J vs w (b fixed at best b)
ws = np.linspace(0.2, 3.3, 200)
Jw = np.array([J(w, b_best) for w in ws])
picks = [(0.9, BLUE), (w_best, TEAL), (2.6, ORANGE)]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax = axs[0]
ax.scatter(x, y, color=WINE, s=40, zorder=3)
for w, c in picks:
    ax.plot(xs, w*xs+b_best, color=c, label=f"w = {w:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=10); ax.set_title("Three slopes (b fixed)")
ax = axs[1]
ax.plot(ws, Jw, color=INK, lw=1.8)
for w, c in picks:
    ax.scatter([w], [J(w, b_best)], color=c, s=70, zorder=3)
    ax.annotate(f"J = {J(w, b_best):.2f}", (w, J(w, b_best)), (w+0.08, J(w, b_best)+0.25), fontsize=10, color=c)
ax.annotate("minimum: best w", (w_best, 0), (w_best-0.6, 1.3), arrowprops=dict(arrowstyle="->", color=INK), fontsize=11)
ax.set_xlabel("w (slope)"); ax.set_ylabel("cost J(w)"); ax.set_title("Each line is ONE point on J(w)")
fig.tight_layout(); save(fig, "linreg-cost-w")

# 4. contour of J(w,b)
W, B = np.meshgrid(np.linspace(-0.5, 4.0, 200), np.linspace(-3, 4, 200))
JJ = ((W[..., None]*x + B[..., None] - y)**2).sum(-1) / (2*m)
levels = [0.02, 0.1, 0.3, 0.6, 1, 2, 3.5, 6, 9]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = fig.add_subplot(1, 2, 1, projection="3d"); axs[0].remove()
ax.plot_surface(W, B, JJ, cmap="cividis", alpha=0.85, linewidth=0, rstride=10, cstride=10, rasterized=True)
ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("J"); ax.set_title("J(w, b) is a bowl")
ax.view_init(28, -55)
ax = axs[1]
cs = ax.contour(W, B, JJ, levels=levels, colors=[MUTED], linewidths=1)
ax.clabel(cs, fontsize=8, fmt="%.2g")
ax.scatter([w_best], [b_best], color=WINE, marker="*", s=180, zorder=3)
ax.annotate(f"minimum\nw = {w_best:.2f}, b = {b_best:.2f}", (w_best, b_best), (2.55, 3.2),
            arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_title("Same bowl from above (contours)")
save(fig, "linreg-contour")

# 5. gradient descent path + cost vs iteration
def run(alpha, iters, w=0.0, b=-2.0):
    path = [(w, b)]
    for _ in range(iters):
        e = w*x + b - y
        dw, db = (e*x).mean(), e.mean()
        w, b = w - alpha*dw, b - alpha*db
        path.append((w, b))
    return np.array(path)
P = run(0.1, 400)
costs = [J(*p) for p in P]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
ax = axs[0]
ax.contour(W, B, JJ, levels=levels, colors=[MUTED], linewidths=0.8)
ax.plot(P[:60, 0], P[:60, 1], "-o", color=TEAL, ms=3, lw=1.2)
ax.scatter([P[0, 0]], [P[0, 1]], color=ORANGE, s=60, zorder=3); ax.annotate("start (0, −2)", P[0], (0.3, -2.7), fontsize=10)
ax.scatter([w_best], [b_best], color=WINE, marker="*", s=180, zorder=3)
ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_title("Descent path on the contour plot")
ax = axs[1]
ax.plot(costs, color=TEAL)
ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("J(w, b)  (log scale)")
ax.set_title("Cost vs iteration: should only fall")
ax.annotate("flattens = converged", (300, costs[300]), (150, 0.4), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
fig.tight_layout(); save(fig, "linreg-descent")

# 6. learning rate (1-D: J(w) with b fixed)
a_max = 2 / (x**2).mean()
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for ax, (alpha, title, c) in zip(axs, [(0.02, "α too small: crawls", BLUE),
                                       (0.25, "α about right: converges", TEAL),
                                       (0.58, "α too big: overshoots, diverges", WINE)]):
    w = 0.4; pts = [w]
    for _ in range(8):
        w = w - alpha * ((w*x + b_best - y)*x).mean(); pts.append(w)
    pts = np.array(pts)
    ww = np.linspace(-1.5, 5, 300)
    ax.plot(ww, [J(v, b_best) for v in ww], color=MUTED, lw=1.4)
    jp = [J(v, b_best) for v in pts]
    ax.plot(pts, jp, "-o", color=c, ms=4, lw=1.2)
    ax.set_title(f"{title}\nα = {alpha}", fontsize=11); ax.set_xlabel("w")
    ax.set_ylim(0, 12); ax.set_xlim(-1.5, 5)
axs[0].set_ylabel("J(w)")
fig.tight_layout(); save(fig, "linreg-alpha")
print("alpha max (1-D)", a_max)

# 7. diagnostics: residuals vs fitted
rng = np.random.default_rng(0)
n = 120; xx = rng.uniform(0, 10, n)
cases = [("Good: random band around 0", 2 + 1.5*xx + rng.normal(0, 1.5, n), TEAL),
         ("Bad: curve (non-linear)", 2 + 0.3*(xx-5)**2 + 1.5*xx + rng.normal(0, 1, n), WINE),
         ("Bad: funnel (heteroscedastic)", 2 + 1.5*xx + rng.normal(0, 0.3 + 0.5*xx, n), ORANGE)]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, (t, yy, c) in zip(axs, cases):
    a, bb = np.polyfit(xx, yy, 1); fit = a*xx + bb
    ax.scatter(fit, yy - fit, s=12, color=c, alpha=0.8)
    ax.axhline(0, color=INK, lw=1)
    ax.set_title(t, fontsize=11); ax.set_xlabel("fitted value ŷ")
axs[0].set_ylabel("residual  y − ŷ")
fig.tight_layout(); save(fig, "linreg-diagnostics")

# 8. multicollinearity: coefficients across resamples
def coefs(rho, reps=40):
    out = []
    for r in range(reps):
        g = np.random.default_rng(100 + r)
        x1 = g.normal(size=40); x2 = rho*x1 + np.sqrt(1-rho**2)*g.normal(size=40)
        yy = 3*x1 + 3*x2 + g.normal(0, 2, 40)
        Xd = np.c_[np.ones(40), x1, x2]
        out.append(np.linalg.lstsq(Xd, yy, rcond=None)[0][1:])
    return np.array(out)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (rho, t) in zip(axs, [(0.0, "x1, x2 unrelated (r = 0)"), (0.98, "x1, x2 almost copies (r = 0.98)")]):
    C = coefs(rho)
    ax.scatter(np.zeros(len(C)) + rng.uniform(-0.12, 0.12, len(C)), C[:, 0], color=TEAL, s=18, label="w1")
    ax.scatter(np.ones(len(C)) + rng.uniform(-0.12, 0.12, len(C)), C[:, 1], color=WINE, s=18, label="w2")
    ax.axhline(3, color=GOLD, ls="--", lw=1.4); ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["w1", "w2"]); ax.set_title(t, fontsize=11)
axs[0].set_ylabel("fitted coefficient\n(40 resamples)")
axs[0].text(1.3, 3.3, "true value 3", color=GOLD, fontsize=10)
fig.tight_layout(); save(fig, "linreg-multicollinearity")

# 9. outlier + extrapolation
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
ax = axs[0]
xo = np.append(x, 3.2); yo = np.append(y, 1.0)
a1, c1 = np.polyfit(x, y, 1); a2, c2 = np.polyfit(xo, yo, 1)
ax.scatter(x, y, color=WINE, s=40); ax.scatter([3.2], [1.0], color=ORANGE, s=70, marker="X")
ax.plot(xs, a1*xs+c1, color=TEAL, label="without outlier")
ax.plot(xs, a2*xs+c2, color=ORANGE, ls="--", label="with one outlier")
ax.annotate("one bad point\ndrags the line", (3.2, 1.0), (1.9, 0.6), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
ax.legend(fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Outliers pull the fit (squared error)")
ax = axs[1]
xt = np.linspace(0, 10, 200); g = np.random.default_rng(1)
xd = g.uniform(0, 4, 25); yd = 10*np.sqrt(xd) + g.normal(0, 0.8, 25)
a3, c3 = np.polyfit(xd, yd, 1)
ax.scatter(xd, yd, color=WINE, s=20, label="data you have")
ax.plot(xt, 10*np.sqrt(xt), color=MUTED, ls=":", label="true relationship")
ax.plot(xt, a3*xt+c3, color=TEAL, label="linear fit")
ax.axvspan(4, 10, color=LIGHT_WINE, alpha=0.4); ax.text(5.2, 8, "extrapolating:\nno data here", fontsize=10)
ax.legend(fontsize=9, loc="upper left"); ax.set_xlabel("x"); ax.set_title("Extrapolation beyond the data")
fig.tight_layout(); save(fig, "linreg-outlier")
