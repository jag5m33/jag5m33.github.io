import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

y    = np.array([200, 250, 310, 180, 420, 300])
yhat = np.array([210, 240, 330, 170, 380, 310])
idx = np.arange(1, 7)

# 1. R^2 picture: SS_tot vs SS_res
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
ax = axs[0]; ybar = y.mean()
ax.axhline(ybar, color=GOLD, lw=2, label=f"baseline: always predict the mean ({ybar:.0f})")
for i, v in zip(idx, y): ax.plot([i, i], [ybar, v], color=GOLD, lw=1.6)
ax.scatter(idx, y, color=WINE, s=50, zorder=3, label="actual y")
ax.set_title("SS_tot: misses of the mean-only model\n(sum of squares = 38,133)", fontsize=11)
ax.set_xlabel("house"); ax.set_ylabel("price (£k)"); ax.legend(fontsize=9, loc="upper left")
ax = axs[1]
for i, a, p in zip(idx, y, yhat): ax.plot([i, i], [a, p], color=TEAL, lw=1.6)
ax.scatter(idx, yhat, color=TEAL, marker="s", s=40, zorder=3, label="model prediction ŷ")
ax.scatter(idx, y, color=WINE, s=50, zorder=3, label="actual y")
ax.set_title("SS_res: misses of your model\n(sum of squares = 2,400)", fontsize=11)
ax.set_xlabel("house"); ax.legend(fontsize=9, loc="upper left")
fig.tight_layout(); save(fig, "regmetrics-r2")

# 2. MAE vs MSE with an outlier: share of the total each point contributes
yhat2 = yhat.copy(); yhat2[4] = 220
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
for ax, pred, t in zip(axs, [yhat, yhat2], ["Original predictions", "House 5 predicted badly (220 instead of 380)"]):
    e = np.abs(y - pred)
    a = e / e.sum() * 100; s = e**2 / (e**2).sum() * 100
    ax.bar(idx - 0.2, a, width=0.4, color=TEAL, label="share of MAE (|error|)")
    ax.bar(idx + 0.2, s, width=0.4, color=WINE, label="share of MSE (error²)")
    ax.set_title(t, fontsize=11); ax.set_xlabel("house"); ax.set_xticks(idx)
    mae = e.mean(); rmse = np.sqrt((e**2).mean())
    ax.text(0.03, 0.97, f"MAE = {mae:.1f}\nRMSE = {rmse:.1f}", transform=ax.transAxes, va="top", fontsize=10)
axs[0].set_ylabel("% of the total")
axs[1].annotate(f"{(200**2)/((y-yhat2)**2).sum()*100:.0f}% of MSE\nfrom one house", (5.2, 94), (2.2, 70), arrowprops=dict(arrowstyle="->", color=INK), fontsize=10)
axs[1].legend(fontsize=9, loc="center left")
fig.tight_layout(); save(fig, "regmetrics-outlier")

# 3. predicted vs actual and residual plot (simulated 80 houses)
rng = np.random.default_rng(0)
n = 80; size = rng.uniform(50, 200, n)
price = 40 + 1.6 * size + rng.normal(0, 18, n)
X = size.reshape(-1, 1); mdl = LinearRegression().fit(X, price); p = mdl.predict(X)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axs[0]
ax.scatter(price, p, s=15, color=TEAL); lim = [100, 380]
ax.plot(lim, lim, color=INK, lw=1, ls="--"); ax.text(300, 340, "perfect: ŷ = y", color=MUTED, fontsize=10)
ax.set_xlabel("actual price (£k)"); ax.set_ylabel("predicted price (£k)"); ax.set_title("Predicted vs actual", fontsize=11)
ax = axs[1]
ax.scatter(p, price - p, s=15, color=TEAL); ax.axhline(0, color=INK, lw=1)
sd = np.std(price - p)
ax.axhspan(-sd, sd, color=LIGHT_TEAL, alpha=0.5)
ax.text(p.min(), sd + 4, f"±1 RMSE ≈ ±{sd:.0f}", fontsize=10, color=MUTED)
ax.set_xlabel("predicted price ŷ (£k)"); ax.set_ylabel("residual y − ŷ (£k)"); ax.set_title("Residual plot", fontsize=11)
fig.tight_layout(); save(fig, "regmetrics-residuals")
print("RMSE sim", sd)

# 4. R^2 always rises with junk features; adjusted R^2 doesn't
rng = np.random.default_rng(3)
n = 50; x1 = rng.normal(size=n); yy = 2 * x1 + rng.normal(0, 2, n)
junk = rng.normal(size=(n, 20))
ks = list(range(0, 21)); r2s, adj = [], []
for k in ks:
    Xk = np.c_[x1, junk[:, :k]]; r2 = r2_score(yy, LinearRegression().fit(Xk, yy).predict(Xk))
    pk = Xk.shape[1]; r2s.append(r2); adj.append(1 - (1 - r2) * (n - 1) / (n - pk - 1))
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(ks, r2s, "-o", ms=4, color=WINE, label="plain R² (training data)")
ax.plot(ks, adj, "-o", ms=4, color=TEAL, label="adjusted R²")
ax.set_xlabel("number of pure-noise features added"); ax.set_ylabel("score")
ax.set_xticks(range(0, 21, 2))
ax.set_title("50 rows, 1 real feature, then junk columns", fontsize=12); ax.legend(fontsize=10)
fig.tight_layout(); save(fig, "regmetrics-adjr2")
for k in [0, 5, 10, 20]: print(k, round(r2s[k], 3), round(adj[k], 3))
