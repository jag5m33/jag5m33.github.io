import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np, pandas as pd, seaborn as sns

rng = np.random.default_rng(0)
# synthetic demand-like table (half-hours) used for the generic panels
n = 400
hour = rng.integers(0, 24, n)
month = rng.integers(1, 13, n)
winter = np.isin(month, [11, 12, 1, 2])
season = np.where(winter, "winter", np.where(np.isin(month, [5, 6, 7, 8]), "summer", "spring/autumn"))
solar = np.clip(np.where((hour > 7) & (hour < 18), rng.normal(4000, 2000, n) * (1 - 0.6 * winter), 0), 0, None)
daytime = ((hour >= 7) & (hour <= 20)).astype(float)
demand = 22000 + 8000 * daytime + 6000 * winter - 0.9 * solar + rng.normal(0, 2000, n)
temp = 15 - 9 * winter + rng.normal(0, 3, n)
df = pd.DataFrame(dict(hour=hour, month=month, season=season, solar=solar.round(), demand=demand.round(), temp=temp.round(1)))
df["period"] = np.where(daytime == 1, "day", "night")

# (1) numeric-numeric
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
xs = rng.normal(0, 1, 120)
for ax, (r, title) in zip(axes, [(0.9, "strong positive"), (0.0, "none"), (-0.6, "moderate negative")]):
    ys = r * xs + np.sqrt(1 - r**2) * rng.normal(0, 1, 120)
    ax.scatter(xs, ys, s=14, color=TEAL, alpha=0.7)
    rr = np.corrcoef(xs, ys)[0, 1]
    ax.set_title(f"{title}: r = {rr:.2f}")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout()
save(fig, "unibimulti-scatter")

# (2) categorical-numeric
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
order = ["summer", "spring/autumn", "winter"]
sns.boxplot(data=df, x="season", y="demand", order=order, ax=axes[0], color=LIGHT_TEAL, linecolor=TEAL, width=0.5)
axes[0].set_title("Box plot per group"); axes[0].set_ylabel("demand (MW)"); axes[0].set_xlabel("")
sns.violinplot(data=df, x="season", y="demand", order=order, ax=axes[1], color=LIGHT_GOLD, linecolor=GOLD, inner="quartile", cut=0)
axes[1].set_title("Violin: shape + quartiles"); axes[1].set_ylim(top=41500); axes[1].set_ylabel(""); axes[1].set_xlabel("")
axes[1].annotate("two bulges = day\nand night groups", xy=(2, 34000), xytext=(0.55, 37600), fontsize=9.5, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=MUTED))
fig.tight_layout()
save(fig, "unibimulti-catnum")

# (3) categorical-categorical: real Source2 numbers are not available as a crosstab, so synthetic
road = rng.choice(["Local", "State Hwy", "Interstate"], 1000, p=[0.55, 0.25, 0.20])
p_sev = np.select([road == "Local", road == "State Hwy"], [0.15, 0.24], 0.80)
severe = np.where(rng.random(1000) < p_sev, "severe", "not severe")
ct = pd.crosstab(road, severe, normalize="index").loc[["Local", "State Hwy", "Interstate"]] * 100
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
sns.heatmap(ct, annot=True, fmt=".1f", cmap=sns.light_palette(WINE, as_cmap=True), cbar=False, ax=axes[0],
            annot_kws={"fontsize": 12}, linewidths=1, linecolor="white")
axes[0].set_title("Row-normalised crosstab (%)"); axes[0].set_ylabel("road type"); axes[0].set_xlabel("")
axes[0].tick_params(axis="y", rotation=0)
bottom = np.zeros(3)
ctr = ct.iloc[::-1]
for col, c in [("severe", WINE), ("not severe", LIGHT_TEAL)]:
    axes[1].barh(ctr.index, ctr[col], left=bottom, color=c, label=col)
    bottom += ctr[col].values
axes[1].set_xlabel("% of rows in that road type"); axes[1].set_title("100% stacked bar")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
fig.tight_layout()
save(fig, "unibimulti-catcat")

# (4) multivariate: hue + facets (scatter solar vs demand, faceted by season, hue=period)
g = sns.FacetGrid(df, col="season", col_order=order, height=3.2, aspect=1.05)
g.map_dataframe(sns.scatterplot, x="solar", y="demand", hue="period", palette={"day": GOLD, "night": BLUE}, s=18)
g.set_titles("{col_name}"); g.set_axis_labels("solar (MW)", "demand (MW)"); g.add_legend(title="")
save(g.figure, "unibimulti-facets")

# (5) pairplot + corr heatmap
small = df[["demand", "solar", "temp", "hour"]].sample(150, random_state=0)
g = sns.pairplot(small, corner=True, height=1.9, plot_kws=dict(s=10, color=TEAL, alpha=0.7), diag_kws=dict(color=TEAL))
save(g.figure, "unibimulti-pairplot")

corr = df[["demand", "solar", "temp", "hour"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(5.5, 4.2))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, linewidths=1, linecolor="white")
ax.set_title("Correlation heatmap (lower triangle)")
save(fig, "unibimulti-corr")

# (6) target-focused: Jag's real Source2 % severe by street type
st = pd.Series({"Local": 14.67, "County Road": 17.60, "State Highway": 23.73, "US Highway": 30.79,
                "Other": 36.84, "Freeway": 43.53, "Unknown": 49.72, "Interstate": 80.40})
fig, ax = plt.subplots(figsize=(8.5, 4))
cols = [WINE if v > 33.92 else TEAL for v in st.values]
ax.barh(st.index, st.values, color=cols)
ax.axvline(33.92, color=INK, ls="--", lw=1.4)
ax.text(34.8, 7.35, "baseline 33.92%", fontsize=10, color=INK)
for i, v in enumerate(st.values):
    ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=10)
ax.set_xlim(0, 92); ax.set_xlabel("% severe (Source2)")
ax.set_title("Feature vs target: % severe by road type")
save(fig, "unibimulti-target")

# (7) split test: traffic_signal within road type (Jag's numbers)
rows = ["Local", "State Highway", "US Highway", "Freeway", "Interstate"]
no_sig = [19.47, 26.04, 32.67, 46.95, 80.36]; sig = [5.77, 6.76, 17.53, 19.89, 81.62]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
y = np.arange(len(rows))
ax.hlines(y, sig, no_sig, color=MUTED, lw=2)
ax.scatter(no_sig, y, color=WINE, s=60, label="no signal", zorder=3)
ax.scatter(sig, y, color=TEAL, s=60, label="signal", zorder=3)
ax.set_yticks(y, rows); ax.set_xlabel("% severe (Source2)")
ax.legend(loc="lower right")
ax.annotate("gap gone: everything\non interstates is severe", xy=(81, 4), xytext=(48, 3.6), fontsize=9.5, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_title("Split test: does the signal gap survive within road type?")
save(fig, "unibimulti-split")

day = df[df.period == "day"]
print("r all", round(df.solar.corr(df.demand), 2), "r day", round(day.solar.corr(day.demand), 2))
for se in order:
    d = day[day.season == se]; print(se, round(d.solar.corr(d.demand), 2))
