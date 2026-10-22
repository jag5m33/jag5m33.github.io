import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np

# (1) % severe by year, all data, and by source (Jag's notes.txt 2.4 / 2.5)
yrs = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
allr = [34.2, 35.6, 35.6, 27.9, 18.4, 11.4, 6.9, 2.9]
s1 = [28.4, 32.3, 33.4, 21.0, 10.2, 2.4, 2.7, 2.9]
s2 = [36.8, 36.4, 36.0, 30.3, 30.6, 34.3, 34.5, None]
s3 = [47.5, 47.2, 48.6, 36.3, 29.7, 19.5, 22.5, None]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
ax = axes[0]
ax.plot(yrs, allr, marker="o", color=INK)
ax.axvspan(2019.5, 2023.5, color=LIGHT_GOLD, alpha=0.6, lw=0)
ax.text(2019.65, 37, "after the end-2019\nwording switch", fontsize=9.5, color=MUTED, va="top")
ax.annotate("35.6% → 2.9%:\ntwelvefold fall", xy=(2023, 2.9), xytext=(2016.2, 8), fontsize=10, color=WINE,
            arrowprops=dict(arrowstyle="->", color=WINE))
ax.set_title("All sources: % severe by year"); ax.set_ylabel("% severe"); ax.set_xticks(yrs[::1]); ax.tick_params(axis="x", rotation=45)
ax = axes[1]
for vals, name, c in [(s1, "Source1", WINE), (s2, "Source2", TEAL), (s3, "Source3", GOLD)]:
    xv = [y for y, v in zip(yrs, vals) if v is not None]; yv = [v for v in vals if v is not None]
    ax.plot(xv, yv, marker="o", color=c, label=name)
ax.text(2022.2, 36.5, "stable", color=TEAL, fontsize=10)
ax.set_title("Split by data provider"); ax.set_xticks(yrs); ax.tick_params(axis="x", rotation=45)
ax.legend(loc="lower left")
fig.tight_layout()
save(fig, "labelchecks-year")

# (2) synonym gaps, full dataset vs Source2
fig, ax = plt.subplots(figsize=(8, 2.8))
pairs = ["overcast vs cloudy", "clear vs fair"]
full = [15.1, 21.8]; s2g = [0.4, 4.6]
y = np.arange(2)
ax.barh(y + 0.18, full, height=0.34, color=WINE, label="all sources")
ax.barh(y - 0.18, s2g, height=0.34, color=TEAL, label="Source2 only")
for i in range(2):
    ax.text(full[i] + 0.3, y[i] + 0.18, f"{full[i]}", va="center", fontsize=10)
    ax.text(s2g[i] + 0.3, y[i] - 0.18, f"{s2g[i]}", va="center", fontsize=10)
ax.set_yticks(y, pairs); ax.set_xlabel("gap in % severe between two words for the same sky (points)")
ax.set_xlim(0, 25); ax.legend(loc="lower right")
ax.set_title("Synonyms should agree: the instrument check")
save(fig, "labelchecks-synonyms")

# (3) LEDGAR most-confused pairs (README), highlighting pairs also found as conflicting duplicates
pairs = ["Governing Laws → Applicable Laws", "Applicable Laws → Governing Laws", "Amendments → Modifications",
         "Governing Laws → Venues", "Authorizations → Authority", "Indemnifications → Indemnity",
         "Tax Withholdings → Withholdings", "Definitions → Defined Terms", "Waivers → No Waivers"]
cnt = [73, 33, 25, 24, 21, 18, 18, 16, 15]
dup = {"Governing Laws → Applicable Laws", "Applicable Laws → Governing Laws", "Waivers → No Waivers"}
fig, ax = plt.subplots(figsize=(8.5, 4.2))
cols = [WINE if p in dup else LIGHT_TEAL for p in pairs]
ax.barh(pairs[::-1], cnt[::-1], color=cols[::-1], edgecolor=[TEAL if c == LIGHT_TEAL else WINE for c in cols[::-1]])
for i, v in enumerate(cnt[::-1]):
    ax.text(v + 0.8, i, str(v), va="center", fontsize=10)
ax.set_xlabel("validation clauses (true → predicted)")
ax.set_title("LEDGAR: most-confused label pairs")
ax.text(40, 3.5, "wine = pair also seen as\nduplicate text with\nconflicting labels in train", color=WINE, fontsize=9.5)
ax.set_xlim(0, 82)
save(fig, "labelchecks-ledgar")
