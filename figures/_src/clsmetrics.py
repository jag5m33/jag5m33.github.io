import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

# 1. confusion matrix diagram
fig, ax = plt.subplots(figsize=(6.4, 4.6)); ax.axis("off")
cells = [((0, 1), "TP = 40", "true positive\nsaid 1, was 1", LIGHT_TEAL),
         ((1, 1), "FN = 10", "false negative\nmissed: said 0, was 1", LIGHT_WINE),
         ((0, 0), "FP = 20", "false positive\nfalse alarm: said 1, was 0", LIGHT_WINE),
         ((1, 0), "TN = 130", "true negative\nsaid 0, was 0", LIGHT_TEAL)]
for (c, r), big, small, fc in cells:
    ax.add_patch(Rectangle((c * 2, r * 1.6), 2, 1.6, fc=fc, ec=PAPER, lw=3))
    ax.text(c * 2 + 1, r * 1.6 + 1.05, big, ha="center", fontsize=15, fontweight="bold")
    ax.text(c * 2 + 1, r * 1.6 + 0.45, small, ha="center", fontsize=9.5, color=INK)
ax.text(1, 3.4, "predicted 1", ha="center", fontweight="bold"); ax.text(3, 3.4, "predicted 0", ha="center", fontweight="bold")
ax.text(-0.15, 2.4, "actual 1", ha="right", va="center", fontweight="bold"); ax.text(-0.15, 0.8, "actual 0", ha="right", va="center", fontweight="bold")
ax.text(2, -0.35, "precision reads DOWN the 'predicted 1' column;  recall reads ACROSS the 'actual 1' row",
        ha="center", fontsize=9.5, color=MUTED)
ax.set_xlim(-1.4, 4.1); ax.set_ylim(-0.6, 3.7)
save(fig, "clsmetrics-confusion")

# simulated scores for curves: 30% positives
rng = np.random.default_rng(0)
n = 1500; y = (rng.uniform(size=n) < 0.3).astype(int)
s = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 1.8 * y - 0.9)))
prec, rec, thr = precision_recall_curve(y, s)

# 2. threshold trade-off
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(thr, prec[:-1], color=TEAL, label="precision"); ax.plot(thr, rec[:-1], color=WINE, label="recall")
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12); i = np.argmax(f1)
ax.plot(thr, f1, color=GOLD, ls="--", lw=1.6, label="F1")
ax.axvline(thr[i], color=MUTED, ls=":", lw=1.2)
ax.text(thr[i] + 0.01, 0.02, f"best F1 at\nthreshold {thr[i]:.2f}", color=MUTED, fontsize=10)
ax.annotate("low threshold: catch everything,\nmany false alarms", xy=(0.12, 0.97), xytext=(0.03, 0.12), fontsize=10, color=WINE)
ax.annotate("high threshold: few alarms,\nbut miss positives", xy=(0.8, 0.2), xytext=(0.62, 0.36), fontsize=10, color=TEAL, bbox=dict(fc=PAPER, ec=TEAL, alpha=0.95))
ax.set_xlabel("threshold on predict_proba"); ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.set_xlim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3); ax.set_title("Moving the threshold trades precision for recall")
save(fig, "clsmetrics-threshold")

# 3. PR and ROC
ap = average_precision_score(y, s); auc = roc_auc_score(y, s); fpr, tpr, _ = roc_curve(y, s)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]; ax.plot(rec, prec, color=TEAL); ax.fill_between(rec, prec, color=LIGHT_TEAL, alpha=0.5, step="post")
ax.axhline(y.mean(), color=WINE, ls="--", lw=1.4); ax.text(0.03, y.mean() - 0.08, f"random guessing = positive rate {y.mean():.2f}", color=WINE, fontsize=10)
ax.text(0.25, 0.55, f"PR-AUC (AP) = {ap:.2f}", color=TEAL, fontsize=11)
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Precision-recall curve"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax = axes[1]; ax.plot(fpr, tpr, color=BLUE); ax.fill_between(fpr, tpr, color=LIGHT_GOLD, alpha=0.5)
ax.plot([0, 1], [0, 1], color=WINE, ls="--", lw=1.4); ax.text(0.68, 0.45, "random = 0.5", color=WINE, fontsize=10, rotation=0)
ax.text(0.35, 0.72, f"ROC-AUC = {auc:.2f}", color=BLUE, fontsize=11)
ax.set_xlabel("false positive rate (1 − specificity)"); ax.set_ylabel("true positive rate (recall)"); ax.set_title("ROC curve")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
fig.tight_layout(); save(fig, "clsmetrics-prroc")
print("pos rate", y.mean(), "AP", ap, "AUC", auc, "best thr", thr[i])
