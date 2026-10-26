import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from figstyle import *
from matplotlib.patches import FancyBboxPatch

# A simple visual chooser: one question at the top, three columns of "situation -> metric".
fig, ax = plt.subplots(figsize=(10.5, 5.4)); ax.axis("off")
ax.set_xlim(0, 10.5); ax.set_ylim(0, 6)

def rbox(x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=fc, ec=MUTED, lw=1))

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

# top question
rbox(2.75, 5.15, 5.0, 0.6, LIGHT_GOLD)
ax.text(5.25, 5.45, "Do you have the true answers (labels)?", ha="center", va="center", fontweight="bold", fontsize=11.5)

cols = [
    (0.1, "Yes, predicting a number\n(regression)",
     [("outliers are just noise", "MAE"), ("big misses are much worse", "RMSE"),
      ("how much is explained?", "R²")]),
    (3.6, "Yes, predicting a class\n(classification)",
     [("balanced classes", "accuracy, F1"), ("a miss is costly", "recall"),
      ("a false alarm is costly", "precision"), ("rare positives", "PR-AUC"),
      ("many classes, all matter", "macro F1")]),
    (7.1, "No labels\n(unsupervised)",
     [("clusters", "silhouette, elbow"), ("anomalies", "stability, sense"),
      ("a few known anomalies", "precision@k")]),
]
W = 3.3
for x, head, rows in cols:
    rbox(x, 3.9, W, 0.75, LIGHT_GOLD)
    ax.text(x + W / 2, 4.275, head, ha="center", va="center", fontsize=10.5, fontweight="bold")
    arrow(5.25, 5.12, x + W / 2, 4.68)
    rbox(x, 0.55, W, 3.1, LIGHT_TEAL)
    arrow(x + W / 2, 3.88, x + W / 2, 3.68)
    for i, (situation, metric) in enumerate(rows):
        y = 3.42 - i * 0.6
        ax.text(x + 0.2, y, situation, va="center", fontsize=9.5, color=MUTED)
        ax.text(x + 0.2, y - 0.25, "→ " + metric, va="center", fontsize=10.5, fontweight="bold", color=TEAL)

ax.text(5.25, 0.12, "Whatever you pick, compare it with a baseline: always predict the mean, or the majority class.",
        ha="center", fontsize=10, color=WINE, style="italic")
save(fig, "metricsintro-chooser")
