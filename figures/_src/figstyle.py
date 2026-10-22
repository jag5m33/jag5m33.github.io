"""Shared look for every chapter figure.

Each chapter has its own script in this folder, e.g. figures/_src/descriptive.py.
Run it from the repo root:  python3 figures/_src/descriptive.py
It writes SVG files into figures/ which the chapter Markdown points at.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette (matches the chapter pages)
INK    = "#22223a"
MUTED  = "#6f6c80"
RULE   = "#e6dfcc"
PAPER  = "#fbf7ee"
GOLD   = "#c9a227"
WINE   = "#8b1e3f"
TEAL   = "#23877a"
BLUE   = "#3a6ea5"
ORANGE = "#d9822b"
LIGHT_GOLD = "#f3e3b0"
LIGHT_TEAL = "#bfe3dd"
LIGHT_WINE = "#ecc4cf"
CYCLE  = [TEAL, WINE, GOLD, BLUE, ORANGE, MUTED]

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "font.family": "serif",
    "font.serif": ["Georgia", "DejaVu Serif"],
    "font.size": 12,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=CYCLE),
    "lines.linewidth": 2.2,
    "svg.fonttype": "none",      # keep text as text: smaller files, crisp on any screen
    "mathtext.fontset": "dejavuserif",
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def save(fig, name):
    """Save as figures/<name>.svg"""
    path = os.path.join(OUT, name + ".svg")
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", path)
