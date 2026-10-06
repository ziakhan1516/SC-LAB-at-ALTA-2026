"""Figures for the ALTA 2026 system-description paper.

Both figures are single-series magnitude comparisons, so they use one hue and
carry direct value labels: that keeps them readable in greyscale print, where a
categorical palette would collapse.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE   = "#2a78d6"
INK    = "#0b0b0b"
MUTED  = "#52514e"
GRID   = "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

df = pd.read_csv("fig_data.csv")

# ---------------------------------------------------------------- figure 1
# The pool is saturated: the best single model nearly reaches the ensemble.
# Reference lines are the message, so they are styled (dash pattern, not hue)
# to survive greyscale.
ENS_HELDOUT, BLIND = 0.8621, 0.8526

fig, ax = plt.subplots(figsize=(3.35, 3.1))
d = df.sort_values("score")                      # ascending: best at top
y = list(range(len(d)))
ax.barh(y, d["score"], height=0.68, color=BLUE, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(d["model"], fontsize=6.2)
ax.set_xlim(0.830, 0.8775)
ax.set_ylim(-0.8, len(d) + 1.35)                 # headroom for the reference labels
ax.set_xlabel("out-of-fold official score", fontsize=7)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))
ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)

# reference lines stop below the label band so nothing overlaps
for xv, style in ((BLIND, (0, (3, 2))), (ENS_HELDOUT, "-")):
    ax.plot([xv, xv], [-0.8, len(d) + 0.35], color=INK, linewidth=0.9,
            linestyle=style, zorder=4, clip_on=False)
ax.annotate("blind test\n.8526", xy=(BLIND, len(d) + 0.5), ha="center",
            va="bottom", fontsize=5.7, color=MUTED, linespacing=1.2)
ax.annotate("ensemble,\nheld-out .8621", xy=(ENS_HELDOUT, len(d) + 0.5),
            ha="center", va="bottom", fontsize=5.7, color=INK, linespacing=1.2)

for yi, v in zip(y, d["score"]):
    ax.text(v + 0.0006, yi, f"{v:.4f}"[1:], va="center", ha="left",
            fontsize=5.6, color=MUTED, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.35))
fig.tight_layout(pad=0.2)
fig.savefig("fig_pool.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Where the difficulty actually is: one panel per scored cell, same x-range,
# so the reader sees sentiment solved and sarcasm both lower and more spread.
cells = [("sent-AU", "sentiment en-AU"), ("sent-UK", "sentiment en-UK"),
         ("sarc-AU", "sarcasm en-AU"),   ("sarc-UK", "sarcasm en-UK")]
fig, axes = plt.subplots(4, 1, figsize=(3.35, 1.95), sharex=True)
for ax, (col, title) in zip(axes, cells):
    v = df[col]
    ax.scatter(v, [0] * len(v), s=11, color=BLUE, alpha=0.75,
               edgecolors="white", linewidths=0.3, zorder=3, clip_on=False)
    ax.scatter([v.min()], [0], s=11, color=BLUE, edgecolors=INK,
               linewidths=0.6, zorder=4, clip_on=False)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0.75, 1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, 0.62, title, transform=ax.transAxes, fontsize=6.8,
            color=INK, va="bottom")
    ax.text(1.0, 0.62, f"min {v.min():.3f}", transform=ax.transAxes,
            fontsize=5.8, color=MUTED, va="bottom", ha="right")
axes[-1].set_xlabel("macro-$F_1$ of each of the 17 models", fontsize=7)
fig.tight_layout(pad=0.2, h_pad=0.9)
fig.savefig("fig_cells.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print("wrote fig_pool.pdf, fig_cells.pdf")
