"""
01_genus_size_selfsimilarity.py

Computes Head/Tail Breaks thresholds on accepted genus species counts from
resource.csv, then visualizes the self-similar, fractal-like structure of the
heavy-tailed distribution: each tier looks qualitatively the same as the last.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA = Path("data/770398df9fbf743cdadb51e63f2f4abffe6e8159/resource.csv")
OUT  = Path("plots/01_genus_size_selfsimilarity.png")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, sep=",", low_memory=False)

# Keep accepted genera with at least 1 species
df = df[df["role"] == "accepted"].copy()
sizes = df["species"].dropna().astype(int)
sizes = sizes[sizes > 0].values

print(f"Total accepted genera with ≥1 species: {len(sizes):,}")
print(f"Max genus size: {sizes.max():,}")

# ---------------------------------------------------------------------------
# Head/Tail Breaks
# ---------------------------------------------------------------------------
def head_tail_breaks(values: np.ndarray, max_head_fraction: float = 0.40):
    """
    Recursively split around the mean until the head (above-mean subset) is
    no longer a minority (<= max_head_fraction of the current set), indicating
    the distribution is no longer heavy-tailed at that scale.

    Returns list of (mean_threshold, head_array) tuples, one per level.
    """
    levels = []
    current = values.copy()
    while True:
        mean = current.mean()
        head = current[current > mean]
        if len(head) == 0 or len(head) / len(current) > max_head_fraction:
            break
        levels.append((mean, head))
        current = head
    return levels

levels = head_tail_breaks(sizes)
thresholds = [t for t, _ in levels]

print("\nHead/Tail Breaks thresholds (species count):")
for i, (t, h) in enumerate(levels):
    print(f"  Level {i+1}: mean = {t:.1f}  →  head n = {len(h):,} "
          f"({100*len(h)/len(sizes):.2f}% of all genera)")

# ---------------------------------------------------------------------------
# Build tier arrays for plotting
# (tier 0 = all genera, tier k = genera above threshold[k-1])
# ---------------------------------------------------------------------------
tier_arrays = [sizes] + [h for _, h in levels]
tier_labels  = ["All genera"] + [
    f">{t:,.0f} species  (tier {i+1})" for i, t in enumerate(thresholds)
]

# ---------------------------------------------------------------------------
# Visualization: one panel per tier
# ---------------------------------------------------------------------------
n_panels = len(tier_arrays)
fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4.5))

if n_panels == 1:
    axes = [axes]

for ax, data, label, idx in zip(axes, tier_arrays, tier_labels,
                                 range(n_panels)):
    # Log-spaced bins spanning the range of this tier
    lo = max(1, data.min())
    hi = data.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 30)

    counts, edges = np.histogram(data, bins=bins)

    # Bar chart in log-log space
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color="#4C72B0",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )

    # Mark next threshold (if one exists) as a vertical line
    if idx < len(thresholds):
        next_t = thresholds[idx]
        ax.axvline(next_t, color="#C44E52", linewidth=1.5, linestyle="--")
        ax.text(next_t * 1.08, 0.98, f"{next_t:,.0f}",
                color="#C44E52", fontsize=7.5, va="top", ha="left",
                rotation=90, transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Species per genus", fontsize=9)
    ax.set_ylabel("Number of genera", fontsize=9) if idx == 0 else ax.set_ylabel("")
    ax.set_title(label, fontsize=9, pad=6)

    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1g}"
    ))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y >= 1 else f"{y:.1g}"
    ))
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate n
    ax.text(0.97, 0.97, f"n = {len(data):,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555555")

fig.suptitle(
    "Self-similar structure of genus size distributions",
    fontsize=11, y=1.02
)

plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"\nSaved → {OUT}")
plt.show()
