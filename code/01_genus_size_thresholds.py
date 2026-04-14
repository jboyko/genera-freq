"""
01_genus_size_thresholds.py

Defines Small / Medium / Big / Megadiverse genus size thresholds using:
  1. Clauset et al. (2009) to estimate xmin — the entry to the heavy tail
  2. Head/Tail Breaks (Jiang & Yin 2010) on genera >= xmin to find
     internal tiers

Produces:
  plots/fig_distribution.png  — full distribution with tiers shaded
  plots/fig_fits.png          — power-law vs log-normal above xmin
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import powerlaw
import scipy.stats as stats
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA = Path("data/770398df9fbf743cdadb51e63f2f4abffe6e8159/resource.csv")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df    = pd.read_csv(DATA, low_memory=False)
df    = df[df["role"] == "accepted"]
sizes = df["species"].dropna().astype(int)
sizes = sizes[sizes > 0].values

# ---------------------------------------------------------------------------
# Step 1: estimate xmin via Clauset et al. (2009)
# ---------------------------------------------------------------------------
fit   = powerlaw.Fit(sizes, discrete=True, verbose=False)
xmin  = int(fit.xmin)
alpha = fit.power_law.alpha
R, p  = fit.distribution_compare("power_law", "lognormal")

print(f"Power-law fit:  xmin = {xmin},  alpha = {alpha:.3f}")
print(f"LR test vs log-normal:  R = {R:.3f},  p = {p:.3f}\n")

# ---------------------------------------------------------------------------
# Step 2: Head/Tail Breaks on genera >= xmin
# ---------------------------------------------------------------------------
def head_tail_breaks(values, max_head_fraction=0.40):
    levels, current = [], values.copy()
    while True:
        mean = current.mean()
        head = current[current > mean]
        if len(head) == 0 or len(head) / len(current) > max_head_fraction:
            break
        levels.append((mean, head))
        current = head
    return levels

tail_data = sizes[sizes >= xmin]
levels    = head_tail_breaks(tail_data)
thresholds = [int(np.ceil(t)) for t, _ in levels]

t_big      = thresholds[0]   # ~420
t_mega     = thresholds[1]   # ~883

n_medium = np.sum((sizes >= xmin)  & (sizes < t_big))
n_big    = np.sum((sizes >= t_big) & (sizes < t_mega))
n_mega   = np.sum(sizes >= t_mega)

print(f"Thresholds from HTB on tail: {thresholds}")
print(f"\nTier summary:")
print(f"  Small       (<{xmin} sp.):     n = {np.sum(sizes < xmin):,}  "
      f"({100*np.mean(sizes < xmin):.1f}%)")
print(f"  Medium      ({xmin}–{t_big-1} sp.):   n = {n_medium:,}  "
      f"({100*n_medium/len(sizes):.1f}%)")
print(f"  Big         ({t_big}–{t_mega-1} sp.):  n = {n_big:,}  "
      f"({100*n_big/len(sizes):.2f}%)")
print(f"  Megadiverse (≥{t_mega} sp.):   n = {n_mega:,}  "
      f"({100*n_mega/len(sizes):.2f}%)")

# ---------------------------------------------------------------------------
# Colours shared across figures
# ---------------------------------------------------------------------------
C_SMALL  = "#BBBBBB"
C_MEDIUM = "#4C72B0"
C_BIG    = "#DD8452"
C_MEGA   = "#C44E52"

# ---------------------------------------------------------------------------
# Figure 1: full distribution with tiers
# ---------------------------------------------------------------------------
fig1, ax = plt.subplots(figsize=(7, 4))

lo, hi = sizes.min(), sizes.max()
edges  = np.logspace(np.log10(lo), np.log10(hi), 80)
bins   = edges

# colour each bar by tier using left edge — with 80 uniform bins the
# offset between a bar's left edge and the dashed threshold line is <1 bin width
counts, _ = np.histogram(sizes, bins=bins)
for i, (left, right, count) in enumerate(zip(edges[:-1], edges[1:], counts)):
    if left < xmin:
        c = C_SMALL
    elif left < t_big:
        c = C_MEDIUM
    elif left < t_mega:
        c = C_BIG
    else:
        c = C_MEGA
    ax.bar(left, count, width=right - left, align="edge",
           color=c, edgecolor="white", linewidth=0.2)

# threshold lines with tier name and range as label
for x, label, color in [
    (xmin,   f"Medium ({xmin}–{t_big-1} sp.)",   C_MEDIUM),
    (t_big,  f"Big ({t_big}–{t_mega-1} sp.)",    C_BIG),
    (t_mega, f"Megadiverse (≥{t_mega} sp.)",      C_MEGA),
]:
    ax.axvline(x, color=color, linewidth=1.2, linestyle="--", zorder=3)
    ax.text(x * 1.07, 0.97, label,
            color=color, fontsize=7.5, rotation=90,
            va="top", transform=ax.get_xaxis_transform())

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Species per genus", fontsize=10)
ax.set_ylabel("Number of genera", fontsize=10)
ax.set_title("Genus size distribution with data-derived thresholds", fontsize=10)
ax.tick_params(labelsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}" if x >= 1 else ""))
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = Path("plots/fig_distribution.png")
plt.savefig(out1, dpi=180, bbox_inches="tight")
print(f"\nSaved → {out1}")

# ---------------------------------------------------------------------------
# Figure 2: power-law vs log-normal fits above xmin
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(5.5, 4))

tail = sizes[sizes >= xmin]
lo_t, hi_t = tail.min(), tail.max()

# histogram above xmin
bins_t  = np.logspace(np.log10(lo_t), np.log10(hi_t), 30)
cnts, edges_t = np.histogram(tail, bins=bins_t)
widths  = np.diff(edges_t)
density = cnts / (cnts.sum() * widths)

ax2.bar(edges_t[:-1], density, width=widths, align="edge",
        color=C_MEDIUM, alpha=0.45, edgecolor="white", linewidth=0.3,
        label="data (above xmin)")

x_curve = np.logspace(np.log10(lo_t), np.log10(hi_t), 400)

# log-normal: conditioned on xmin (same basis as LR test)
ln_mu    = fit.lognormal.mu
ln_sigma = fit.lognormal.sigma
ln_surv  = 1 - stats.lognorm.cdf(xmin, s=ln_sigma, scale=np.exp(ln_mu))
y_ln     = stats.lognorm.pdf(x_curve, s=ln_sigma, scale=np.exp(ln_mu)) / ln_surv
ax2.plot(x_curve, y_ln, color="#55A868", lw=1.8, label="log-normal")

# power law: p(x) = (α-1)/xmin * (x/xmin)^(-α)
y_pl = (alpha - 1) / xmin * (x_curve / xmin) ** (-alpha)
ax2.plot(x_curve, y_pl, color=C_MEGA, lw=1.8, linestyle="--",
         label=f"power law (α = {alpha:.2f})")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Species per genus", fontsize=10)
ax2.set_ylabel("Density", fontsize=10)
ax2.set_title(f"Distributional fits above xmin = {xmin}\n"
              f"LR test: R = {R:.2f}, p = {p:.2f}  (indistinguishable)",
              fontsize=9)
ax2.legend(fontsize=8, frameon=False)
ax2.tick_params(labelsize=8)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}" if x >= 1 else ""))
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out2 = Path("plots/fig_fits.png")
plt.savefig(out2, dpi=180, bbox_inches="tight")
print(f"Saved → {out2}")

plt.close("all")

# ---------------------------------------------------------------------------
# Figure 3: Head/Tail Breaks self-similarity across all scales
# Run HTB on all genera; each panel = one recursive level
# Dashed line = the mean that defines the next split
# ---------------------------------------------------------------------------
all_levels = head_tail_breaks(sizes)
all_thresholds = [t for t, _ in all_levels]

htb_panels = [sizes] + [h for _, h in all_levels]
htb_means  = all_thresholds          # mean of panel i → next split boundary
htb_titles = ["All genera"] + [
    f">{t:,.0f} sp." for t in all_thresholds
]

n_panels = len(htb_panels)
fig3, axes3 = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 4))

for i, (ax_i, data, title) in enumerate(zip(axes3, htb_panels, htb_titles)):
    lo_i = max(1, data.min())
    hi_i = data.max()
    bins_i = np.logspace(np.log10(lo_i), np.log10(hi_i), 25)
    counts_i, _ = np.histogram(data, bins=bins_i)

    ax_i.bar(bins_i[:-1], counts_i, width=np.diff(bins_i), align="edge",
             color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.3)

    # mark the mean = the HTB split that defines the next panel
    if i < len(htb_means):
        m = htb_means[i]
        ax_i.axvline(m, color="#C44E52", lw=1.4, ls="--", zorder=3)
        ax_i.text(m * 1.08, 0.97, f"mean = {m:,.0f}",
                  color="#C44E52", fontsize=7.5, rotation=90,
                  va="top", transform=ax_i.get_xaxis_transform())

    ax_i.set_xscale("log")
    ax_i.set_yscale("log")
    ax_i.set_title(title, fontsize=9, pad=5)
    ax_i.set_xlabel("Species per genus", fontsize=9)
    ax_i.tick_params(labelsize=7)
    ax_i.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax_i.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y >= 1 else ""))
    ax_i.spines[["top", "right"]].set_visible(False)
    ax_i.text(0.97, 0.97, f"n = {len(data):,}",
              transform=ax_i.transAxes, ha="right", va="top",
              fontsize=8, color="#555555")

axes3[0].set_ylabel("Number of genera", fontsize=9)
fig3.suptitle("Self-similar structure of genus size: Head/Tail Breaks across scales",
              fontsize=9)
plt.tight_layout()
out3 = Path("plots/fig_nested.png")
plt.savefig(out3, dpi=180, bbox_inches="tight")
print(f"Saved → {out3}")

# ---------------------------------------------------------------------------
# Table: tier summary — genera, species, and median genus size
# ---------------------------------------------------------------------------
total_genera  = len(sizes)
total_species = sizes.sum()

tier_specs = [
    ("Small",        sizes[sizes < xmin],                          f"<{xmin}"),
    ("Medium",       sizes[(sizes >= xmin)  & (sizes < t_big)],    f"{xmin}–{t_big-1}"),
    ("Big",          sizes[(sizes >= t_big) & (sizes < t_mega)],   f"{t_big}–{t_mega-1}"),
    ("Megadiverse",  sizes[sizes >= t_mega],                       f"≥{t_mega}"),
]

rows = []
print("\nTier summary")
print(f"{'Tier':<14} {'Range':>10}  {'Genera':>7}  {'% genera':>9}  "
      f"{'Species':>10}  {'% species':>10}  {'Median sp.':>11}")
print("-" * 80)

for label, data, rng in tier_specs:
    n_gen   = len(data)
    n_sp    = data.sum()
    pct_gen = 100 * n_gen / total_genera
    pct_sp  = 100 * n_sp  / total_species
    med     = int(np.median(data))
    rows.append({
        "Tier":          label,
        "Range (sp.)":   rng,
        "Genera":        n_gen,
        "% genera":      round(pct_gen, 1),
        "Species":       int(n_sp),
        "% species":     round(pct_sp, 1),
        "Median sp.":    med,
    })
    print(f"{label:<14} {rng:>10}  {n_gen:>7,}  {pct_gen:>8.1f}%  "
          f"{n_sp:>10,}  {pct_sp:>9.1f}%  {med:>11,}")

table_out = Path("tables/table_tier_summary.csv")
table_out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(table_out, index=False)
print(f"\nSaved → {table_out}")
