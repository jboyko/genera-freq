"""
02_powerlaw_test.py

Tests whether genus species-count data follow a power law (Clauset et al. 2009)
and whether the exponent alpha is consistent across Head/Tail Break tiers —
the key test of self-similarity.

HOW THE TEST WORKS
------------------
Clauset et al. fit a power law p(x) ~ x^(-alpha) only to data above some
lower bound xmin. They don't assume the whole distribution is a power law —
just the tail. xmin is chosen automatically: the algorithm tries every possible
value and picks the one where the fitted power law deviates least from the
empirical data (minimum KS statistic). This matters because many distributions
look like a power law in the tail even if they aren't globally.

The log-likelihood ratio test (R_ln) then compares whether the power law or
a log-normal fits the data above that same xmin better. R > 0 favors power law;
R < 0 favors log-normal. The p-value tells you whether the difference is
statistically meaningful.

Three outputs:
  1. plots/02_powerlaw_ccdf.png      — complementary CDF with power-law fit per tier
  2. plots/02_powerlaw_alpha.png     — alpha estimates (+/- 2 SE) across tiers
  3. plots/02_powerlaw_fits.png      — histogram per tier with lognormal + power-law overlaid
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import powerlaw
import scipy.stats as stats
from pathlib import Path

warnings.filterwarnings("ignore")  # powerlaw is verbose

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA = Path("data/770398df9fbf743cdadb51e63f2f4abffe6e8159/resource.csv")

# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, low_memory=False)
df = df[df["role"] == "accepted"].copy()
sizes = df["species"].dropna().astype(int)
sizes = sizes[sizes > 0].values

# ---------------------------------------------------------------------------
# Head/Tail Breaks (same as script 01)
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

levels = head_tail_breaks(sizes)
thresholds = [t for t, _ in levels]
tier_arrays = [sizes] + [h for _, h in levels]
tier_labels = ["All genera (tier 0)"] + [
    f"Tier {i+1}  (>{t:,.0f} sp.)" for i, t in enumerate(thresholds)
]

# ---------------------------------------------------------------------------
# Fit power law to each tier and collect results
# ---------------------------------------------------------------------------
print("Fitting power laws per tier (Clauset et al. 2009)\n")
print(f"{'Tier':<28} {'n':>6}  {'xmin':>7}  {'alpha':>6}  {'sigma':>6}  "
      f"{'R LN':>7}  {'p LN':>5}")
print("-" * 75)

fits = []
for data, label in zip(tier_arrays, tier_labels):
    fit = powerlaw.Fit(data, discrete=True, verbose=False)

    # Log-likelihood ratio vs log-normal (positive R = power law better)
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")

    fits.append({
        "label":   label,
        "n":       len(data),
        "xmin":    fit.xmin,
        "alpha":   fit.power_law.alpha,
        "sigma":   fit.power_law.sigma,
        "R_ln":    R_ln,
        "p_ln":    p_ln,
        "fit":     fit,
    })

    print(f"{label:<28} {len(data):>6,}  {fit.xmin:>7.0f}  "
          f"{fit.power_law.alpha:>6.3f}  {fit.power_law.sigma:>6.3f}  "
          f"{R_ln:>7.3f}  {p_ln:>5.3f}")

# ---------------------------------------------------------------------------
# Figure 1: CCDF on log-log axes, one line per tier, with power-law fit
# ---------------------------------------------------------------------------
colors = plt.cm.Blues_r(np.linspace(0.15, 0.75, len(fits)))
fig1, ax1 = plt.subplots(figsize=(6, 4.5))

for f, c in zip(fits, colors):
    # Empirical CCDF
    data_sorted = np.sort(f["fit"].data)
    ccdf = 1 - np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    ax1.plot(data_sorted, ccdf, color=c, lw=1.2, alpha=0.8,
             label=f["label"].split("  ")[0])

    # Power-law fit line (above xmin)
    xmin, alpha = f["xmin"], f["alpha"]
    x_fit = np.logspace(np.log10(xmin), np.log10(data_sorted.max()), 200)
    # CCDF of a discrete power law: P(X >= x) ~ x^(1-alpha) / xmin^(1-alpha)
    y_fit = (x_fit / xmin) ** (1 - alpha)
    # Scale to empirical CCDF at xmin
    y_at_xmin = ccdf[np.searchsorted(data_sorted, xmin)]
    ax1.plot(x_fit, y_fit * y_at_xmin, color=c, lw=1.5, linestyle="--", alpha=0.6)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Species per genus", fontsize=10)
ax1.set_ylabel("P(X ≥ x)", fontsize=10)
ax1.set_title("Complementary CDF by tier — dashed = power-law fit", fontsize=10)
ax1.legend(fontsize=7.5, frameon=False)
ax1.spines[["top", "right"]].set_visible(False)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}" if x >= 1 else ""
))

plt.tight_layout()
out1 = Path("plots/02_powerlaw_ccdf.png")
plt.savefig(out1, dpi=180, bbox_inches="tight")
print(f"\nSaved → {out1}")

# ---------------------------------------------------------------------------
# Figure 2: alpha +/- 2*sigma per tier
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 3.5))

alphas  = [f["alpha"] for f in fits]
sigmas  = [f["sigma"] for f in fits]
labels  = [f["label"] for f in fits]
xs      = np.arange(len(fits))

ax2.errorbar(xs, alphas, yerr=[2 * s for s in sigmas],
             fmt="o", color="#4C72B0", capsize=5, capthick=1.5,
             elinewidth=1.5, markersize=6, zorder=3)

# Shade the band of the tier-0 estimate for visual reference
a0, s0 = alphas[0], sigmas[0]
ax2.axhspan(a0 - 2 * s0, a0 + 2 * s0, alpha=0.10, color="#4C72B0",
            label="Tier 0 ± 2σ band")

ax2.set_xticks(xs)
ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax2.set_ylabel("Power-law exponent  α", fontsize=10)
ax2.set_title("Consistency of α across tiers\n"
              "Overlap with shaded band = self-similar", fontsize=10)
ax2.legend(fontsize=8, frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out2 = Path("plots/02_powerlaw_alpha.png")
plt.savefig(out2, dpi=180, bbox_inches="tight")
print(f"Saved → {out2}")

plt.show()

# ---------------------------------------------------------------------------
# Figure 3: histogram per tier with log-normal and power-law fits overlaid
#
# Log-normal is fit to ALL data in each tier (MLE on log-transformed values),
# so its PDF covers the full range and the intercept is correct.
# Power law is only valid above xmin (dotted line); drawn from there only.
# ---------------------------------------------------------------------------
n_panels = len(fits)
ncols = 3
nrows = int(np.ceil(n_panels / ncols))
fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
axes3 = axes3.flatten()

for idx, (f, ax) in enumerate(zip(fits, axes3)):
    data   = tier_arrays[idx]
    xmin   = f["xmin"]
    alpha  = f["alpha"]
    lo, hi = data.min(), data.max()

    # --- histogram (log-spaced bins, normalized to PDF) ---
    bins = np.logspace(np.log10(lo), np.log10(hi), 35)
    counts, edges = np.histogram(data, bins=bins)
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths)

    ax.bar(edges[:-1], density, width=widths, align="edge",
           color="#4C72B0", alpha=0.55, edgecolor="white", linewidth=0.3)

    x_curve    = np.logspace(np.log10(lo), np.log10(hi), 400)
    frac_above = (data >= xmin).mean()

    # --- log-normal: same xmin-conditioned fit used in the LR test ---
    # Truncated to [xmin, inf), scaled to match histogram (frac_above)
    ln_mu    = f["fit"].lognormal.mu
    ln_sigma = f["fit"].lognormal.sigma
    x_ln = np.logspace(np.log10(xmin), np.log10(hi), 400)
    ln_pdf_raw = stats.lognorm.pdf(x_ln, s=ln_sigma, scale=np.exp(ln_mu))
    # Renormalize over [xmin, inf) so the conditional PDF integrates to 1,
    # then scale by frac_above to match the full-data histogram
    ln_survival = 1 - stats.lognorm.cdf(xmin, s=ln_sigma, scale=np.exp(ln_mu))
    y_ln = ln_pdf_raw / ln_survival * frac_above
    ax.plot(x_ln, y_ln, color="#55A868", lw=1.8, label="log-normal")

    # --- power law: valid only above xmin ---
    x_pl = np.logspace(np.log10(xmin), np.log10(hi), 400)
    y_pl = (alpha - 1) / xmin * (x_pl / xmin) ** (-alpha) * frac_above
    ax.plot(x_pl, y_pl, color="#C44E52", lw=1.8, linestyle="--",
            label=f"power law (α={alpha:.2f})")

    # --- mark xmin ---
    ax.axvline(xmin, color="gray", lw=1.0, linestyle=":")
    ax.text(xmin * 1.08, 0.02, f"xmin\n{xmin:,.0f}",
            color="gray", fontsize=6.5, va="bottom",
            transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Species per genus", fontsize=9)
    if idx % ncols == 0:
        ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f["label"], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # legend only on first panel; n annotation on all
    if idx == 0:
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.text(0.03, 0.03, f"n = {len(data):,}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color="#555555")

for ax in axes3[n_panels:]:
    ax.set_visible(False)

fig3.suptitle(
    "Genus size distribution per tier — log-normal vs power-law fit\n"
    "Both fits conditioned on xmin (dotted line) — same basis as likelihood-ratio test",
    fontsize=11, y=1.01
)
plt.tight_layout()
out3 = Path("plots/02_powerlaw_fits.png")
plt.savefig(out3, dpi=180, bbox_inches="tight")
print(f"Saved → {out3}")
plt.show()
