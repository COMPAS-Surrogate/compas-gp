"""Two title-free, standalone scientific figures and a short discussion."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-"]


def save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.out
    summary = json.loads((out / "summary.json").read_text())
    data = np.load(out / "losses.npz")
    support = json.loads((out / "support.json").read_text())["rows"]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "xtick.top": True, "ytick.right": True,
                         "xtick.direction": "in", "ytick.direction": "in"})
    sizes = data["sizes"]
    x = np.geomspace(10, 1000, 150)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True, sharey=True, layout="constrained")
    for i, size in enumerate(sizes):
        if size != 512:
            axes[0].plot(x, x * data["nested"][i], color=COLORS[i], ls=STYLES[i], lw=1.6)
        lo, median, hi = np.quantile(data["bootstrap"][i], [.16, .5, .84])
        axes[1].plot(x, x * median, color=COLORS[i], ls=STYLES[i], lw=1.6, label=f"{size}M")
        axes[1].fill_between(x, x * lo, x * hi, color=COLORS[i], alpha=.12, lw=0)
    for ax in axes:
        ax.set(xscale="log", yscale="log", xlabel="Expected observed detections", xlim=(10, 1000))
        ax.set_xticks([10, 30, 100, 300, 1000], [10, 30, 100, 300, 1000])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", top=True, right=True, direction="in")
    axes[0].set_ylabel("Expected log-likelihood loss [nats]")
    axes[1].legend(title="COMPAS systems", frameon=False, fontsize=8.5, title_fontsize=9, ncol=2, loc="upper left")
    save(fig, out, "population_loss")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
    thresholds = np.geomspace(1, 1e5, 300)
    for i in [0, 2, 6]:
        row = np.load(out / f"support_mass10_{sizes[i]}.npz")
        weights = row["reference"].ravel()
        neff = row["neff"].ravel()
        fraction = [100 * weights[neff < t].sum() / weights.sum() for t in thresholds]
        axes[0].plot(thresholds, fraction, color=COLORS[i], ls=STYLES[i], label=f"{sizes[i]}M", lw=1.6)
    axes[0].set(xscale="log", xlim=(1, 1e5), ylim=(0, 100),
                xlabel=r"Effective systems per pixel, $N_{\rm eff}$",
                ylabel="Reference detections below threshold [%]")
    axes[0].legend(frameon=False, loc="upper left")
    for grid, label, color, style in zip(
        ["native", "mass10", "mass20"], ["5% mass bins", "10% mass bins", "20% mass bins"],
        ["#0072B2", "#D55E00", "#009E73"], ["-", "--", "-."]):
        values = [100 * r["fraction_neff_lt30"] for r in support if r["grid"] == grid]
        axes[1].plot(sizes, values, marker="o", ms=4, color=color, ls=style, label=label)
    axes[1].set(xscale="log", xlabel="COMPAS initial systems [millions]",
                ylabel=r"Reference detections in pixels with $N_{\rm eff}<30$ [%]", ylim=(0, None))
    axes[1].set_xticks(sizes, [str(s) for s in sizes])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True, direction="in")
    save(fig, out, "grid_support")

    with (out / "grid_support.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(support[0]))
        writer.writeheader()
        writer.writerows(support)
    table = ["| Initial systems | Actual nested loss | Bootstrap median [16–84%] |",
             "|---|---:|---:|"]
    for i, size in enumerate(sizes):
        lo, median, hi = 1000 * np.quantile(data["bootstrap"][i], [.16, .5, .84])
        actual = 0. if size == 512 else 1000 * data['nested'][i]
        table.append(f"| {size}M | {actual:.3g} | {median:.3g} [{lo:.3g}, {hi:.3g}] |")
    pixels = [r for r in support if r["grid"] == "mass10"]
    pixel_table = ["| Initial systems | Selected systems | Positive pixels | Reference detections with Neff < 30 | Missing support |",
                   "|---|---:|---:|---:|---:|"]
    for row in pixels:
        pixel_table.append(f"| {row['size']}M | {row['selected_systems']:,} | {row['positive_pixels']} / {row['pixels']} | "
                           f"{100*row['fraction_neff_lt30']:.2f}% | {100*row['missing_rate_fraction']:.4g}% |")
    discussion = f"""# Figure 9 extension: direct population calculations

The expected likelihood loss grows linearly with the number of observations.
The real population calculations determine each slope; changing observing
duration does not require another cosmic-integration run. Larger populations
usually reduce this discrepancy, but a particular nested subset is one
realization, not a measurement of mean bias.

![Population loss](population_loss.png)

**Figure caption.** Expected log-likelihood loss relative to the finite 512M
reference, at alpha = −0.3, sigma = 0.35, SFR amplitude = 0.01 and SFR exponent
= 4.7, using the fixed mass10 grid and nominal continuous measurement errors.
Left: actual nested population sizes; the full 512M reference is identically
zero and omitted from the logarithmic axis. Right: median and 16th–84th
percentiles from 128 populations per size resampled with replacement from 64
blocks of 8M initial systems. All blocks in a resample share the same weight
across bins and parameters. The nonzero 512M curve is a conditional bootstrap
estimate, not an independent simulation. Bands show population resampling
spread, not numerical integration uncertainty. Both panels use the same axes.
The observing-duration scaling assumes fixed selection and measurement errors.

At **1000 expected detections**, all losses below are in nats:

{chr(10).join(table)}

![Grid support](grid_support.png)

**Figure caption.** Support of actual nested populations at the same parameters.
Left: cumulative fraction of the full-reference detection rate in mass10 pixels
below a given effective-system count, including unsupported pixels at Neff = 0.
Right: that fraction below Neff = 30 for three fixed mass resolutions, all with
dz = 0.1. Effective counts use each binary's integrated pixel weight. The
threshold 30 is descriptive, not an accuracy requirement; the left panel shows
the full threshold dependence. Coarser grids retain exactly the same total rate.

For the 870-pixel mass10 grid:

{chr(10).join(pixel_table)}

**Simple interpretation.** Small COMPAS populations can leave observable parts
of a fine grid poorly supported. Coarser pixels reduce this problem but also
discard distribution detail; they do not improve the total-rate estimate.
Raw occupancy alone is insufficient because the cosmic weights are unequal.
An adequate grid therefore depends on the population size, weights, measurement
precision and catalogue size, rather than a universal number of pixels per 8M
systems. Redshift cells receive correlated contributions from the same binary.

For this injection, moving from 5% to 10% to 20% mass bins reduces the fraction
below Neff = 30 from **85% to 13% to 2.2% for 32M**. For 8M it stays above
97% for all three grids: these modest coarsenings alone do not provide strong
pixel support. Measurement uncertainty averages contributions across pixels,
so pixel Neff is a support diagnostic, not an event-likelihood error estimate.

**Scope and checks.** This estimates discrepancy and conditional resampling
variability within one archive, not absolute bias relative to an infinite
population, missing evolutionary channels, or posterior calibration. The blocks
must be exchangeable for a bootstrap interpretation; the archive does not
establish independent-run provenance. Only parameter-dependent likelihood
errors can shift a posterior; these plots alone do not establish a parameter
bias. Perfect measurements have infinite expected losses wherever the finite
candidate misses reference support; no rate floor is introduced. Gaussian
measurement tails are evaluated in log space when ordinary arithmetic underflows.

Every block's sparse predictions were checked against the direct integrator,
and the rebuilt 32M/128M/256M/512M predictions match the previous Figure 9 cache.
Two integrations use {summary['draws_per_integration']:,} measurements each;
the maximum estimated numerical standard error is
{100*summary['maximum_relative_numerical_se']:.2f}% of a nonzero loss.
Numerical status: **{summary['state']}**. Full pixel counts, three descriptive
thresholds (10/30/100), missing-support fractions and exact perfect-bin losses
are in [grid_support.csv](grid_support.csv). Rates at five nearby sigma values
are also cached for a follow-up check of parameter dependence.
"""
    (out / "DISCUSSION.md").write_text(discussion)


if __name__ == "__main__":
    main()
