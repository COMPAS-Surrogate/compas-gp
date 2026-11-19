"""Render the two approved appendix figures from the frozen direct study."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
import numpy as np

from plot_study import COLORS, STYLES, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "grid_resolution"))
from probe_grids import configurations


def main() -> None:
    source = HERE / "results"
    dest = REPO / "overleaf/figures/bo_study"
    dest.mkdir(parents=True, exist_ok=True)
    assert json.loads((source / "summary.json").read_text())["state"] == "complete"
    losses = np.load(source / "losses.npz")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.top": True, "ytick.right": True,
        "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(figsize=(4.8, 3.3), layout="constrained")
    x = np.geomspace(10, 1000, 150)
    for i, size in enumerate(losses["sizes"]):
        if size == 512:
            continue  # Identically zero reference is not on the logarithmic axis.
        ax.plot(x, x * losses["nested"][i], color=COLORS[i], ls=STYLES[i],
                lw=1.5, label=f"{size}M")
    ax.set(xscale="log", yscale="log", xlim=(10, 1000),
           xlabel=r"Expected observed detections, $\mathbb{E}[N_{\rm obs}]$",
           ylabel="Expected log-likelihood loss [nats]")
    ax.set_xticks([10, 30, 100, 300, 1000], [10, 30, 100, 300, 1000])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.legend(title="COMPAS initial systems", frameon=False, fontsize=8,
              title_fontsize=9, loc="upper left", ncol=2)
    save(fig, dest, "population_likelihood_loss")

    edges, groups, _ = configurations()["mass10"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.05), sharex=True, sharey=True,
                             layout="constrained")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#d9d9d9")
    norm = LogNorm(vmin=1, vmax=1e4)
    diagnostics = {}
    for ax, size in zip(axes, [8, 32, 512]):
        data = np.load(source / f"support_mass10_{size}.npz")
        rate, neff = data["rates"], data["neff"]
        assert rate.shape == neff.shape == (len(edges) - 1, len(groups) - 1)
        assert np.all(rate[-1] == 0), "Do not omit a populated infinite guard bin"
        values = np.ma.masked_where(rate[:-1] == 0, neff[:-1])
        mesh = ax.pcolormesh(edges[:-1], groups * .1, values.T, cmap=cmap, norm=norm,
                              shading="flat", rasterized=True)
        ax.set(xlim=(0, 45), ylim=(0, 1.5), xlabel=r"Chirp mass, $\mathcal{M}\,[M_\odot]$")
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_yticks([0, .5, 1., 1.5])
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True, direction="in")
        # The plot crops a finite within-bin-uniform density; likelihood support
        # is unchanged. Include the fraction of a bin crossing 45 solar masses.
        visible = np.clip((45 - edges[:-2]) / np.diff(edges[:-1]), 0, 1)
        omitted = float(np.sum(rate[:-1] * (1 - visible[:, None])) / rate.sum())
        assert omitted < 1e-6, "Displayed range no longer satisfies the manuscript caption"
        diagnostics[str(size)] = dict(omitted_detection_fraction=omitted,
                                     maximum_neff=float(neff.max()))
    axes[0].set_ylabel("Redshift, $z$")
    bar = fig.colorbar(mesh, ax=axes, pad=.02, fraction=.025, ticks=[1, 10, 100, 1000, 10000])
    bar.set_label(r"Effective systems per pixel, $N_{\rm eff}$")
    save(fig, dest, "population_pixel_support")
    (source / "manuscript_figures.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


if __name__ == "__main__":
    main()
