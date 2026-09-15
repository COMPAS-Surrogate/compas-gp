"""Publish the 512M fiducial mass-redshift endpoint of the integration figure."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.sparse import load_npz

from cosmic_integration.ratesSampler.binned_cosmic_integrator import BinnedCosmicIntegrator

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Contract all cached population blocks and verify the plotted marginal."""
    source = ROOT / "output/pdf/cosmic_integration_512M"
    out = ROOT / "overleaf/figures/bo_study"
    out.mkdir(parents=True, exist_ok=True)
    block = ROOT / "docs/studies/population_stability/block_results"
    meta = json.loads((block / "population_meta.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    assert provenance["initial_systems"] == 512_000_000
    params = provenance["parameters"]
    ci = BinnedCosmicIntegrator(SimpleNamespace(), None,
                              p_MaxRedshift=10., p_MaxRedshiftDetection=1.5)
    ci.CalculateRedshiftRelatedParams()
    density, _, pdraw = ci.FindZdistribution(
        ci.redshifts, np.log(meta["zmin"]), np.log(meta["zmax"]),
        p_Alpha=params["alpha"], p_Sigma=params["sigma_lnZ"])
    formed = ci.CalculateSFR(ci.redshifts, params["a_SFR"], params["d_SFR"]) / (
        512_000_000 * meta["mass_per_binary"])
    operator = sum((load_npz(block / f"block_{i:02d}.npz") for i in range(1, 16)),
                   load_npz(block / "block_00.npz"))
    with np.load(ROOT / "docs/studies/rate_heatmaps/rate_maps.npz") as saved:
        mass, redshift = saved["mass_edges"], saved["z_edges"]
    rates = (operator @ (formed[:, None] * density / pdraw).ravel()).reshape(
        len(mass)-1, len(redshift)-1)
    with np.load(source / "physical_arrays.npz") as saved:
        assert np.allclose(rates.sum(axis=0), saved["detected"], rtol=1e-11, atol=1e-10)
        assert np.array_equal(redshift, saved["redshift_edges"])
    keep = (mass[:-1] > 0) & np.isfinite(mass[1:])
    assert np.all(rates[~keep] == 0)
    idx = np.flatnonzero(keep)
    assert np.all(np.diff(idx) == 1)
    edges = mass[idx[0]:idx[-1]+2]
    area = np.diff(edges)[:, None] * np.diff(redshift)[None, :]
    grid = rates[keep] / area
    assert np.isclose(np.sum(grid * area), rates.sum(), rtol=1e-12)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 9, "axes.labelsize": 9, "pdf.fonttype": 42,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    cmap = ListedColormap(plt.get_cmap("inferno")(np.linspace(.08, 1, 256)))
    cmap.set_bad("#eeeeee")
    cmap.set_under(cmap(0))
    vmax = 10**np.ceil(np.log10(grid.max()))
    fig, ax = plt.subplots(figsize=(3.5, 3.5), layout="constrained")
    im = ax.pcolormesh(redshift, edges, np.ma.masked_less_equal(grid, 0),
                       norm=LogNorm(vmax/1e5, vmax), cmap=cmap,
                       shading="flat", rasterized=True)
    ax.set(xlabel=r'Merger redshift $z$', ylabel=r'Chirp mass $\mathcal{M}_c$ [$M_\odot$]',
           yscale="log", xlim=(redshift[0], redshift[-1]), ylim=(edges[0], edges[-1]))
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, color=".6")
    for spine in ax.spines.values():
        spine.set_color(".6")
    fig.legend(handles=[Patch(facecolor="#eeeeee", edgecolor=".7", label="Zero rate")],
               loc="outside upper center", fontsize=7, frameon=False)
    cb = fig.colorbar(im, ax=ax, extend="min", pad=.02)
    cb.set_label(r'$r_{kl}/(\Delta\mathcal{M}_{c,k}\,\Delta z_l)$ [$\mathrm{yr}^{-1}\,M_\odot^{-1}$]', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    assert not ax.get_title() and fig._suptitle is None
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cosmic_integration_grid.{ext}", dpi=230)
        shutil.copy2(source / f"cosmic_integration_physical.{ext}",
                     out / f"cosmic_integration_physical.{ext}")
    plt.close(fig)
    np.savez_compressed(source / "rate_grid.npz", rates=rates, mass_edges=mass,
                        redshift_edges=redshift)
    info = {"parameters": params, "initial_systems": 512_000_000,
            "rate_grid_shape": list(rates.shape), "total_rate_per_year": float(rates.sum()),
            "density_measure": "per unit linear chirp mass and merger redshift",
            "checks": ["mass marginal matches physical panel d", "density integral recovers total rate",
                       "omitted zero-edge and infinite-edge bins contain no rate"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in out.glob("cosmic_integration_*.pdf")}}
    (source / "rate_grid_provenance.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
