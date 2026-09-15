"""Physical cosmic-integration illustration from one fixed COMPAS archive.

Run from the repository root with the project Python. No manuscript files change.
The display histogram in panel (c) is not used to calculate panel (d).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import load_npz

from cosmic_integration.ratesSampler.binned_cosmic_integrator import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.ratesSampler.population_stream import load_selected_population
from cosmic_integration.ratesSampler.ratesSampler import SelectionEffects

ROOT = Path(__file__).resolve().parents[3]
PARAMETERS = (-0.3, 0.35, 0.01, 4.7)


class CompleteSelectionIntegrator(BinnedCosmicIntegrator):
    """Same rate calculation with detection probability fixed to unity."""

    def FindDetectionProbability(self, masses, eta, redshifts, distances,
                                 n_redshifts, n_binaries, snr_grid, probability):
        return np.ones((n_binaries, n_redshifts))


def calculate(population_path: Path) -> tuple[dict, dict]:
    """Compute physical grids and verify their normalization and rate parity."""
    meta_path = ROOT / "docs/studies/population_stability/block_results/population_meta.json"
    meta = json.loads(meta_path.read_text())
    logging.info("Streaming selected binaries from %s", population_path.name)
    population = load_selected_population(population_path, meta["mass_per_binary"])
    np.random.seed(0)
    selection = SelectionEffects(p_SNRsensitivity="O3")
    ci = BinnedCosmicIntegrator(population, selection,
                              p_MaxRedshift=10., p_MaxRedshiftDetection=1.5)
    ci.CalculateRedshiftRelatedParams()
    alpha, sigma, amplitude, exponent = PARAMETERS
    tmin, tmax = ci.times[-1] / 1000, ci.times[0] / 1000
    time_edges = np.linspace(tmin, tmax, 161)
    time = (time_edges[:-1] + time_edges[1:]) / 2
    # Match the time/redshift interpolation used by FixedPopulationRates.
    redshift = interp1d(ci.times / 1000, ci.redshifts)(time)
    density, metals, pdraw = ci.FindZdistribution(
        redshift, np.log(population.zamsZmin), np.log(population.zamsZmax),
        p_Alpha=alpha, p_Sigma=sigma)
    sfr = ci.CalculateSFR(redshift, amplitude, exponent) / 1e9  # Mpc^-3
    psi = sfr[:, None] * density  # per d ln Z, not per dZ
    du = np.diff(np.log(metals))[0]
    assert np.allclose(density.sum(axis=1) * du, 1, rtol=1e-12)
    assert np.allclose(psi.sum(axis=1) * du, sfr, rtol=1e-12)

    total_mass = population.nSystems * population.massEvolvedPerBinary
    delays = population.delayTime / 1000  # Myr -> Gyr
    delay_edges = np.linspace(0, max(tmax, delays.max()) * (1 + 1e-12), 101)
    widths = np.diff(delay_edges)
    lnz_edges = np.linspace(np.log(population.zamsZmin), np.log(population.zamsZmax), 65)
    counts, _, _ = np.histogram2d(np.log(population.Zsystems), delays,
                                bins=(lnz_edges, delay_edges))
    assert counts.sum() == len(delays)
    # Empirical yield per formed stellar mass, per d ln Z and per Gyr.
    # The inverse draw density removes the archive's uniform-lnZ sampling.
    yield_density = counts / (total_mass * pdraw * np.diff(lnz_edges)[:, None] * widths)
    recovered = np.sum(yield_density * np.diff(lnz_edges)[:, None] * widths)
    assert np.isclose(recovered, len(delays) / (total_mass * pdraw), rtol=1e-12)

    # Evaluate formation weights at each binary's metallicity using exactly
    # the legacy digitize convention; only delay is binned for this display.
    metal_index = np.digitize(population.Zsystems, metals)
    delay_index = np.searchsorted(delay_edges, delays, side="right") - 1
    contribution = np.zeros((len(time), len(widths)))
    for i in range(len(time)):
        weights = psi[i, metal_index] / (total_mass * pdraw)
        contribution[i] = np.bincount(delay_index, weights=weights, minlength=len(widths)) / widths
        assert np.isclose(np.sum(contribution[i] * widths), weights.sum(), rtol=1e-12)

    groups = np.arange(ci.nRedshiftsDetection + 1)
    logging.info("Building detected-rate operator for %d DCOs", len(delays))
    fast = FixedPopulationRates(ci, np.array([0., np.inf]), groups)
    detected = fast(*PARAMETERS).ravel()
    complete_ci = CompleteSelectionIntegrator(population, selection,
                                             p_MaxRedshift=10., p_MaxRedshiftDetection=1.5)
    logging.info("Building complete-selection operator")
    complete = FixedPopulationRates(complete_ci, np.array([0., np.inf]), groups)
    unselected = complete(*PARAMETERS).ravel()
    assert np.all(np.isfinite(detected)) and np.all(detected >= 0)
    assert np.all(detected <= unselected * (1 + 1e-12))
    doubled = list(PARAMETERS)
    doubled[2] *= 2
    assert np.allclose(fast(*doubled).ravel(), 2 * detected, rtol=1e-12)

    # Independent cached contraction for all matching 32M population blocks.
    # The full archive's sampling bounds were used to construct this cache.
    pdraw_full = 1 / np.log(meta["zmax"] / meta["zmin"])
    rho, _, _ = ci.FindZdistribution(ci.redshifts, np.log(meta["zmin"]),
                                   np.log(meta["zmax"]), p_Alpha=alpha, p_Sigma=sigma)
    formation = ci.CalculateSFR(ci.redshifts, amplitude, exponent) / total_mass
    if population.nSystems not in (32_000_000, 512_000_000):
        raise ValueError("Reference caches support only the 32M and 512M archives")
    block_count = population.nSystems // 32_000_000
    cached = sum((load_npz(meta_path.parent / f"block_{i:02d}.npz")
                  for i in range(1, block_count)),
                 load_npz(meta_path.parent / "block_00.npz"))
    reference = (cached @ (formation[:, None] * rho / pdraw_full).ravel()).reshape(-1, len(groups)-1).sum(axis=0)
    # Only the sampling-range convention differs; correct it explicitly.
    reference *= pdraw_full / pdraw
    assert np.allclose(reference, detected, rtol=1e-11, atol=1e-10)

    data = dict(time_edges=time_edges, metals=metals, psi=psi,
                delay_edges=delay_edges, lnz_edges=lnz_edges, yield_density=yield_density,
                contribution=contribution, redshift_edges=groups * ci.redshiftStep,
                detected=detected, unselected=unselected)
    manifest = dict(parameters=dict(zip(["alpha", "sigma_lnZ", "a_SFR", "d_SFR"], PARAMETERS)),
                    population=str(population_path), initial_systems=int(population.nSystems),
                    reference_blocks=list(range(block_count)),
                    selected_dcos=len(delays), mass_per_initial_system_Msun=population.massEvolvedPerBinary,
                    metallicity_bounds=[population.zamsZmin, population.zamsZmax],
                    legacy_metallicity_order_matches=population.legacy_metallicity_order_matches,
                    cosmology="WMAP9", selection="O3; SNR threshold 8; random seed 0",
                    detected_rate_per_observer_year=float(detected.sum()),
                    complete_rate_per_observer_year=float(unselected.sum()),
                    cached_rate_relative_error=float(np.max(abs(reference-detected)) / np.max(detected)),
                    max_delay_Gyr=float(delays.max()), present_age_Gyr=float(tmax),
                    checks=["metallicity PDF integrates to one", "formation integrates to SFR",
                            "yield histogram retains all selected DCOs", "delay histogram conserves formation weights",
                            "detected rate bounded by complete selection", "linear amplitude scaling",
                            "agreement with matching independent block rate caches"],
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), meta_path,
                                             ROOT / "src/cosmic_integration/ratesSampler/fixed_population.py",
                                             ROOT / "src/cosmic_integration/ratesSampler/ratesSampler.py"]})
    return data, manifest


def draw(data: dict, out: Path) -> None:
    """Render four panels with explicit density measures and integration geometry."""
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "axes.linewidth": .7, "pdf.fonttype": 42, "legend.frameon": False})
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.0))
    fig.subplots_adjust(left=.105, right=.92, bottom=.1, top=.95, hspace=.37, wspace=.64)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#ededed")
    cmap.set_under(cmap(0))

    def mesh(ax, x, y, values, limits, label):
        im = ax.pcolormesh(x, y, np.ma.masked_less_equal(values, 0),
                           cmap=cmap, norm=LogNorm(*limits), rasterized=True, shading="flat")
        cb = fig.colorbar(im, ax=ax, pad=.025, fraction=.046, extend="min")
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        return im

    # Display the full finite metallicity grid used to normalize star formation.
    metals = data["metals"]
    du = np.diff(np.log(metals))[0]
    metal_edges = np.exp(np.r_[np.log(metals)-du/2, np.log(metals[-1])+du/2])
    ax = axs[0, 0]
    mesh(ax, data["time_edges"], metal_edges, data["psi"].T, (1e-5, 1),
         r'$\psi_u\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}]$')
    ax.set(xlabel=r'Formation time $t_{\rm f}$ [Gyr]', ylabel=r'Metallicity $Z$', yscale="log")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    # Dashed boundaries identify simulation support, without concealing the PDF.
    for bound in np.exp(data["lnz_edges"][[0, -1]]):
        ax.axhline(bound, color="white", linestyle="--", linewidth=.65)

    ax = axs[0, 1]
    positive = data["yield_density"][data["yield_density"] > 0]
    limits = (10**np.floor(np.log10(positive.min())), 10**np.ceil(np.log10(positive.max())))
    mesh(ax, data["delay_edges"], np.exp(data["lnz_edges"]), data["yield_density"], limits,
         r'$Y(u,t_{\rm d})\;[M_\odot^{-1}\,\mathrm{Gyr}^{-1}]$')
    ax.set(xlabel=r'Delay time $t_{\rm d}$ [Gyr]', ylabel=r'Metallicity $Z$', yscale="log")

    ax = axs[1, 0]
    values = data["contribution"].T
    vmax = 10**np.ceil(np.log10(values.max()))
    mesh(ax, data["time_edges"], data["delay_edges"], values, (vmax/1e5, vmax),
         r'$C\;[\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}\,\mathrm{Gyr}^{-1}]$')
    ax.set(xlabel=r'Formation time $t_{\rm f}$ [Gyr]', ylabel=r'Delay time $t_{\rm d}$ [Gyr]')
    tmin, tmax = data["time_edges"][[0, -1]]
    xx = np.linspace(tmin, tmax, 300)
    for tm, style in [(4., ":"), (8., "--"), (tmax, "-")]:
        valid = xx < tm
        ax.plot(xx[valid], tm-xx[valid], color="white", lw=1, ls=style)
        xf = max(tmin+.3, tm*.42)
        ax.text(xf, tm-xf+.25, rf'$t_{{\rm m}}={tm:.1f}$', color="white", fontsize=7,
                rotation=-42, ha="center", va="bottom")
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0, data["delay_edges"][-1])

    ax = axs[1, 1]
    edges = data["redshift_edges"]
    ax.stairs(data["unselected"]/np.diff(edges), edges, color=".25", lw=1.3,
              linestyle="--", label=r'$p_{\rm detect}=1$')
    ax.stairs(data["detected"]/np.diff(edges), edges, color="#275c88", lw=1.5,
              label='O3 selection')
    ax.set(xlabel=r'Merger redshift $z_{\rm m}$', ylabel=r'$\mathrm{d}R/\mathrm{d}z_{\rm m}\;[\mathrm{yr}^{-1}]$',
           xlim=(0, 1.5))
    # Retain zeros and negligible positive values without a misleading
    # thirty-decade logarithmic range. The caption specifies this scale.
    ax.set_yscale("symlog", linthresh=1., linscale=.5)
    ax.set_ylim(0, 1e6)
    ax.set_yticks([0, 1, 1e2, 1e4, 1e6])
    ax.legend(loc="upper left", fontsize=8)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    for label, ax in zip("abcd", axs.flat):
        ax.text(0, 1.025, f"({label})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10)
        assert not ax.get_title()
    assert not fig.texts, "Explanatory equations belong in the appendix text"
    fig.savefig(out / "cosmic_integration_physical.pdf", bbox_inches="tight")
    fig.savefig(out / "cosmic_integration_physical.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path,
                        default=ROOT / "tests/large_test_data/h5out_512M_reduced.h5")
    parser.add_argument("--outdir", type=Path, default=ROOT / "output/pdf/cosmic_integration_512M")
    parser.add_argument("--replot", action="store_true", help="Use the saved numerical snapshot")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        with np.load(args.outdir / "physical_arrays.npz") as saved:
            data = dict(saved)
    else:
        data, manifest = calculate(args.population)
        np.savez_compressed(args.outdir / "physical_arrays.npz", **data)
        (args.outdir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    draw(data, args.outdir)
    # Keep numerical-generation provenance when only the layout is revised.
    manifest_path = args.outdir / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["render_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
