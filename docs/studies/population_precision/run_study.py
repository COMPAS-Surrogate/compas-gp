"""Direct cached 8M-block study; no GP and no new COMPAS simulations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.special import logsumexp

from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.population_stream import load_selected_population
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.ratesSampler.grid_diagnostics import system_grid_moments, effective_size
from cosmic_integration.observation.grid_likelihood import aggregate_nested_grid
from cosmic_integration.observation.log_measurement import bin_coefficients
from cosmic_integration.observation.population_precision import (
    poisson_loss, bootstrap_coefficients, unsupported_fraction, log_bin_coefficients,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "grid_resolution"))
from probe_grids import configurations

LOG = logging.getLogger(__name__)
TRUTH = np.array([-.3, .35, .01, 4.7])  # Same injection as current Figure 9.
SIZES = np.array([8, 16, 32, 64, 128, 256, 512])
GRIDS = ["native", "mass10", "mass20"]
BLOCK = 8_000_000
REPLICATES = 128


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(out: Path) -> tuple[dict, dict]:
    """Cache direct per-system moments and five nearby sigma predictions."""
    h5 = REPO / "tests/large_test_data/h5out_512M_reduced.h5"
    sources = [Path(__file__), *sorted((REPO / "src/cosmic_integration/ratesSampler").glob("*.py")),
               REPO / "src/cosmic_integration/observation/population_precision.py"]
    signature = dict(input=str(h5), bytes=h5.stat().st_size, mtime_ns=h5.stat().st_mtime_ns,
                     source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sources}, block_size=BLOCK, blocks=64,
                     truth=TRUTH.tolist(), selection_seed=0)
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Source/input provenance changed: choose a new --out directory")
    write_json(manifest, signature)
    grids = configurations()
    cache = out / "blocks"
    cache.mkdir(exist_ok=True)
    sigma = np.array([.25, .30, .35, .40, .45])
    files = [cache / f"block_{b:02d}.npz" for b in range(64)]
    if not all(p.exists() for p in files):
        ci = BinnedCosmicIntegrator.from_compas_fpath(
            str(REPO / "tests/large_test_data/h5out_32M_reduced.h5"))
        pop = load_selected_population(h5, ci.compas.massEvolvedPerBinary)
        assert pop.nSystems == 64 * BLOCK
        write_json(out / "population.json", dict(selected=len(pop.mass1),
                   mass_per_binary=pop.massEvolvedPerBinary, zmin=pop.zamsZmin, zmax=pop.zamsZmax,
                   legacy_metallicity_order_matches=pop.legacy_metallicity_order_matches))
        for b, path in enumerate(files):
            if path.exists():
                continue
            chosen = pop.system_rows // BLOCK == b
            part = copy.copy(pop)
            for attr in ["mass1", "mass2", "Zsystems", "delayTime", "dcoSeeds", "system_rows"]:
                setattr(part, attr, getattr(pop, attr)[chosen])
            part.nSystems = BLOCK
            temporary = copy.copy(ci)
            temporary.compas = part
            direct, masses = temporary.FindDetectionRate(
                p_BinaryFraction=.7, p_Alpha=TRUTH[0], p_Sigma=TRUTH[1],
                p_SFRa=TRUTH[2], p_SFRd=TRUTH[3])
            data = dict(selected=np.array(chosen.sum()), sigma=sigma)
            for name in GRIDS:
                edges, groups, _ = grids[name]
                total, square = system_grid_moments(direct, masses, edges, groups)
                raw, _ = system_grid_moments((direct > 0).astype(float), masses, edges, groups)
                data.update({name: total, name + "_square": square, name + "_raw": raw})
            data["total_square"] = np.sum(direct.sum(axis=1)**2)
            model = FixedPopulationRates(temporary, *grids["native"][:2])
            save_npz(cache / f"operator_{b:02d}.npz", model.operator)
            predictions = []
            for s in sigma:
                theta = TRUTH.copy()
                theta[1] = s
                predictions.append(model(*theta))
            data["sigma_native"] = np.array(predictions)
            np.testing.assert_allclose(predictions[2], data["native"], rtol=2e-11, atol=1e-13)
            np.savez_compressed(path, **data)
            LOG.info("Direct block %d/64: %d selected systems; rate %.3f/yr", b + 1, chosen.sum(), direct.sum())
    entries = [dict(np.load(p)) for p in files]
    arrays = {key: np.array([row[key] for row in entries]) for key in entries[0]}
    # Independent match to the previously computed Figure 9 rates.
    old = np.load(HERE.parent / "paper_completion/likelihood_loss_20260912/rates.npz")
    for i, size in enumerate(old["sizes"]):
        np.testing.assert_allclose(arrays["mass10"][:size // 8].mean(0).ravel()[old["active"]],
                                   old["rates"][i], rtol=2e-10, atol=1e-12)
    return arrays, grids


def measurement_losses(
    rates: np.ndarray, coefficients: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    draws: int, seed: int,
) -> dict:
    """Integrate common measurements for every population using block rates.

All candidate populations have support contained in the full reference. The
per-measurement integrand r - 1 - log(r) includes the Poisson rate term.
"""
    rng = np.random.default_rng(seed)
    reference = rates.mean(0)
    candidates = coefficients @ rates
    count = len(coefficients)
    sums, squares = np.zeros(count), np.zeros(count)
    fallback = 0
    for start in range(0, draws, 1024):
        n = min(1024, draws - start)
        bins = rng.choice(len(reference), n, p=reference / reference.sum())
        latent = rng.uniform(lower[bins], upper[bins])
        rho = 12 * rng.random(n)**(-1 / 3)
        scales = np.array([.03, .3])[None, :] * 12 / rho[:, None]
        y = np.log(latent) + scales * rng.normal(size=(n, 2))
        sensor = bin_coefficients(y, scales, lower, upper)
        sensor /= sensor.max(axis=1, keepdims=True)
        intensities = sensor @ rates.T
        baseline = intensities.mean(1)
        ratio = (intensities @ coefficients.T) / baseline[:, None]
        with np.errstate(divide="ignore"):
            log_ratio = np.log(ratio)
        bad = np.argwhere(~np.isfinite(log_ratio))
        # Gaussian response has full support: numerical underflow is not a
        # physical zero. Resolve only affected event/population pairs in logs.
        for row in np.unique(bad[:, 0]):
            log_sensor = log_bin_coefficients(y[row:row+1], scales[row:row+1], lower, upper)[0]
            with np.errstate(divide="ignore"):
                log_ref = logsumexp(log_sensor + np.log(reference))
                for column in bad[bad[:, 0] == row, 1]:
                    log_ratio[row, column] = logsumexp(log_sensor + np.log(candidates[column])) - log_ref
        fallback += len(bad)
        delta = ratio - 1
        loss = delta - log_ratio
        small = abs(delta) < 1e-4
        loss[small] = delta[small]**2 * (.5 - delta[small] / 3 + delta[small]**2 / 4)
        assert np.isfinite(loss).all() and np.all(loss >= -1e-12)
        sums += loss.sum(0)
        squares += (loss**2).sum(0)
    mean = sums / draws
    se = np.sqrt(np.maximum(squares / draws - mean**2, 0) / (draws - 1))
    return dict(mean=mean, se=se, fallback=np.array(fallback), draws=np.array(draws))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--draws", type=int, default=262144)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tick = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    arrays, grids = prepare(out)
    nested = np.array([np.r_[np.ones(m // 8) / (m // 8), np.zeros(64 - m // 8)] for m in SIZES])
    bootstrap = np.concatenate([bootstrap_coefficients(64, m // 8, REPLICATES, 2026091200 + int(m)) for m in SIZES])
    coefficients = np.vstack([nested, bootstrap])
    np.save(out / "coefficients.npy", coefficients)
    support = []
    for name in GRIDS:
        block_rates = arrays[name].reshape(64, -1)
        ref = block_rates.mean(0)
        for i, size in enumerate(SIZES):
            k = size // 8
            total = block_rates[:k].sum(0)
            square = arrays[name + "_square"][:k].sum(0).ravel()
            neff = effective_size(total, square)
            raw = arrays[name + "_raw"][:k].sum(0).ravel()
            candidate = total / k
            loss = float(poisson_loss(ref, candidate))
            support.append(dict(grid=name, size=int(size), pixels=len(ref),
                positive_pixels=int(np.count_nonzero(candidate)),
                selected_systems=int(arrays["selected"][:k].sum()),
                rate=float(candidate.sum()), perfect_loss_per_event=loss if np.isfinite(loss) else None,
                perfect_loss_infinite=bool(not np.isfinite(loss)),
                missing_rate_fraction=float(unsupported_fraction(ref, candidate)),
                fraction_neff_lt10=float(ref[neff < 10].sum() / ref.sum()),
                fraction_neff_lt30=float(ref[neff < 30].sum() / ref.sum()),
                fraction_neff_lt100=float(ref[neff < 100].sum() / ref.sum())))
            np.savez_compressed(out / f"support_{name}_{size}.npz", reference=ref.reshape(arrays[name].shape[1:]),
                                rates=candidate.reshape(arrays[name].shape[1:]),
                                neff=neff.reshape(arrays[name].shape[1:]), raw=raw.reshape(arrays[name].shape[1:]))
    write_json(out / "support.json", dict(rows=support))
    rates = arrays["mass10"].reshape(64, -1)
    active = np.flatnonzero(rates.sum(0) > 0)
    edges, groups, _ = grids["mass10"]
    mi, zi = np.unravel_index(active, arrays["mass10"].shape[1:])
    lower = np.column_stack([edges[mi], groups[zi] * .1])
    upper = np.column_stack([edges[mi + 1], groups[zi + 1] * .1])
    assert np.isfinite(upper).all(), "Nonzero overflow requires an explicit sensor model"
    integrations = []
    for rep in range(2):
        path = out / f"measurement_{args.draws}_{rep}.npz"
        if path.exists():
            result = dict(np.load(path))
        else:
            result = measurement_losses(rates[:, active], coefficients, lower, upper,
                                        args.draws, 2026091300 + rep)
            np.savez(path, **result)
        integrations.append(result)
        LOG.info("Measurement integration %d: nested losses at N=1000 %s", rep, result["mean"][:7] * 1000)
    means = np.array([r["mean"] for r in integrations])
    errors = np.array([r["se"] for r in integrations])
    mean, se = means.mean(0), np.linalg.norm(errors, axis=0) / 2
    np.savez(out / "losses.npz", sizes=SIZES, nested=mean[:7], nested_se=se[:7],
             bootstrap=mean[7:].reshape(7, REPLICATES), bootstrap_se=se[7:].reshape(7, REPLICATES),
             integration_means=means, integration_se=errors)
    # A small direct sigma slice records parameter dependence without making
    # posterior-recovery claims. Perfect likelihood may have missing support.
    slices = arrays["sigma_native"]
    native_edges, native_groups, _ = grids["native"]
    mass10 = np.array([[aggregate_nested_grid(r, native_edges, native_groups, edges, groups)
                       for r in block] for block in slices])
    np.savez_compressed(out / "sigma_rates.npz", sigma=arrays["sigma"][0],
                        rates=mass10, truth=TRUTH)
    relevant = mean > 1e-10
    relative_se = se[relevant] / mean[relevant]
    summary = dict(state="complete" if np.max(relative_se) < .05 else "numerically_unresolved",
        seconds=time.perf_counter() - tick, draws_per_integration=args.draws,
        replicates=REPLICATES, maximum_relative_numerical_se=float(np.max(relative_se)),
        numerical_se_p95=float(np.quantile(relative_se, .95)),
        log_tail_fallback_pairs=sum(int(r["fallback"]) for r in integrations),
        nested_loss_at1000=(mean[:7] * 1000).tolist(),
        bootstrap_median_loss_at1000=(np.median(mean[7:].reshape(7, REPLICATES), axis=1) * 1000).tolist(),
        scope="Finite nested population comparison and conditional 8M-block bootstrap, nominal continuous measurements at one fixed injection",
        population_spread="16th–84th percentiles across 128 coherent bootstrap populations per size; not numerical error or independent campaigns",
        scaling="Exact linear scaling in exposure with fixed selection and measurement model",
        grid_support="Full-reference rate weighted; Neff thresholds are descriptive, not likelihood accuracy gates")
    write_json(out / "summary.json", summary)
    LOG.info("Completed: %s", summary)


if __name__ == "__main__":
    main()
