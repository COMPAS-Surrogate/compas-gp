"""Prepare a noisy, matched-grid pilot and an analytic amplitude reference.

This is an event/grid consistency check, NOT a four-parameter posterior or GP
validation. The optional tail audit evaluates the 32M forward model directly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gamma

from cosmic_integration.lnl_computer import core_ln_likelihood
from cosmic_integration.observation.grid_likelihood import (
    aggregate_nested_grid, counts_log_likelihood,
)
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from probe_grids import BOUNDS, FID, KEYS, configurations, parameter_points, provenance


def audit_tail(path: Path, out: Path) -> None:
    """Measure exact per-system support above 40, without cutting any pixels."""
    ci = BinnedCosmicIntegrator.from_compas_fpath(str(path))
    rows = []
    for name, theta in parameter_points():
        rate, mass = ci.FindDetectionRate(p_BinaryFraction=.7, **dict(zip(KEYS, theta)))
        tail = rate[mass > 40].sum(axis=1)
        rows.append(dict(point=name, parameters=theta.tolist(),
                         above40_systems=int(np.sum(mass > 40)),
                         max_chirp_mass=float(mass.max()), total_rate=float(rate.sum()),
                         above40_rate=float(tail.sum()),
                         above40_fraction=float(tail.sum()/rate.sum()),
                         tail_neff=float(tail.sum()**2/(tail@tail)) if tail@tail > 0 else 0.))
    payload = {"provenance": provenance(path, configurations()), "points": rows}
    (out / "mass_tail_audit.json").write_text(json.dumps(payload, indent=2)+"\n")


def prepare(out: Path, seed: int, duration: float) -> None:
    grids = configurations()
    manifest = json.loads((out / "manifest.json").read_text())
    current = provenance(Path(manifest["input"]), grids)
    for key, value in current.items():
        if manifest.get(key) != value:
            raise ValueError(f"Stale forward cache ({key}); rerun probe_grids.py")
    cache = Path(manifest["cache"])
    with np.load(cache / "fiducial.npz") as data:
        reference = data["native"]
    rng = np.random.default_rng(seed)
    counts_native = rng.poisson(duration*reference)
    source_mass, source_z, _ = grids["native"]
    payload = {"counts_native": counts_native, "source_mass_edges": source_mass,
               "source_z_edges": source_z*.1}
    counts = {}
    for key, (edges, z, _) in grids.items():
        counts[key] = aggregate_nested_grid(counts_native, source_mass, source_z*.1, edges, z*.1)
        payload[f"counts_{key}"] = counts[key]
    # Direct evaluations already made in the grid study are reusable for each dataset.
    rows = []
    max_rate_aggregation_error = 0.
    for name, theta in parameter_points():
        with np.load(cache / f"{name}.npz") as data:
            for key, (edges, z, _) in grids.items():
                aggregated = aggregate_nested_grid(data["native"], source_mass, source_z*.1, edges, z*.1)
                err = float(np.max(np.abs(aggregated-data[key])))
                max_rate_aggregation_error = max(max_rate_aggregation_error, err)
                np.testing.assert_allclose(aggregated, data[key], rtol=1e-12, atol=1e-12)
                rows.append(dict(point=name, grid=key,
                                 log_likelihood=counts_log_likelihood(data[key], counts[key], duration)))
    with (out / "pilot_direct_probes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Under fixed other parameters, rates scale exactly as a / a_fid.
    n = int(counts_native.sum())
    beta = duration*reference.sum()/FID[2]
    low, high = BOUNDS[2]
    distribution = gamma(a=n+1, scale=1/beta)
    pl, ph = distribution.cdf([low, high])
    quantiles = distribution.ppf(pl + np.array([.025, .16, .5, .84, .975])*(ph-pl))
    amp = np.linspace(low, high, 2001)
    amplitude_curves = []
    for key in ["native", "mass10"]:
        with np.load(cache / "fiducial.npz") as data:
            rates = data[key]
        lnl = np.array([counts_log_likelihood(rates*a/FID[2], counts[key], duration) for a in amp])
        amplitude_curves.append(lnl-lnl.max())
    np.testing.assert_allclose(*amplitude_curves, rtol=1e-11, atol=1e-10)
    density = np.exp(amplitude_curves[1])
    cdf = np.r_[0., cumulative_trapezoid(density, amp)]
    cdf /= cdf[-1]
    numerical_q = np.interp([.025, .16, .5, .84, .975], cdf, amp)
    np.testing.assert_allclose(numerical_q, quantiles, atol=2e-8, rtol=0)
    # Both forms include the same exposure and omit data-only factorials.
    flat = np.repeat(np.arange(counts["mass10"].size), counts["mass10"].ravel())
    weights = np.zeros((n, counts["mass10"].size))
    weights[np.arange(n), flat] = 1.
    weights = weights.reshape(n, *counts["mass10"].shape)
    with np.load(cache / "fiducial.npz") as data:
        count_lnl = counts_log_likelihood(data["mass10"], counts["mass10"], duration)
        old_lnl = core_ln_likelihood(data["mass10"], duration, weights)
    np.testing.assert_allclose(count_lnl, old_lnl, atol=1e-10)
    payload.update(amplitude_grid=amp, amplitude_relative_lnl=amplitude_curves[1])
    np.savez_compressed(out / "pilot_catalogue.npz", **payload)
    summary = dict(seed=seed, duration_years=duration,
                   generation="Independent Poisson draws per native bin; same events coarsened",
                   scope="Perfect binned events; same finite population for generation and inference",
                   expected_events=float(duration*reference.sum()), observed_events=n,
                   max_rate_aggregation_error=max_rate_aggregation_error,
                   conditional_amplitude_quantiles=quantiles.tolist(),
                   quantile_probabilities=[.025, .16, .5, .84, .975],
                   max_amplitude_quantile_error=float(np.max(np.abs(numerical_q-quantiles))),
                   fixed_other_parameters=FID.tolist(), amplitude_prior=[float(low), float(high)],
                   forward_manifest=manifest,
                   warning="This is conditional 1D validation, not joint inference or SBC.")
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != 'forward_manifest'}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--duration", type=float, default=.1)
    parser.add_argument("--audit-tail", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    prepare(args.outdir.resolve(), args.seed, args.duration)
    if args.audit_tail:
        manifest = json.loads((args.outdir / "manifest.json").read_text())
        audit_tail(Path(manifest["input"]), args.outdir)
