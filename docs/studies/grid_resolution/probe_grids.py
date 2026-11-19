"""Plot data-free COMPAS grid diagnostics; no likelihood or GP training.

Run from the repository root:
  .venv/bin/python docs/studies/grid_resolution/probe_grids.py --compas-h5 FILE

The existing 0.1 integration grid is unchanged. Output pixels are exact nested
aggregations; no resampling of binaries and no observed catalogue is involved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.binned_cosmic_integrator import MakeChirpMassBins
from cosmic_integration.ratesSampler.grid_diagnostics import (
    effective_size, nested_mass_edges, shape_js, system_grid_moments,
)
from cosmic_integration.ratesSampler.parameter_grid import (
    ALPHA_VALUES, SIGMA_VALUES, SFR_A_VALUES, SFR_D_VALUES,
)

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEYS = ["p_Alpha", "p_Sigma", "p_SFRa", "p_SFRd"]
FID = np.array([-.325, .213, .012, 4.253])
BOUNDS = np.array([[x.min(), x.max()] for x in
                   [ALPHA_VALUES, SIGMA_VALUES, SFR_A_VALUES, SFR_D_VALUES]])
LABELS = ["alpha", "sigma", "SFR amplitude", "SFR exponent"]


def configurations() -> dict:
    right, _ = MakeChirpMassBins()
    regular = np.arange(16)
    mixed = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 15])
    return {
        "native": (nested_mass_edges(right, 1), regular, "5% mass; dz=0.1"),
        "mass10": (nested_mass_edges(right, 2), regular, "~10% mass; dz=0.1"),
        "mass20": (nested_mass_edges(right, 4), regular, "~20% mass; dz=0.1"),
        "mass10_mixedz": (nested_mass_edges(right, 2), mixed, "~10% mass; variable dz"),
        "mass20_mixedz": (nested_mass_edges(right, 4), mixed, "~20% mass; variable dz"),
    }


def parameter_points() -> list[tuple[str, np.ndarray]]:
    points = [("fiducial", FID.copy())]
    for i in range(4):
        for j, side in enumerate(["low", "high"]):
            theta = FID.copy()
            theta[i] = BOUNDS[i, j]
            points.append((f"{KEYS[i]}_{side}", theta))
    points += [("joint_low", BOUNDS[:, 0]), ("joint_high", BOUNDS[:, 1])]
    return points


def provenance(path: Path, grids: dict) -> dict:
    source_paths = sorted((REPO / "src/cosmic_integration/ratesSampler").glob("*.py"))
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in source_paths)).hexdigest()
    return {"input": str(path), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "rates_source_sha256": digest,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": 0, "selection": "existing O3 approximation; all selected DCO types",
            "integration_dz": .1, "grids": {k: {"mass_edges":
            [float(v) if np.isfinite(v) else "infinity" for v in e],
            "z_edges": (g * .1).tolist()} for k, (e, g, _) in grids.items()}}


def run(path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    grids, points = configurations(), parameter_points()
    meta = provenance(path, grids)
    signature = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
    cache = out / f"cache_{signature}"
    cache.mkdir(exist_ok=True)
    results, timings = {}, []
    ci = None
    for name, theta in points:
        file = cache / f"{name}.npz"
        if file.exists():
            with np.load(file) as data:
                results[name] = {k: data[k] for k in data.files}
            LOG.info("Cached %s", name)
            continue
        if ci is None:
            start = time.perf_counter()
            ci = BinnedCosmicIntegrator.from_compas_fpath(str(path))
            meta["load_seconds"] = time.perf_counter() - start
            LOG.info("Loaded population in %.2fs", meta["load_seconds"])
        start = time.perf_counter()
        rates, masses = ci.FindDetectionRate(p_BinaryFraction=.7, **dict(zip(KEYS, theta)))
        compute = time.perf_counter() - start
        assert rates.shape[1] == 15, "This study requires the native 15 z cells"
        row = {"theta": theta, "system_count": np.array(len(masses))}
        whole = rates.sum(axis=1)
        row["whole_neff"] = np.array(whole.sum()**2 / np.sum(whole**2))
        for key, (edges, groups, _) in grids.items():
            total, square = system_grid_moments(rates, masses, edges, groups)
            np.testing.assert_allclose(total.sum(), rates.sum(), rtol=1e-12)
            row[key] = total
            row[f"{key}_neff"] = effective_size(total, square)
        row["compute_seconds"] = np.array(compute)
        row["all_grids_seconds"] = np.array(time.perf_counter() - start - compute)
        np.savez_compressed(file, **row)
        results[name] = row
        LOG.info("%s: %.1f/yr, N_eff=%.0f; integration %.2fs, grids %.2fs",
                 name, rates.sum(), row["whole_neff"], compute, row["all_grids_seconds"])
    rows = []
    for name, theta in points:
        row = results[name]
        timings.append({"point": name, "integration_seconds": float(row["compute_seconds"]),
                        "all_grids_seconds": float(row["all_grids_seconds"])})
        native_js = shape_js(row["native"], results["fiducial"]["native"])
        for key, (edges, groups, _) in grids.items():
            rate, neff = row[key], row[f"{key}_neff"]
            order = np.argsort(neff.ravel())
            cumulative = np.cumsum(rate.ravel()[order]) / rate.sum()
            median = neff.ravel()[order[np.searchsorted(cumulative, .5)]]
            js = shape_js(rate, results["fiducial"][key])
            rows.append({"point": name, "grid": key, "cells": rate.size,
                         "rate_per_year": float(rate.sum()),
                         "whole_neff": float(row["whole_neff"]),
                         "rate_weighted_median_neff": float(median),
                         "rate_fraction_neff_lt_10": float(rate[neff < 10].sum()/rate.sum()),
                         "rate_fraction_neff_lt_30": float(rate[neff < 30].sum()/rate.sum()),
                         "empty_cell_fraction": float(np.mean(rate == 0)),
                         "shape_js_nats": js,
                         "shape_js_retained": js/native_js if native_js > 1e-12 else None,
                         **dict(zip(KEYS, theta))})
    with (out / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta.update({"timings": timings, "cache": str(cache),
                 "parameter_points": {n: t.tolist() for n, t in points}})
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    plot_maps(results["fiducial"], grids, out)
    plot_parameter_slices(results, grids, out)
    plot_parameter_maps(results, grids, out)
    plot_tradeoff(rows, grids, out)


def plot_maps(row: dict, grids: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, len(grids), figsize=(16, 7), layout="constrained")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f5f8fc", "#2563a6", "#092c52"])
    for col, (key, (edges, groups, title)) in enumerate(grids.items()):
        # Guard bins retained numerically; finite positive mass bins shown as density.
        e, r = edges[1:-1], row[key][1:-1]
        z = groups * .1
        density = r / (np.diff(np.log(e))[:, None] * np.diff(z)[None, :])
        im = axes[0, col].pcolormesh(e, z, np.ma.masked_less_equal(density.T, 0),
                                    cmap=cmap, norm=LogNorm(.01, 2000), rasterized=True)
        en = row[f"{key}_neff"][1:-1]
        ine = axes[1, col].pcolormesh(e, z, np.ma.masked_where(r.T <= 0, en.T),
                                     cmap=cmap, norm=LogNorm(1, 2000), rasterized=True)
        axes[0, col].set_title(f"{title}\n{row[key].size} cells incl. mass guards", fontsize=10)
        for ax in axes[:, col]:
            ax.set_xscale("log")
            ax.set_xlim(.5, e[-1])
            ax.set_ylim(0, 1.5)
            ax.set_xlabel(r"Source chirp mass [$M_\odot$]")
    axes[0, 0].set_ylabel("Redshift")
    axes[1, 0].set_ylabel("Redshift")
    fig.colorbar(im, ax=axes[0, :], label=r"detection rate / d ln $\mathcal{M}$ / dz [yr$^{-1}$]", extend="both")
    fig.colorbar(ine, ax=axes[1, :], label="Per-pixel system Kish effective size", extend="max")
    fig.suptitle("32M COMPAS: output-grid resolution and weight concentration\n"
                 "Fiducial parameters; fixed O3 selection and dz=0.1 integration; white = no model support", fontsize=13)
    fig.savefig(out / "grid_comparison.png", dpi=170)
    plt.close(fig)


def plot_parameter_slices(results: dict, grids: dict, out: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), layout="constrained")
    colors = ["#2563a6", "#414141", "#cb741a"]
    edges = grids["native"][0]
    for i, key in enumerate(KEYS):
        for name, color, ls in zip([f"{key}_low", "fiducial", f"{key}_high"], colors, ["--", "-", ":"]):
            r = results[name]["native"]
            val = results[name]["theta"][i]
            axes[i, 0].stairs(r[1:-1].sum(1)/np.diff(np.log(edges[1:-1])), edges[1:-1],
                              color=color, linestyle=ls, label=f"{val:g}")
            axes[i, 1].stairs(r.sum(0)/.1, np.arange(16)*.1, color=color, linestyle=ls)
        axes[i, 0].set_xscale("log")
        axes[i, 0].set_xlim(.5, edges[-2])
        axes[i, 0].set_ylabel(r"d rate / d ln $\mathcal{M}$ [yr$^{-1}$]")
        axes[i, 1].set_ylabel(r"d rate / dz [yr$^{-1}$]")
        axes[i, 0].legend(title=LABELS[i], fontsize=9)
        axes[i, 0].set_xlabel(r"Source chirp mass [$M_\odot$]")
        axes[i, 1].set_xlabel("Redshift")
    fig.suptitle("32M COMPAS: parameter dependence of predicted detections\n"
                 "One-at-a-time prior endpoints and fiducial; native output grid; no observations", fontsize=13)
    fig.savefig(out / "parameter_dependence.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(rows: list[dict], grids: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for x, (key, (_, _, label)) in enumerate(grids.items()):
        selected = [r for r in rows if r["grid"] == key]
        for ax, metric, scale in [(axes[0], "rate_fraction_neff_lt_30", 100),
                                   (axes[1], "shape_js_retained", 100)]:
            values = [scale*r[metric] for r in selected if r[metric] is not None]
            ax.plot(np.full(len(values), x), values, "o", color="#2563a6", alpha=.45, ms=5)
            ax.plot(x, np.median(values), "_", color="#202020", ms=20, mew=2)
    for ax in axes:
        ax.set_xticks(range(len(grids)), [v[2].replace("; ", "\n") for v in grids.values()], fontsize=9)
        ax.grid(axis="y", color=".9")
    axes[0].set_ylabel("Predicted rate in pixels with effective size <30 [%]")
    axes[1].set_ylabel("Shape separation retained versus native grid [%]")
    axes[1].set_ylim(0, 103)
    fig.suptitle("Output-grid trade-offs across parameter probes\n"
                 "Dots: probes; bars: medians. Shape separation = JS to fiducial; amplitude-only probes excluded", fontsize=12)
    fig.savefig(out / "grid_tradeoff.png", dpi=170)
    plt.close(fig)


def plot_parameter_maps(results: dict, grids: dict, out: Path) -> None:
    """Compare normalized shapes on the provisional fixed mass10 grid."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    edges, groups, _ = grids["mass10"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_shape", ["#f5f8fc", "#2563a6", "#092c52"])
    for ax, (name, row) in zip(axes.flat, results.items()):
        r = row["mass10"]
        density = r[1:-1] / r.sum() / (np.diff(np.log(edges[1:-1]))[:, None] * .1)
        im = ax.pcolormesh(edges[1:-1], groups*.1,
                          np.ma.masked_less_equal(density.T, 0),
                          cmap=cmap, norm=LogNorm(1e-5, 10))
        ax.set_xscale("log")
        ax.set_xlim(.5, edges[-2])
        ax.set_ylim(0, 1.5)
        if name.startswith("p_"):
            key = name.rsplit("_", 1)[0]
            i = KEYS.index(key)
            title = f"{LABELS[i]} = {row['theta'][i]:g}"
        else:
            title = name.replace("_", " ")
        ax.set_title(f"{title}; {r.sum():.1f}/yr", fontsize=10)
        ax.set_xlabel(r"Source chirp mass [$M_\odot$]")
        ax.set_ylabel("Redshift")
    axes.flat[-1].axis("off")
    fig.colorbar(im, ax=list(axes.flat), label=r"Normalized density per d ln $\mathcal{M}$ dz", extend="both")
    fig.suptitle("32M COMPAS: predicted shapes at eleven parameter settings\n"
                 "Fixed ~10% mass bins and dz=0.1; normalization removes total-rate differences", fontsize=13)
    fig.savefig(out / "parameter_maps.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compas-h5", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.compas_h5.resolve(), args.outdir.resolve())
