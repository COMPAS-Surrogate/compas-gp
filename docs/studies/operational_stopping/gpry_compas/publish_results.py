"""Paper-style COMPAS convergence figures from frozen, offline-scored runs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"correct_counter": "Four-correct counter", "counter_and_gaussian": "Counter + Gaussian KL",
          "joint_stability": "Joint posterior stability"}
BLUE = "#1681BA"


def plot_panel(ax, curves: np.ndarray, grid: np.ndarray, stops: list[int], total: int,
               regime: tuple, limits: tuple, individual: bool = False) -> None:
    if individual:
        for i, curve in enumerate(curves):
            ax.plot(grid, curve, color=BLUE, lw=1.35, ls=["-", "--", ":"][i], label=f"Catalogue {i}")
    else:
        q25, median, q75 = np.quantile(curves, [.25, .5, .75], axis=0)
        ax.fill_between(grid, q25, q75, color=BLUE, alpha=.2, zorder=2)
        ax.plot(grid, median, color=BLUE, lw=1.6, zorder=3)
    if stops:
        if len(stops) > 1:
            a, b = np.quantile(stops, [.25, .75])
            ax.axvspan(a, b, color=".55", alpha=.2, zorder=1)
        ax.axvline(np.median(stops), color=".45", ls=":", lw=1.5, zorder=4)
        ax.plot(stops, np.full(len(stops), .025), "|", color=".3", ms=8,
                transform=ax.get_xaxis_transform(), clip_on=False)
    n, mode = regime
    ax.text(.97, .97, f"{n} expected events\n{mode} measurements", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
    ax.text(.04, .05, f"{len(stops)}/{total} stopped", transform=ax.transAxes, fontsize=9, color=".3")
    ax.set_yscale("log"); ax.set_ylim(*limits)
    ax.set_xlim(grid[0]-10, grid[-1]+10)
    ax.set_xticks(np.arange(100, grid[-1]+1, 100))
    ax.grid(which="major", color=".68", lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    assert json.loads((args.input/"offline/status.json").read_text())["state"] == "complete"
    manifest = json.loads((args.input/"offline/manifest.json").read_text())
    frozen = (args.input/"decisions.json").read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == manifest["decision_sha256"]
    verification = json.loads((args.input/"verification.json").read_text())
    assert verification["state"] == "passed" and verification["decision_sha256"] == manifest["decision_sha256"]
    decisions = json.loads(frozen)
    records = json.loads((args.input/"offline/records.json").read_text())
    outcomes = json.loads((args.input/"offline/outcomes.json").read_text())
    maximum = min(d["total_labels"] for d in decisions.values())
    grid = np.arange(58, maximum+1, 30)
    regimes = [(n, mode) for n in [100, 1000] for mode in ["perfect", "uncertain"]]
    data = {}
    for regime in regimes:
        n, mode = regime
        selected = [r for r in records if r["expected_events"] == n and r["mode"] == mode]
        names = sorted(set(r["name"] for r in selected))
        curves = []
        for name in names:
            by_n = {r["points"]: r for r in selected if r["name"] == name}
            assert all(p in by_n for p in grid), "A common-budget curve is missing a checkpoint"
            curves.append([by_n[p]["symmetric_kl"] for p in grid])
        data[regime] = (np.asarray(curves), names)
    values = np.concatenate([a.ravel() for a, _ in data.values()])
    assert np.isfinite(values).all() and np.all(values > 0), "Log plot needs positive finite estimates"
    limits = (10**np.floor(np.log10(values.min())), 10**np.ceil(np.log10(values.max())))
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
                         "axes.labelsize": 13, "savefig.dpi": 200})
    for rule in LABELS:
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True, sharey=True)
        for ax, regime in zip(axes.ravel(), regimes):
            curves, names = data[regime]
            stops = [decisions[name]["rules"][rule] for name in names if decisions[name]["rules"][rule] is not None]
            plot_panel(ax, curves, grid, stops, len(names), regime, limits)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{sym}}$")
        for ax in axes[-1]:
            ax.set_xlabel("Number of COMPAS likelihood evaluations")
        handles = [Line2D([], [], color=BLUE, label="Median KL"),
                   Patch(facecolor=BLUE, alpha=.2, label="KL interquartile range"),
                   Line2D([], [], color=".45", ls=":", label="Median stopping count"),
                   Patch(facecolor=".55", alpha=.2, label="Stopping interquartile range")]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .9])
        for ext in ["png", "pdf"]:
            fig.savefig(args.outdir/f"convergence_{rule}.{ext}", bbox_inches="tight")
        plt.close(fig)
    # Individual panels matching the supplied example's footprint.
    for regime, (curves, names) in data.items():
        n, mode = regime; rule = "counter_and_gaussian"
        stops = [decisions[name]["rules"][rule] for name in names if decisions[name]["rules"][rule] is not None]
        fig, ax = plt.subplots(figsize=(6.1, 4.4))
        plot_panel(ax, curves, grid, stops, len(names), regime, limits)
        ax.set_xlabel("Number of COMPAS likelihood evaluations")
        ax.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{sym}}$")
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(args.outdir/f"n{n}_{mode}.{ext}", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True, sharey=True)
    for ax, regime in zip(axes.ravel(), regimes):
        curves, names = data[regime]
        stops = [decisions[name]["rules"]["counter_and_gaussian"] for name in names
                 if decisions[name]["rules"]["counter_and_gaussian"] is not None]
        plot_panel(ax, curves, grid, stops, len(names), regime, limits, individual=True)
    for ax in axes[:, 0]: ax.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{sym}}$")
    for ax in axes[-1]: ax.set_xlabel("Number of COMPAS likelihood evaluations")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, .94])
    for ext in ["png", "pdf"]:
        fig.savefig(args.outdir/f"individual_runs.{ext}", bbox_inches="tight")
    plt.close(fig)
    summary = []
    for rule, label in LABELS.items():
        selected = [r for r in outcomes if r["rule"] == rule]
        counts = Counter(r["outcome"] for r in selected)
        stops = [r["points"] for r in selected if r["points"] is not None]
        summary.append({"rule": rule, "label": label, "counts": dict(counts),
                        "median_stop": float(np.median(stops)) if stops else None})
    (args.outdir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.outdir/"plot_data.json").write_text(json.dumps({"grid": grid.tolist(), "regimes": [
        {"expected_events": n, "mode": mode, "names": names, "symmetric_kl": a.tolist()}
        for (n, mode), (a, names) in data.items()]}, indent=2)+"\n")
    table = "\n".join(f"| {s['label']} | {s['counts'].get('accurate', 0)} | {s['counts'].get('premature', 0)} | "
        f"{s['counts'].get('reference_unresolved', 0)} | {s['counts'].get('budget_limit', 0)} | {s['median_stop']} |"
        for s in summary)
    detail = "\n".join(f"| {name} | {d['rules']['correct_counter']} | {d['rules']['counter_and_gaussian']} | "
                       f"{d['rules']['joint_stability']} |" for name, d in decisions.items())
    failed = [r for r in outcomes if r["outcome"] in ["premature", "reference_unresolved", "budget_limit"]]
    failures = "\n".join(f"- {r['name']}, {LABELS[r['rule']]}: {r['outcome']}"
        +(f" at {r['points']} labels; symmetric KL {r['symmetric_kl']:.5g}." if r['points'] is not None else ".") for r in failed)
    total = sum(d["total_labels"] for d in decisions.values())
    train_time = sum(d["wall_seconds"] for d in decisions.values())
    numerical_failures = sum(not (r["reference_numerical"] and r["gp_numerical"]) for r in records)
    combined_stops = [r for r in outcomes if r["rule"] == "counter_and_gaussian" and r["points"] is not None]
    max_combined_sym = max(r["symmetric_kl"] for r in combined_stops)
    max_combined_directed = max(r["maximum_directed_kl"] for r in combined_stops)
    precision_note = ""
    if (HERE/"precision_diagnostic.json").exists():
        diagnostic = json.loads((HERE/"precision_diagnostic.json").read_text())
        assert diagnostic["exact_likelihood_calls"] == 0
        assert not diagnostic["stopping_decisions_changed"]
        precision_note = """A separate final-checkpoint diagnostic doubled the GP-only integration
design from 16384 to 32768 base points per replicate. The unresolved uncertain,
count 1000 case then passed numerical precision, with **zero extra COMPAS
likelihood calls**. Its original stopping decision and the plots are unchanged.
This diagnoses unnecessary continuation caused by numerical error; it does not
retroactively establish an earlier online stop. Refine GP-only integration
before spending more exact evaluations when the numerical check fails.
See [the diagnostic](../precision_diagnostic.json) and
[its script](../check_precision.py).
"""
    text = f"""# GPry-style stopping on the COMPAS LnL surrogate

Implemented in the existing JAX/GPJax environment without installing or vendoring
GPry. Twelve new BO trajectories cover three fresh paired catalogue/BO-seed
realisations at each combination of expected count100/1000 and perfect/uncertain
measurements. Three stopping criteria are monitored on every shared trajectory.

**Counter plus Gaussian KL is the most useful candidate in this experiment.**
It reduced premature stopping from four cases to one, with a median stopping
count of 328 likelihood labels. One further run did not resolve the GP-only
integration check. The joint-stability comparator had the same overall outcome
counts and a median of 358 labels; its premature case was a different catalogue.
This twelve-run result is not a guarantee that the rule meets our strict accuracy
criterion on a new problem, so it has not been enabled as a production default.

| Candidate | Accurate stops | Premature stops | Numerically unresolved stops | No stop by budget | Median labels at stop |
|---|---:|---:|---:|---:|---:|
{table}

Medians include only runs that stopped. Accuracy requires independent numerical
agreement, both directed KLs <.01, mean and interval-endpoint errors <.2 SD, and
68%/95% width errors <10%. No stop is counted as success merely because its
own convergence monitor passed.

Every combined-rule stop had symmetric KL at most {max_combined_sym:.5f}. The one
classified as premature failed the stricter requirement that **each direction**
be below 0.01: its maximum directed KL was {max_combined_directed:.5f}. Its largest
interval-endpoint discrepancy was 0.10 posterior SD, and its 95% width error was
below 2.6%. The four-correct rule alone also had a clearer failure, with symmetric
KL 0.066 and an interval-endpoint discrepancy of 0.30 SD. These distinctions
matter when interpreting the curves and the binary accuracy table.

{precision_note}

## Convergence plots

![Counter and Gaussian KL stopping](convergence_counter_and_gaussian.png)

Blue lines show the median symmetric KL to direct COMPAS posterior estimates;
blue shading is the interquartile range across three runs per panel. Grey
shading and the dotted line show the interquartile range and median stopping
count for the counter-plus-Gaussian-KL rule. Bottom ticks show individual stops.
The number stopped is printed in every panel; censored runs are never assigned
a stopping count. Bands are descriptive between-run spread, not confidence
intervals. Perfect/uncertain panels share the underlying mock catalogues.

All curves continue to {maximum} labels. Decisions are frozen when the candidate
first passes; subsequent labels are validation continuation. The common plotted
grid is every30 labels beginning at58. Extra saved checkpoints score stops that
occur between those plotting points. There is no smoothing, monotonic envelope,
interpolation between different run lengths, or replacement of direct truth by
a larger GP. Symmetric KL is half the sum of the two directed KLs, as in the
paper; stopping stability uses the documented maximum-directed convention.

Other stopping overlays: [four-correct counter](convergence_correct_counter.png),
[joint posterior stability](convergence_joint_stability.png).
Vector exports: [preferred-rule PDF](convergence_counter_and_gaussian.pdf),
[counter PDF](convergence_correct_counter.pdf), [joint-stability PDF](convergence_joint_stability.pdf).

Individual example-format panels: [100 perfect](n100_perfect.png),
[100 uncertain](n100_uncertain.png), [1000 perfect](n1000_perfect.png),
[1000 uncertain](n1000_uncertain.png). Each also has a same-named PDF.

![Individual COMPAS trajectories](individual_runs.png)

The companion displays all three trajectories in each regime. Catalogue labels
0/1/2 correspond to BO seeds5011/5012/5013. These fresh catalogues are all generated
at the same fixed physical injection. Three repetitions do not establish a
rare-failure probability or simulation-based calibration.

## What was borrowed and what was adapted

The [GPry paper](https://arxiv.org/html/2211.02045v2) supplies the posterior-weighted
uncertainty acquisition, small Kriging-believer batches, and Eq.16 prediction
counter. The [current documentation](https://gpry.readthedocs.io/en/latest/module_convergence.html)
also supports combining a prediction check with Gaussian posterior stability.
The experiment follows the paper's prediction-based relative tolerance, rather
than the actual-label form in current documentation.

For our three-dimensional shape inference, epsilon_abs=.0352674038,
epsilon_rel=.01, and four consecutive correct labels are required. Predictions
and the best-known LnL are saved before the whole batch is evaluated. A later
failure within a batch resets an earlier successful streak. The combined rule
requires the counter still to pass at a 30-label checkpoint, alongside two
Gaussian KL<.02 passes. The third rule requires two joint KL<.001 passes plus
stable moments and intervals, on the same checkpoint schedule.

This is a COMPAS adaptation, not a run of the GPry package. The existing
Matérn-5/2 kernel, sqrt compression, all training points, full prior and analytic
amplitude marginalization are retained. No SVM is used. GP hyperparameters and
scaling are refitted every30 labels; every three-label batch is conditioned
exactly between refits. JAX gradients optimize acquisition from multiple starts.
The nonlinear sqrt inverse's predictive uncertainty is propagated through its
second/fourth moments; it is not treated as a linearly rescaled Gaussian in LnL.
Uncertainty guides acquisition and is not assumed calibrated by a passing counter.

Fixed-capacity arrays remove compilation triggered solely by growth in the
training-array shape. Numerical tests compare cached means/variances and
Kriging-believer conditioning with GPJax, compare the transformed uncertainty
with numerical integration, and compare acquisition autodiff with finite
differences. The capacity slots are algebraically inactive padding, not data
subsetting. The rate evaluator and its physical model are unchanged.

## Exact stop counts and exceptional outcomes

| Run | Counter | Counter + Gaussian KL | Joint stability |
|---|---:|---:|---:|
{detail}

`None` denotes no acceptance by the finite evaluation budget.

{failures if failures else 'All selected checkpoints passed the independent accuracy criteria.'}

## Evidence and reproducibility

The experiment used {total} training/continuation likelihood labels. The sum of
per-run measured wall times was {train_time:.1f}s; separate offline scoring took
{manifest['wall_seconds']:.1f}s. Population setup and catalogue generation are
additional. These figures are not an end-to-end speedup comparison. Earlier
studies used different initial designs and refit schedules, so their label
counts are not a matched acquisition baseline for this experiment.

The expensive reference-rate arrays already existed. A separate scoring process
read them only after all decisions were saved, then formed each new catalogue's
direct likelihood and posterior weights. Two independently scrambled importance
designs assess numerical precision. There were {numerical_failures} numerically
unresolved checkpoint scores out of {len(records)}; plotted curves are numerical
estimates, and unresolved stop scores are explicitly excluded from successes.

The [frozen protocol](../PROTOCOL.md) specifies the experiment. In the input
directory, `manifest.json` and `population/source_snapshot` capture configuration
and source, `batches.json` saves pre-training predictions and exact labels,
`checkpoint_*.npz` saves each scored GP, and `offline/` contains reference hashes,
scores and decision-hash verification. `plot_data.json` supplies every plotted
curve. The failed first smoke and corrected small smoke are preserved and excluded.

The original v1 implementation passed all 119 fast tests (3 heavy tests
deselected). A separate prefix replay verified all {verification['batches_replayed']}
batches, {verification['checkpoint_hashes']} checkpoint hashes and all stopping
decisions. It reconstructed {verification['pre_training_predictions_checked']}
pre-training predictions, with maximum discrepancy
{verification['max_pre_training_prediction_difference']:.3g} LnL. Twelve spot
checks of the cached reference rates matched the online COMPAS rate evaluator
exactly. The input directory's `verification.json` and offline manifest contain
the machine-readable evidence.

From the repository root, with the existing `.venv`, set
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-mpl` and run:

```bash
.venv/bin/python docs/studies/operational_stopping/gpry_compas/run_experiment.py --outdir NEW_RUN_DIRECTORY
.venv/bin/python docs/studies/operational_stopping/gpry_compas/score_experiment.py --input NEW_RUN_DIRECTORY
.venv/bin/python docs/studies/operational_stopping/gpry_compas/verify_experiment.py --input NEW_RUN_DIRECTORY
.venv/bin/python docs/studies/operational_stopping/gpry_compas/publish_results.py --input NEW_RUN_DIRECTORY --outdir NEW_REPORT_DIRECTORY
```

No GPry installation is needed. The source COMPAS block caches and existing
reference-rate arrays are required to reproduce this benchmark. The online
learner itself does not load reference arrays. Production stopping defaults
remain unchanged pending assessment of these results.
"""
    text = re.sub(r"\b(from|to|of|below|every|at|counts?|seeds?)(?=[0-9])", r"\1 ", text)
    (args.outdir/"RESULTS.md").write_text(text.replace("Eq.16", "Eq. 16"))
    hashes = {str(p.relative_to(args.input)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [args.input/"decisions.json", args.input/"offline/records.json", args.input/"offline/outcomes.json"]}
    (args.outdir/"provenance.json").write_text(json.dumps({"input": str(args.input.resolve()), "input_sha256": hashes,
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+"\n")


if __name__ == "__main__":
    main()
