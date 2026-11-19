"""Build the COMPAS-only stopping report from frozen experiment outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    first, later = HERE/"compas_v1", HERE/"later_audit_v1"
    for folder in [first, later]:
        assert json.loads((folder/"status.json").read_text())["state"] == "complete"
    decisions = json.loads((first/"decisions.json").read_text())
    followup = json.loads((later/"decisions.json").read_text())
    scores = json.loads((first/"scores.json").read_text())
    rows = []
    for name, decision in decisions.items():
        frozen = (first/name/"decision.json").read_bytes()
        provenance = json.loads((first/name/"score_provenance.json").read_text())
        assert hashlib.sha256(frozen).hexdigest() == provenance["decision_sha256"]
        history = json.loads((first/name/"records.json").read_text())
        audit_row = next(r for r in history if "audit" in r)
        early, late = audit_row["audit"], followup[name]["audit"]
        split, count, mode, case = decision["spec"]
        ref = {r["rule"]: r for r in scores if r["case"] == name}
        rows.append({"case": name, "label": f"{count}, {mode}, {case}",
            "early_points": audit_row["points"], "late_points": followup[name]["points"],
            "early_calls_with_audit": audit_row["points"]+early["calls"],
            "late_calls_with_audit": followup[name]["calls"],
            "early_profile_error": early["profile_max_scaled_error"],
            "late_profile_error": late["profile_max_scaled_error"],
            "early_profile_pass": early["profile_passed"], "late_profile_pass": late["profile_passed"],
            "late_derivative_pass": late["passed"],
            "early_reference_kl": ref["gaussian_stability"]["reference_kl"],
            "late_reference_kl": ref["posterior_stability"]["reference_kl"]})
    status = json.loads((later/"status.json").read_text())
    assert hashlib.sha256((later/"decisions.json").read_bytes()).hexdigest() == status["decision_sha256"]
    (HERE/"comparison.json").write_text(json.dumps(rows, indent=2)+"\n")
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 5.7), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    colors = ["#276C9B", "#B76628"]
    for ax, keys in zip(axes, [("early_points", "late_points"),
                               ("early_profile_error", "late_profile_error")]):
        a, b = [[r[k] for r in rows] for k in keys]
        for i, (aa, bb) in enumerate(zip(a, b)):
            ax.plot([aa, bb], [i, i], color=".78", lw=1, zorder=1)
        ax.scatter(a, y, color=colors[0], marker="o", s=43, label="Moment stability", zorder=3)
        ax.scatter(b, y, color=colors[1], marker="s", s=37, label="Joint posterior stability", zorder=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=".92", lw=.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [r["label"] for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Expected events, measurement model, catalogue")
    axes[0].set_xlabel("Training likelihood evaluations at trigger")
    axes[0].set_xlim(430, 1060)
    axes[1].set_xlabel("Largest profile error / tolerance")
    axes[1].set_xlim(0, 4.5)
    axes[1].axvline(1, color=".3", ls="--", lw=1)
    axes[1].text(1.07, 1.025, "accept ≤ 1", transform=axes[1].get_xaxis_transform(), va="bottom", fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, .92], w_pad=2)
    for extension in ["png", "pdf"]:
        fig.savefig(HERE/f"stopping_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    table = "\n".join(f"| {r['label']} | {r['early_points']} | {r['late_points']} | "
        f"{r['early_profile_error']:.2f} | {r['late_profile_error']:.2f} | "
        f"{r['late_reference_kl']:.5f} |" for r in rows)
    lo = min(r["late_reference_kl"] for r in rows)
    hi = max(r["late_reference_kl"] for r in rows)
    report = f"""# COMPAS GP-LnL stopping: checkpoint experiments

**Prefer posterior stability as the stopping trigger. Use fresh likelihood
checks to test accuracy; do not require the local GP Hessian to converge.**
This ten-catalogue screen supports that design choice, but does not establish
a production stopping rule or a general failure probability.

## Tested rules and results

All decisions use training labels, successive GP fits, or small fresh audits.
No reference GP or dense exact LnL grid is used by a decision. Existing exact
references are revealed only for offline scoring. The first experiment is
checkpoint replay; the later audit is explicitly a development follow-up on
the same catalogues after seeing the first results.

| Candidate | Accurate selections | Premature selections | No acceptance on saved path |
|---|---:|---:|---:|
| Two entirely correct batches of new-label predictions | 0 | 0 | 10 |
| Prediction batches plus moment stability | 0 | 0 | 10 |
| Gaussian moment stability | 10 | 0 | 0 |
| Joint posterior stability | 10 | 0 | 0 |
| Moment stability + local derivative audit | 0 | 0 | 10 |
| Moment stability + profile/mode audit | 4 | 0 | 6 |
| Joint stability + local derivative audit, follow-up | 1 | 0 | 9 |
| Joint stability + profile/mode audit, follow-up | 7 | 0 | 3 |

The prediction rule borrows GPry's error scaling, but requiring two complete
128-label batches to pass is substantially stricter than GPry's consecutive
correct-prediction counter. Its rejection rate does **not** evaluate the actual
GPry implementation. The moment rule uses both directed Gaussian KLs <0.02 on
two successive updates. [GPry convergence documentation](https://gpry.readthedocs.io/en/latest/module_convergence.html).

Joint stability requires both directed joint KLs <0.001, changes in means and
68%/95% endpoints <0.05 posterior SD, and changes in 68% widths <3%, twice. Two
independently scrambled GP-only importance designs must resolve those same
summary tolerances and have ESS ≥100. Their proposal includes the full uniform
prior and a broad Student-t component; proposal refinement never changes the prior.

Joint stability selected 738–994 training labels. Its maximum directed
GP-versus-direct posterior KL ranged from {lo:.6f} to {hi:.6f}.
Offline accuracy additionally requires mean/endpoint shifts <0.2 SD, 68% and
95% width errors <10%, both KLs <0.01, and resolved reference integration.
The earlier moment trigger selected 482–610 labels; all ten already passed
these offline accuracy criteria. This cannot establish that moment stability
alone is safe outside these paths.

## Fresh likelihood checks and JAX

The profile audit evaluates the GP posterior mean, an autodiff-optimized GP
maximum, and ±1/±2 posterior-SD positions along three covariance directions.
These **14 locations** compare exact-minus-GP residuals relative to the centre,
so a constant LnL offset cancels. The tolerance is 0.1 + 0.01 times the LnL drop
from the best audit point. Directions reaching a prior boundary stop just inside it.

For this matched experiment, each audit also evaluated a two-resolution full
gradient/mixed-Hessian stencil: **50 exact likelihood calls per checkpoint**,
500 in each experiment, 1000 in total. The table and figure show training counts;
add 50 for the actual audit cost. A standalone profile implementation would add
14. These counts are likelihood labels, not a measured end-to-end speedup.

JAX differentiates the inverse-transformed GP mean, caching the GP solve once
per fit. Gradients drive bounded multistart L-BFGS-B mode finding. Curvature
and score differences are measured in posterior SD units. The exact evaluator
used here is `ratesSampler/fixed_population.py`, which is NumPy/SciPy; exact
derivatives are therefore assessed by matched finite differences. The adjacent
COMPAS GPU implementation uses NumPy/CuPy, not JAX. A separate JAX rate evaluator,
if available elsewhere, would allow an exact-score check, provided the complete
likelihood (including amplitude marginalization) also supports differentiation.

The local stencil residuals can be only ~0.01–0.05 LnL while local curvature
and its finite-difference refinement fail. Accurate integrated posteriors do
not require every local second derivative to meet the chosen threshold.
The derivative screen is therefore too restrictive here. This is not a failure
of JAX autodiff: cached GP means/gradients match the existing target, and its
autodiff Hessian passes an independent numerical derivative unit test.

![Stopping costs and unchanged profile-audit errors](stopping_comparison.png)

Each row is one COMPAS catalogue. Blue circles are the first moment-stability
trigger; orange squares are the first joint-stability trigger. The right panel
uses the identical audit tolerance at each checkpoint; values ≤1 pass. Waiting
for joint stability increases profile acceptance from 4/10 to 7/10. The three
remaining rejections are at posteriors that pass the stated offline accuracy
criterion: this audit remains conservative for that criterion.

| Expected events, measurements, case | Moment trigger | Joint trigger | Early profile error/tolerance | Later profile error/tolerance | Later reference KL |
|---|---:|---:|---:|---:|---:|
{table}

## Practical rule to take forward

1. Train with the existing full prior and all training labels. At checkpoints,
   require two consecutive numerically resolved joint-posterior stability passes.
2. Use JAX gradients to locate candidate GP maxima; evaluate the small fresh
   profile/mode audit before accepting the surrogate.
3. On audit failure, use the offending locations to guide more exact evaluations
   and refitting, then repeat with fresh audit points. Keep a resource cap and
   return an unresolved status if the tests do not pass.

Steps 1 and 2 were screened here. Step 3's adaptive continuation was **not**
executed in these checkpoint experiments. A posterior-weighted audit may reduce
unnecessary rejection relative to the maximum-error profile rule, but that is
a separate candidate to validate on COMPAS. Preserve broad acquisition because
these local checks do not measure whether an unvisited region has posterior mass.

The ten paths share the source BO training seed and fixed population/injection:
three catalogues each at expected counts 30, 100, 1000 with perfect measurements,
plus one independent pilot catalogue at count 500 with uncertain measurements.
They are three-parameter shape posteriors with analytically marginalized
amplitude, a fixed 256M COMPAS population, and the existing detected-rate grid.
The uncertain pilot is not an independent uncertainty ensemble. Source paths
were truncated by a reference-assisted stopping campaign, so non-acceptance is
censored. This tests surrogate accuracy conditional on that setup; it does not
test finite-population stability, real-data calibration, evidence accuracy, or
rare stopping failures.

## Reproduction and verification

From the repository root, use the existing `.venv` with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-mpl`:

```bash
.venv/bin/python docs/studies/operational_stopping/compas_geometry/run_study.py --outdir NEW_SCREEN_DIRECTORY
.venv/bin/python docs/studies/operational_stopping/compas_geometry/audit_later.py --input NEW_SCREEN_DIRECTORY --outdir NEW_AUDIT_DIRECTORY
.venv/bin/python docs/studies/operational_stopping/compas_geometry/publish_results.py
```

The report builder reads the archived `compas_v1` and `later_audit_v1` folders.
Original COMPAS block caches and source checkpoint/catalogue files must be
available; source locations and source snapshots are preserved in the manifests.
`comparison.json` contains chart/table data. Decisions were hashed before offline
scoring and their hashes are verified by the report builder. The source protocol
and [follow-up protocol](FOLLOWUP_PROTOCOL.md) record design adjustments.

Software validation: 109 fast tests passed, 3 heavy tests deselected, including
four new local-geometry tests. This verifies implementation, not statistical
reliability. No production stopping defaults or rate physics were changed.
"""
    (HERE/"RESULTS.md").write_text(report)


if __name__ == "__main__":
    main()
