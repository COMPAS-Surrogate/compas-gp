# Reference-KL comparison revision — 2026-09-12

The manuscript now uses a comparison target rather than an operational stopping
claim: symmetric reference KL below 0.01 nats at two consecutive saved
checkpoints, selecting the second. Symmetric KL is half the sum of the two
directed divergences, averaged across two independent integration designs.
Unresolved numerical estimates cannot select a checkpoint. Practical stopping
without a reference is deferred to future work.

The same exact-likelihood numerical reference supplies the blue corner contours
and their KL trajectories. The 2000-label GP references remain archived, but
are no longer the plotted reference. Each curve shows one catalogue through
all 21 saved checkpoints, without smoothing or an invented ensemble band.
An open marker preserves the one early numerical-integration disagreement.

## Selected posteriors

| Expected events | Measurements | Selected evaluations |
|---|---|---:|
| 100 | Perfect | 358 |
| 1000 | Perfect | 388 |
| 100 | Uncertain | 268 |
| 1000 | Uncertain | 358 |

The full BO/random benchmark is replayed with this same rule. Parameter widths
are recomputed from the selected checkpoint summaries, not the old termination
records. BO selects all 600 catalogues; random selects 597. Three random runs
remain above their saved budgets in the evaluation distributions.

| Expected events | Measurements | Median BO evaluations | Median random evaluations |
|---|---|---:|---:|
| 30 | Perfect | 482 | 610 |
| 100 | Perfect | 610 | 866 |
| 300 | Perfect | 610 | 1122 |
| 1000 | Perfect | 738 | 1378 |
| 50 | Uncertain | 482 | 610 |
| 500 | Uncertain | 610 | 1122 |

The separate paired-data pilot uses the same criterion: both 30-event views
select 482 evaluations and both 1200-expected-event views select 610. Rate
maps and replicated-catalogue checks are rebuilt from these posteriors.
The small-batch and large-batch acquisition schedules remain distinct.
Fixed-budget robustness and finite-population controls remain separate checks.

Original acquisition trajectories, labels, decisions and scientific priors are
unchanged. All new outputs are under `reference_kl_20260912/`. Source hashes,
84 checkpoint KL estimates, four corner sample bundles, 1200 selection records,
600 parameter-width records and predictive provenance are retained there.
Earlier manuscript sections and figures are preserved under
`overleaf/archive/reference_kl_before_20260912/`.

## Reproduction

Run from the package repository:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-paper-mpl .venv/bin/python docs/studies/paper_completion/reference_kl_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-paper-mpl .venv/bin/python docs/studies/paper_completion/reference_selected_predictive.py
python3 /Users/avi/.codex/plugins/cache/openai-bundled/latex/0.2.6/scripts/compile_latex.py overleaf/ms.tex --compiler tectonic --output-directory overleaf/output/reference_kl_review --json
.venv/bin/python -m pytest -q
```

Review PDF: `overleaf/output/reference_kl_review/ms.pdf` (22 pages). The changed
corner, rate-map, evaluation-summary and predictive pages were rendered and
inspected. The fast suite passed 170 tests, with three deselected and one
existing Matplotlib deprecation warning. The final TeX pass has no undefined
references or overfull boxes; two underfull data-availability lines remain.
Checkpoint source hashes, all 1200 selections and all 600 selected parameter
widths were independently replayed and checked. Corner draws are checked
against reference and GP marginal summaries. Nothing was committed or pushed.
