# Corner and stopping-panel redesign — 2026-09-11

Current review PDF: `overleaf/output/corner_review/ms.pdf` (18 pages).
There are still three main figures and five overall. The first two main figures
now contain four corner plots: 100/1000 expected detections with perfect and
uncertain measurements. The realised counts are 97/979. Each exposure uses the
same latent catalogue across the two measurement models.

Every displayed corner shows only the saved 118-evaluation BO posterior, the
posterior at its actual recorded stop, and the injection. The numerical-reference
overlay is removed. Discussion of a separate direct-computation method is removed
from the active manuscript. A short appendix specification identifies the
high-resolution numerical reference used internally, including its independent
importance designs, refinements and accuracy tolerances. It is not described as
a densely trained GP. Raw validation records and computations are preserved.

## Stopping display

The old multiple-curve insets are replaced by aligned lower panels. A single
curve shows max(KL/0.003, bootstrap-95th-percentile/0.005, 32/minimum-ESS).
The horizontal line is one. Marker fill indicates whether the two-consecutive-
resolved-stability condition also holds; a star marks the first joint acceptance.
All four panels share time and audit-score axes. This is a new display of the
frozen rule, with no new tolerance or acquisition tuning.

| Expected events | Measurements | Training at stop | Audit calls | Total |
|---|---|---:|---:|---:|
| 100 | Perfect | 598 | 672 | 1270 |
| 100 | Uncertain | 358 | 288 | 646 |
| 1000 | Perfect | 478 | 480 | 958 |
| 1000 | Uncertain | 538 | 576 | 1114 |

The smaller examples show why event count alone does not determine stopping
cost. The smaller uncertain catalogue passes its audit before it passes posterior
stability. The first fiducial catalogue identifier is used at both exposures;
examples were not chosen by ranking their posterior agreement. The full fresh
ensemble's premature acceptance and conservative non-stops remain in the paper.

## Rendering and checks

Corner plots use common full-prior axes, 50%/90% smoothed marginal contours,
a muted early/stop palette and light stopped-posterior fill. Covariance-oriented
Gaussian smoothing removes finite-sample texture while adapting to the widths
and correlations of each posterior. The bandwidth is 0.2 posterior standard
deviations with a 0.45-bin resolution term on a 64-bin grid and reflection at
prior boundaries. Normalization is retained inside the prior. Diagonal smoothing
uses the matching marginal bandwidth. These are display choices; all quantitative
checks and stopping decisions use unsmoothed weights and are unchanged.

Both independent numerical reconstructions resolve the early and stopped
posteriors in all four cases. Plotting-draw means agree with saved validated
summaries within 0.004 posterior SD and endpoints within 0.016 SD. The fast suite
passes 146 tests (three deselected), including checks that the display score
retains all audit requirements and that boundary smoothing preserves probability.
The final PDF pass has no undefined references, overfull boxes or stuck floats.
All rendered pages were inspected; the text contains no direct-computation
comparison language. The original likelihood and simulation-generation sections
remain unchanged. Nothing was committed, pushed or synced remotely.

Rebuild figures from cached, checked draws:

```sh
MPLCONFIGDIR=/tmp/compas-paper-mpl .venv/bin/python docs/studies/paper_completion/presentation_figures.py --plots-only
```

Omit `--plots-only` to reconstruct any missing cases from frozen checkpoints.
The new `n100_*` and `n1000_*` sample/metadata files, source hashes and
`render_manifest.json` are in `presentation_20260911/`. The preceding text is
retained in `overleaf/archive/corner_layout_before_20260911/`.
