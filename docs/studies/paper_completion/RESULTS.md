# Manuscript completion — 2026-09-11

The subsequent presentation revision is documented in [PRESENTATION.md](PRESENTATION.md);
its three main figures and review PDF supersede the layout described below.

All runs in PLAN.md are complete. The main text retains the cosmic-integration
rate equation, perfect and uncertain catalogue log likelihoods, and mock-data
generation. It now has six main figures: physical rate maps, acquisition
requirements, paired perfect/uncertain accuracy, physical information versus
measurement uncertainty, COMPAS population-size sensitivity, and stopping
accuracy versus total evaluation cost.

## Completed evidence

- **600-catalogue acquisition benchmark:** 1,200 paired BO/random outcomes.
  After independently resolving the reference for perfect-measurement catalogue
  49 at 1,000 expected events, 1,197 runs reach the reference-assisted criterion;
  three random runs remain resource limited. All 100 catalogues per setting
  remain in figure denominators. The resolved pair finishes at 994 BO and 2,530
  random training evaluations. Original records are preserved; the publisher
  overlays only this independently checked pair.
- **Reference refinements:** the double-error, 500-event case and the preceding
  high-count reference are resolved with independent integrations. A pooled
  reference check additionally agrees for the high-count case. These checks
  address numerical reference precision, not generative calibration.
- **Fresh stopping confirmation:** 48 trajectories from 12 independent latent
  catalogues, paired into 24 perfect/uncertain likelihood cases. The frozen
  periodic 96-point audit plus two joint-stability passes gives 23 accurate BO
  stops and one premature stop. Random gives three accurate stops and 21
  non-stops. At the final checkpoint, 24 BO and six random fits are accurate.
  The premature BO stop has maximum directed KL 0.0128, above the 0.01 target.
  Median accepted training-plus-audit counts are 802 BO and 1,270 random;
  different accepted subsets prevent interpreting this as a paired speedup.
- **Population size versus catalogue size:** 36 direct posterior reconstructions:
  three catalogues each at 50, 300 and 1,000 expected detections, each analysed
  with nested 32M, 128M, 256M and 512M COMPAS populations. All numerical checks
  resolve after bounded refinement to 2^17 reference points per design.
  For 32M, the median maximum parameter-mean shift relative to the 512M
  posterior grows from 0.14 to 0.51 to 1.88 posterior standard deviations.
  The figure also reports central 68% interval-width changes and ranges across
  three catalogues. Larger populations are not monotonically ordered in every
  catalogue. The finite 512M population is a comparison, not astrophysical truth.

The supplied extrapolated likelihood-bias plot is replaced by measured changes
in normalized posteriors. A constant likelihood offset can cancel from inference;
the new plot tests the actual impact on the inferred parameters. It does not
establish a universal safe event count or minimum COMPAS size. Catalogues are
Poisson draws, so its horizontal axis uses expected observed events.

## Figure edits

The physical-information, population-size and fresh stopping-cost figures are
new. The full acquisition ensemble supersedes the smaller pilot as the main
budget comparison. The paired uncertainty pilot remains because it isolates
measurement effects on identical latent catalogues. Compact BO/random/Sobol
robustness and continuous posterior prediction are in the appendix.
Redundant initialization surfaces, full corner, categorical predictive-count,
output-grid and stopping-development figures are omitted from the manuscript;
the original files and study records remain available.

## Provenance and validation

Study directories (relative to docs/studies):

- `event_uncertainty/convergence_20260910/ensemble/`
- `paper_completion/reference_refinement_20260911/`
- `paper_completion/refined_case49_20260911/`
- `paper_completion/population_size_20260911/`
- `operational_stopping/gpry_compas/fresh_confirmation_20260911/`

`figure_data/sources.json` records source hashes. Confirmation replay verified
48 cases and 283 audit stages, pairing, checkpoint hashes and evaluation costs.
Frozen decision SHA-256:
`8f3cd6563d6fdaa4be6c48dadc8737e29ba278b99d627a7c424f99d5e5f81351`.
An initial offline scoring call used an obsolete API name; the separate
`score_fresh_confirmation.py` resumed scoring with the existing endpoint scorer.
This did not change training, audits or frozen decisions. Original source and
scoring provenance are retained separately.

Rebuild manuscript figure data and TeX fragments from completed records:

```sh
MPLCONFIGDIR=/tmp/compas-paper-mpl .venv/bin/python docs/studies/paper_completion/publish_completion.py
.venv/bin/python docs/studies/paper_completion/verify_confirmation.py
.venv/bin/python -m pytest -q
```

The fast suite passes: **142 passed, three deselected**, with one existing
Matplotlib deprecation warning. Manuscript PDF:
`overleaf/output/completion_review/ms.pdf` (24 pages including appendices).
Rendered pages were inspected: figures and tables are present and legible.
The build retains a RevTeX deferred-float warning and author-link annotation
warnings; no undefined references or overfull boxes were reported.

No further runs are required for this bounded mock-data revision. Prior-wide
continuous-sensor calibration, independently evolved COMPAS population replicates,
and end-to-end runtime comparisons remain outside its demonstrated claims.
The archive DOI, acknowledgements and computational accounting still need
submission preparation. Nothing was committed, pushed or synced to Overleaf.
