# GPry-style COMPAS stopping experiment

The latest [periodic-audit development replay](stable_scheduled_v2/report/RESULTS.md)
gives 24/24 accurate BO stops after retaining joint-posterior stability and
removing the prediction-counter and pointwise-error vetoes. Fixed 96-point
audits cost less than adaptive expansion (median totals 646 versus 856).
The [audit-only candidate](scheduled_v1/report/RESULTS.md) is preserved: it
accepted all BO cases but had one/two premature stops for fixed/adaptive audits.
These revisions use already-examined catalogues and are development evidence.
See [the revised protocol](STABLE_SCHEDULED_AUDIT_PROTOCOL.md),
`run_stable_scheduled_audit.py`, `score_scheduled_audit.py`, and
`publish_scheduled_audit.py`. Production stopping defaults remain unchanged.

The current extension compares BO with IID random sampling on 24 matched
likelihood cases (12 independent latent catalogues, each with paired perfect
and uncertain measurements). It varies three four-parameter injections and
tests four stopping monitors. Read [the extension results](report_v2/RESULTS.md),
[initial protocol](EXTENSION_PROTOCOL.md), and
[posterior-mass audit protocol](MASS_AUDIT_PROTOCOL.md).

The preceding twelve-run experiment and its development evidence remain in
[the original report](report_v1/RESULTS.md) and [protocol](PROTOCOL.md).

This implements the paper's prediction-counter idea and posterior-weighted
acquisition directly in the existing JAX runtime. GPry is not installed or
vendored. The kernel, sqrt transform and COMPAS likelihood remain those of the
existing project; consequently this is an adaptation, not the complete GPry
algorithm. Training keeps all points and the full prior.

- `run_experiment.py`: generates fresh COMPAS catalogues and runs the online
  acquisition/stopping monitors, continuing after proposed stops for validation.
- `score_experiment.py`: independently reveals direct references afterward.
- `publish_results.py`: reproduces PNG/PDF curves, bands, stopping markers and
  the report from saved outputs.
- `verify_experiment.py`: replays all decisions from training prefixes, checks
  hashes and reconstructs pre-training predictions.
- `check_precision.py`: diagnoses the numerical non-stop at one frozen final
  checkpoint using more GP-only integration points.
- `ensemble_v1/`: completed twelve-trajectory experiment and offline scores.
- `smoke_v1/`: preserved empty-pending-array failure; `smoke_v2/`: corrected
  development smoke, excluded from the ensemble.

Core numerical and counter functions live in
`src/cosmic_integration/lnl_surrogate/gpry_rules.py` at the repository root.
Production stopping defaults have not been changed.

## BO versus random extension

- `compare_random.py`: matched, continuous-coordinate training and three frozen
  stopping monitors, with every trajectory continued through 658 labels.
- `run_mass_audit.py`: independent posterior-mass audits of eligible frozen
  checkpoints, recorded before direct-reference scoring.
- `score_comparison.py`: independent reference scoring and exact amplitude
  reconstruction, checking all four physical marginal posteriors at each stop.
- `verify_comparison.py`: replay of pairing, counters, pre-training predictions,
  audits, source hashes and exact-call accounting.
- `publish_comparison.py`: reproducible curves, stopping costs, tables and
  manuscript text. Numerical failures and non-stops remain in the report.
- `extension_v2/`: training checkpoints, held-out audits and separate offline
  scores. `pipeline_status.json` records completion of the full sequence.
- `report_v2/`: figures, results, plot data and provenance.

The mass-audit variant uses injection 1 for development and injections 2/3
for confirmation (16 likelihood cases per method). Its bootstrap is a finite
audit diagnostic, not a guaranteed error bound. All training points are kept;
audits remain held out to preserve the matched acquisition trajectories.

The physical model has four parameters. The GP has three inputs because the
amplitude is integrated analytically. Training and stopping require no
parameter-space rate grid. Reconstructing amplitude without cached rates still
requires total-rate evaluations at posterior points, beyond the reported
training and audit counts.
