# Paired BO count-history results — 13 September 2026

Current manuscript presentation: Figures 2–3 show 25-point sparse GP contours and shaded stopped GP regions and KL
to validated 1258-point GPs. Figure 5 is the consolidated paired width/cost
summary (`acquisition_summary.pdf`, data in `paired_acquisition_summary.json`).
The earlier standalone convergence and cost figures remain here as diagnostics,
not manuscript figures. The original numerical findings are unchanged.

Start with **ANALYSIS.md** for findings, scope, exceptions and figure captions.
This is a local, inspectable results bundle. It has not been uploaded, committed,
or deposited in a public archive; a local ZIP is not an off-machine backup.

## Tables (UTF-8 CSV)

- `stops.csv`: 70 rows, one measurement view per catalogue. `training`, `audit`
  and `total` are likelihood-call counts at the recorded stop; `comparisons`
  excludes initialization. `maximum_directed_kl` is the maximum over both KL
  directions and both integration designs. `symmetric_kl` is the mean of the
  directions and designs. KL values are in nats.
- `histories.csv`: 2,870 rows, one job/checkpoint. Key: `(name, points)`.
  `points = 58 + 30 * posterior_comparisons`. `elapsed_seconds` is saved
  trajectory elapsed time, not total policy runtime. Numerical flags retain
  unresolved checkpoints. Late-GP KL includes the final zero self-comparison:
  filter `late_reference_is_self` when plotting it, and check
  `late_reference_accepted` before treating the endpoint as a reference.
- `marginals.csv`: 45,920 rows. Key:
  `(name, points, target, scramble, parameter)`. Targets are `direct` and `gp`;
  scrambles 0 and 1 are independent numerical integration designs, not MCMC
  chains. Parameters are alpha, sigma, a and d, in native physical units
  (the plotted amplitude scaling of 100 is not applied). Columns give mean,
  SD, and 2.5%, 16%, 50%, 84%, 97.5% quantiles. Direct summaries are repeated
  across checkpoints for joins; do not count them as independent results.

Join stops to histories on `name` and `training = points`. Join marginals on
`name` and `points`, retaining target, scramble and parameter distinctions.
There are 35 independent latent catalogues with paired measurement views,
not 70 independent catalogues; `expected_events` is an expectation, not the
realized count. Five pairs per setting share one fixed fiducial injection.

## Figures and posterior draws

PDFs are vector outputs; PNGs are previews. `posterior_stop_perfect` and
`posterior_stop_uncertain` are the manuscript's updated paired corners.
The four `*_draws.npz` files contain 131,072 plotting draws per saved state:
`reference`, `25`, and the recorded stopping label. Arrays have columns
`alpha, sigma, a, d`. These are resampled numerical posterior representations,
not independent MCMC chains. `*_figure.json` records actual event counts,
stop costs, source hashes and draw-versus-summary validation.

`snapshot.json` freezes campaign membership and outcome/history input hashes.
`bundle_manifest.json` hashes every other bundled file. Verify the hashes with
Python's `hashlib.sha256` after unpacking. This bundle supports later table
analysis and reuse of plotted posterior draws without the large COMPAS archive.

## Raw evidence and reproduction

The original campaign remains at the sibling directory
`../count_history_20260912/`. It contains `manifest.json`, `campaign_status.json`,
`stops/`, `training/`, `offline/`, normalized posterior weights at every saved
checkpoint, numerical checks, and frozen runtime/source snapshots.
Each `offline/<job>/reference.json` identifies the external cached importance
coordinates/rates and their hashes. The raw normalized weights need those
coordinates to reconstruct a posterior; they are not standalone sample arrays.
The full campaign and external rate/COMPAS caches are **not** inside this ZIP.
Retain them to rerun raw numerical validation or training.

From the package repository root, reproduce the summary figures with:

```bash
MPLCONFIGDIR=/tmp/compas-count-analysis-mpl .venv/bin/python \
  docs/studies/operational_stopping/gpry_compas/analyze_count_history.py \
  --root docs/studies/operational_stopping/gpry_compas/count_history_20260912 \
  --output NEW_OUTPUT_DIRECTORY
```

Rebuild the manuscript corner figures and CSV exports using the existing
analysis directory (requires the repository and hash-identified rate caches):

```bash
MPLCONFIGDIR=/tmp/compas-count-analysis-mpl .venv/bin/python \
  docs/studies/paper_completion/count_history_manuscript.py
```

A copy of each plotting script is included for inspection; repository imports
and cached inputs are required for full regeneration. No new jobs are launched.
Current manuscript build: `overleaf/output/count_history_review/ms.pdf`.

## Additional 100-point fits

`early100/<job>/` stores the four independently refitted early GP states,
normalized posterior weights, numerical scores and fit provenance. Each uses
the first 100 labels in the saved acquisition order and the frozen training
runtime (58 initial labels plus 14 batches of three). Hyperparameters are
refitted with the same 250 optimization steps and seed 17. These are extra
diagnostic fits, separate from the original 2,870 checkpoint rows; original
trajectories and stops are unchanged. The archived 100-point corner draws use NPZ key `100`; current top-level
corner draws use key `25`. The `early100/` fits are retained as an optional
diagnostic and are no longer required by the manuscript renderer. Reproduce first with `fit_100_point_corners.py`, then
`score_100_point_corners.py`, then `count_history_manuscript.py`, all under
`docs/studies/paper_completion/` using the project Python environment.

## Sparse initialization comparisons and revised styling

The manuscript now uses 25-point fits; `preview50/` provides the alternative
50-point corner PDFs, PNGs, plotting draws and metadata. `early25/` and
`early50/` retain fitted checkpoints, normalized weights, scores and provenance;
`small_initialization.csv` summarizes all eight results. All eight numerical
checks pass, while all eight posterior-accuracy checks fail. Symmetric KL
ranges are 1.71–5.49 nats (25 points) and 0.45–1.76 (50 points).
These are fixed prefixes of the initialization, not new BO trajectories:
25 Sobol labels versus 32 Sobol plus 18 boundary anchors. The original
campaign begins at 58 labels and is unchanged. Fewer points need not produce
worse fits monotonically: all four 50-point cases are closer than their
58-point counterparts in symmetric KL. The original 58-point results remain
in histories.csv; 100-point fits remain under early100/.

Reproduce: run `fit_small_initialization.py --points 25` and `--points 50`,
then `score_small_initialization.py` with each count, then
`preview_50_point_corners.py` and `count_history_manuscript.py`, from
`docs/studies/paper_completion/` using the project Python environment.
Running the main manuscript renderer last restores the 25-point main figures.
Blue filled regions are stopped-GP 50%/90% credible regions; dashed grey
lines show the early fit. KL curves retain only their recorded stopping marker.
Figure 5 now labels measurement columns explicitly and uses E[N_obs], N_train
and N_audit notation. The audit curve is total cost, not a random baseline;
the older BO/random benchmark remains a separate appendix result.
