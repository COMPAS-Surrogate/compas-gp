# Event count and acquisition study — frozen 2026-09-09

Primary question: does GP-guided BO allocate a limited number of population-model
parameter evaluations more effectively than space-filling exploration?

- One interior injection: alpha=-0.3, sigma=0.35, a_SFR=0.01, d_SFR=4.7.
- Exposures set to 30, 100, 300, 1000 expected detected events; ten independent
  Poisson catalogues at each exposure, seed 20261110. These are expected counts;
  observed counts fluctuate. No data or failures are discarded.
- Same blocks0–7 population for generation/inference; fixed full mass10 grid and
  O3 selection approximation; perfect binned measurements. No new COMPAS runs.
- Matched BO and Sobol: 128 Sobol + 98 boundary points initially; four batches128;
  checkpoints226/354/482/610/738; training seed17 and the same4096 candidate pool.
  BO selects by GP likelihood variance with10% uniform exploration; all training
  points retained. Candidate labels are accessed only after selection.
- One training seed here tests catalogue variation; earlier three-seed acquisition
  results are separate. This is a bounded fixed-injection study, not precise SBC.
- Physics controls: amplitude doubles all rates; expected Poisson log likelihood
  has its maximum at the generating intensity; finite-difference Poisson Fisher
  geometry and step-size consistency. Fractional expected counts are only a
  deterministic check, never treated as a coverage catalogue.
- Internal references: two independent scrambled-Sobol prior/Student-t mixture integrations,
  each generating16384 points before zero-prior filtering.
  The Student-t location uses the known injection and Fisher geometry ONLY to
  make validation efficient. Exact proposal-density weighting, full prior support,
  and independent numerical checks; no reference information enters acquisition.
  This is an importance-quadrature benchmark, not an analytical full posterior.
- Accuracy gate: means/endpoints within0.2 directSD;68/95 widths within10%; both
  KL directions<0.01. Numerical gate: means/endpoints across scrambles within0.05SD
  and68 widths within3%, for both methods. Report unresolved references separately.
- Main paper figures emphasize exploration, recovery and width scaling. Detailed
  direct-reference comparisons belong in the appendix. Figures have axes labels
  and necessary legends only, no titles or embedded descriptions; captions in TeX.
- Truth-based recovery alone is insufficient to prove posterior accuracy; keep
  reference gates and do not equate broad prior-like intervals with successful BO.
- Training-label counts are distinct from executed offline reference/cache work,
  and the timing includes reference scoring. No end-to-end speedup claim.

Run from the repository root:

```sh
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/event_count_bo/run_study.py
```

Results refuse overwrite. Source hashes, catalogues, reference proposals/rates,
checkpoints and model files are retained. No tuning within this confirmation run.

## Numerical reference refinement

The initial mixture used pseudorandom Student-t points and failed the numerical
precision gate at low counts. The incomplete attempt is preserved in
`initial_mc_reference_attempt/`. It was stopped before further fitting. The
reference was changed to scrambled Sobol inverse-CDF Student-t draws and doubled
to16384 generated points per scramble; GP/acquisition and catalogue seeds did
not change. Numerical precision is assessed independently, not tuned to make BO
win. The complete rerun is in `results/`.

## Completed results — 2026-09-09

Completed 40 independent catalogues, 80 acquisition trajectories and 400
checkpoint assessments. All 40 final numerical references pass independent
scramble checks after reference-only refinement. At 738 training labels:

| Expected events | BO combined passes | Sobol combined passes |
| --- | --- | --- |
| 30 | 10/10 | 10/10 |
| 100 | 10/10 | 8/10 |
| 300 | 10/10 | 5/10 |
| 1000 | 9/10 | 0/10 |

The combined gate includes both KL directions below 0.01 nats. BO passes the
parameter-interval checks in all 40 cases; Sobol passes them in 38. The remaining
BO combined failure (1000 expected events, case 5) is a KL failure, not a failed
marginal-interval check. BO has smaller forward KL in all 40 matched pairs;
this is descriptive evidence at one injection and one training seed.

Alpha's median 68% width decreases from 0.147 to 0.0448 across the exposure
range, whereas d_SFR remains near its prior width (0.676 to 0.638). Doubling the
SFR amplitude doubles all rates; halving the Fisher derivative step changes the
matrix by 1.7e-9 in relative norm. These controls validate model behavior, not
GP uncertainty calibration or finite-population robustness.

Three 100-event cases and four 1000-event cases required 65536 generated
reference points per scramble. Identical saved training designs were rescored;
no validation labels were added to acquisition. Both original and refined
results are retained. Final merged records are `results/validated_records.json`
and `results/validated_references.json`; `results/summary.json` drives the paper.
The main run took 776 s and refinements approximately 212 + 298 s, including
offline validation: these are not online BO timings or a speedup measurement.

```sh
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/event_count_bo/refine_reference.py --expected 100
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/event_count_bo/refine_reference.py --expected 1000
.venv/bin/python docs/studies/event_count_bo/summarize.py
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/event_count_bo/publish_figures.py
```

Validation: 62 fast tests passed, 3 heavy tests deselected; one existing
Matplotlib warning. Six title-free scientific figures and manuscript captions
are in the nested `overleaf` checkout. No new COMPAS simulations or LVK inference
were performed. Remaining work is broader injection/training-seed robustness
and measured online cost, rather than claiming that this benchmark establishes
new stellar-evolution campaign efficiency.

## Figure revision — 2026-09-09

The active manuscript now uses KL-versus-budget median/spread panels instead of
pass-count curves, a single normalized-width panel instead of crowded recovery
and Fisher overlays, and a single central-interval coverage plot instead of four
rank panels. Bands across ten catalogues are empirical 10th--90th percentiles,
not confidence intervals. See `FIGURE_DESIGN.md`. All current figures use four
visible spines and inward mirrored ticks, without titles; compact legends are
allowed. Inference records and validation thresholds are unchanged.

## Random acquisition, early initialization and model notation

`random_baseline.py` adds a supplementary independent-uniform acquisition arm:
all methods share the original 226-point initial design, then the random arm
uses 512 IID prior draws (seed 20260910). This is a matched comparison of added
points, not a wholly random initial design; the same draw sequence is used for
all 40 catalogues. References are reused only for independent validation.
`random_results/` preserves its own records and timing; `summarize_random.py`
requires completion before generating the manuscript result paragraph.

`posterior_evolution.py` now illustrates 32 and 128 points from the initial
space-filling design and the saved 482-point BO stage. The first two are new
diagnostic fits without boundary anchors or acquisitions, and do not alter
any of the original matched-budget results. Existing reference proposals have
full prior support; effective importance sample sizes are retained with the
plot reconstruction. These illustrations are not additional calibration tests.

`MODEL_NOTATION.md` maps the paper's Bayesian model to the code and identifies
the distinction between binned mock data and event posterior-sample recycling.
The two-panel posterior predictive figure overlays all three catalogues with
matching curve/point colours and a legend, without truncating the larger counts.

Random acquisition completed: final combined passes are 10/10, 8/10, 2/10 and 0/10 at expected counts 30, 100, 300 and 1000. All final numerical checks pass. This is one independent-uniform addition design after the common initialization. See random_results/summary.json.
