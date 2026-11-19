# Paired count campaign — 12 September 2026

User-approved grid: E[Nobs] = 30, 100, 300, 900, 1200 for both perfect and
continuous uncertain measurements. The Poisson realization, latent events,
exposure and injection match exactly within each pair. A fixed 256M COMPAS
population supplies both generation and inference, with the existing priors,
O3 selection approximation and independent log-coordinate sensor unchanged.

## Design and gates

- One case at each of five exposures gives ten paired likelihood cases and
  twenty BO/random trajectories in the pilot. All BO cases must pass the existing
  reference accuracy and numerical precision checks before the ensemble starts.
  Random resource limits are retained; numerical failures block expansion.
- After the gate, run 100 realizations per count: 500 independent latent
  catalogues, 1000 likelihood cases and 2000 trajectories. The pilot's case zero
  overlaps the ensemble's case zero and is never pooled as independent evidence.
- Catalogue seed remains [202609120, expected count, case]. This preserves the
  earlier latent 30/100/300 catalogues. BO retains seed 17, 226 initial evaluations,
  128-evaluation batches, 4096 Sobol candidates and 10% uniform exploration.
  Random coordinates are IID uniform with seed 20260910 for BOTH measurement
  models (the historical uncertain experiment used 20260922).
- Termination requires two consecutive accurate, numerically resolved, stable
  checkpoints. The 4322-evaluation guard remains a resource limit, not success.
  This is reference-assisted acquisition benchmarking, not the operational
  small-batch stopping-rule experiment; current stopping figures retain their
  actual 100/1000 expected counts until separately rerun.
- Pilot and ensemble share only immutable numerical-reference rate arrays;
  acquisition never receives those values. Training calls remain selected-only.
  Source hashes, all checkpoints, pairing checks and costs are retained.
- References leave at least 5 GiB disk free. Execution errors or a failed pilot
  gate stop the campaign and leave a failure record instead of silently expanding.

## Figures and interpretation

The pilot writes a 30-expected-event data illustration immediately after data
generation. It shows exact parameters versus displaced measurement centres,
with selected likelihood contours in log coordinates; these are not event
posterior credible regions. No observations are clipped to population support.

After validated pilot fits, `paired_diagnostics.py` can build paired perfect /
uncertain posterior predictive checks at 30 and 1200 expected detections.
These draw shape parameters from the validated GP posterior and amplitude from
its exact bounded-Gamma conditional, then generate Poisson counts, latent
parameters and noisy measurements. Curves are medians of replicated empirical
CDFs with pointwise 90% bands. Count distributions are shown separately.
They are not curves at the vector of median hyperparameters or simultaneous
acceptance bands. Predictive refitting/reconstruction is additional diagnostic cost.

## Run

From the repository root, with one BLAS/OpenMP thread:

```sh
MPLCONFIGDIR=/tmp/compas-paired-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python docs/studies/event_uncertainty/paired_campaign.py \
  --root docs/studies/event_uncertainty/paired_20260912
```

The root must not exist. `campaign_status.json`, `pilot.log`, `ensemble.log`,
and each phase's `status.json`, `records.json`, `outcomes.json` expose progress.
Existing manuscript results remain historical until the replacement analysis
is complete and its failed/censored outcomes have been reviewed.
