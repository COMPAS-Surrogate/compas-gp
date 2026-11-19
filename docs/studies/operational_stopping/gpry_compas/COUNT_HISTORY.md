# Paired count and stopping histories — 12 September 2026

The requested expected counts are 10, 50, 100, 200, 500, 700 and 1000 for
both perfect and continuous uncertain measurements. Five paired catalogues per
count give 35 independent latent catalogues and 70 BO trajectories. These are
fixed-fiducial-injection repetitions, not prior-drawn calibration experiments.

Each iteration visits all seven counts and both models before moving to the
next catalogue seed (case IDs 200–204). Measurement pairs have identical
Poisson counts, latent events, exposure and truth. The existing 256M fixed
population, prior, selection and log-coordinate measurement models are retained.
The historical campaign and manuscript figures are preserved.

## Acquisition and stopping

Use the existing small-batch continuous BO: 58 initial labels, three acquisitions
per batch, refitting and posterior comparisons every 30 labels. All checkpoints
through 1258 labels are saved (41 posterior evaluations, 40 comparisons).
The far-tail anchors are retained and prior bounds never change.

Replay the fixed 96-point scheduled audit with two numerically resolved joint
stability passes using only saved training-time GP states and fresh exact audit
labels. Freeze the first accepted stop before accessing a reference posterior.
Training continues to the fixed endpoint regardless of this shadow decision.
An absent stop is a resource limit. The stored stop distinguishes training cost,
audit cost and total cost from the diagnostic continuation. Legacy monitoring
audits in the reused training driver are separately recorded and are not added
to the fixed96 policy cost. Audit replay wall time is not online stop latency.

## Reference and history

The distant reference is the actual continued 1258-label GP for the same
catalogue. It must pass the existing four-parameter accuracy and numerical
checks, plus maximum directed KL < 0.001 against the direct numerical target,
before it is labelled an accepted reference. A failed reference remains explicit.
The distance from every checkpoint to that endpoint is stored; its self-KL at
the final checkpoint is zero by construction and is not convergence evidence.
If the stop is near the endpoint, its reference is not meaningfully distant;
the saved distance makes that visible for a later extension.

Reuse existing direct-rate designs from the 100/1000-count study as full-support
importance proposals. The old count determines only proposal geometry. Current
catalogue coefficients and exposure determine every direct likelihood. Check
both independent scrambles for each new case; if the first proposal fails, try
the other, retaining both checks. If neither resolves the case, report unresolved.
Do not interpolate likelihoods or give numerical-reference labels to BO.
Selected continuous BO/audit coordinates use the existing fast exact population
operator; reuse of cached reference rates is offline acceleration, not an online
runtime claim.

For every checkpoint, retain the GP parameters and full training data, both
normalized posterior-weight arrays on the hash-identified reference coordinates,
four-parameter summaries, numerical checks, successive-posterior diagnostics,
both directed KLs against the direct target and late GP, and their symmetric
average and maximum. Shared exact conditional amplitude inference makes the
shape KL equal the four-parameter joint KL. The fixed support weights are
quadrature representations, not independent MCMC draws.

Checkpoint metadata distinguishes posterior evaluation index (starts at 1),
posterior comparison count (starts at 0), BO batches, training labels, cumulative
timings and timestamp. This supports either comparison-count or label-count
axes without conflating them.

## Run and inspect

```sh
MPLCONFIGDIR=/tmp/compas-count-history-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python docs/studies/operational_stopping/gpry_compas/run_count_history.py \
  --root docs/studies/operational_stopping/gpry_compas/count_history_20260912
```

The supervisor runs one process per job to bound memory. Every worker checks
source hashes against the saved snapshot. Execution errors stop the supervisor;
scientific failures remain outcomes and do not silently disappear. Resume skips
completed jobs, but refuses an interrupted partial trajectory to preserve its
evidence. The supervisor checks for at least 5 GiB free before each job.

Progress: `campaign_status.json`, `job_status.json`, job logs and
`training/status.json`. Results: `stops/<case>.json`, `offline/<case>/history.json`,
`posterior_<labels>.npz`, `late_reference.json`, `outcome.json`, and root
`outcomes.json`. The source snapshot and reference hashes support reconstruction.
Figure updates follow review of the completed histories; this launch is not a
new scientific result or an automatic manuscript replacement.
