# Four-input manuscript campaigns

This directory owns the reproducible submission workflow. It reuses the frozen
September 15 likelihood/input bundle and September 16 warm-fit policy. It does
not overwrite either campaign, retrain completed compatible runs, or install
packages on compute nodes.


## Stage two: complete the common campaign

Use a new root, for example
`/fred/oz303/avajpeyi/studies/compas_gp4_manuscript_stage2_20260917`.
Copy `campaign.py`, `diagnostics.py`, `run.slurm`, `submit.sh`, this README and
the unchanged `warm_validation.py` / `score_checkpoint_history.py` into it.
Run the prepare command below with the same original and parent paths, adding:

```bash
--previous /fred/oz303/avajpeyi/studies/compas_gp4_manuscript_stage1_20260917
```

The versioned manifest verifies completion and checkpoint hashes of prior runs,
then selects only the remaining **40 primary runs and four predictive examples**.
Budgets are 1884 except for the two 30-event predictive examples (1284).
No training settings or acceptance thresholds change. No additional fit is
launched just to reduce the known KL=0.0105 near miss.

`submit.sh` discovers the training task count from the manifest. It submits a
fresh 174-to-204 resume/prediction/acquisition pilot, a dependent 44-task
training array (four concurrent), and a dependent 12-task diagnostic array
(two concurrent). Thus at most six science tasks run simultaneously. The
12 diagnostics use the already completed catalogue-200 BO fits for expected
counts 10/50/100/200/500/1000 and both measurement views.

- Conditional amplitude slices at the injection and direct-reference posterior
  mean shape are compared with the exact bounded Gamma. The implemented full
  likelihood is checked independently against the Gamma slice shape.
- Actual observed counts determine Gamma shape; exact rates and exposure
  determine its rate. Prior truncation and shape-marginalisation are explicit.
- At 100/1000 events, four selected GPs additionally receive two held-out
  four-dimensional designs: 2048 shape coordinates from each existing reference
  design, 16 balanced amplitude samples per shape. Holdouts are checked for
  training overlap. Save exact/GP values, weights, transformed interval coverage,
  effective sample size and maximum weight. Designs are subsamples of existing
  withheld references, not a new independent calibration ensemble.
- Diagnostics never update the GPs. `diagnostics/<name>/status.json` records
  computational completion, not a claim of calibrated predictive uncertainty.
- Conditional grid integration is checked against the analytic Gamma CDF;
  unresolved integration fails visibly. Holdout between-design scatter must be
  assessed before manuscript use; importance ESS alone is not that check.

All checkpoint arrays and raw diagnostic curves remain available for plotting.
The completed predictive fits are propagated using joint GP importance weights
by `manuscript_figures.py`. Failed random
fits remain budget-censored rather than being silently omitted or extended.

## Freeze and submit (stage-one example)

Copy this directory plus `docs/studies/amplitude_gp/warm_validation.py` and
`score_checkpoint_history.py` into a **new** OzSTAR output directory. Verify the
copied files against local SHA-256 hashes. Then run on OzSTAR:

```bash
export COMPAS_CAMPAIGN=/fred/oz303/avajpeyi/studies/compas_gp4_manuscript_stage1_20260917
export COMPAS_REPO=/fred/oz303/avajpeyi/studies/compas_gp4_paper_20260915/code
export COMPAS_ENV=/fred/oz303/avajpeyi/codes/cosmic_integration_gp4_20260914/.venv
python3 "$COMPAS_CAMPAIGN/campaign.py" prepare --root "$COMPAS_CAMPAIGN" \
  --original /fred/oz303/avajpeyi/studies/compas_gp4_paper_20260915 \
  --parent /fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916
bash "$COMPAS_CAMPAIGN/submit.sh"
```

`prepare` freezes the job matrix, source hashes, rate-input hashes and parent
checkpoint/timing hashes. The source and input paths must continue to exist.
The old 1284-label run identity remains intact; children get new identities and
copy the parent's checkpoints and cumulative timing. Parent completion markers
are never copied. Partial child runs resume from their last saved checkpoint.

`submit.sh` creates a submission lock, submits one continuation pilot, then a
92-element array with an `afterok` dependency and four concurrent tasks. It
records both Slurm IDs immediately. If submission is interrupted, inspect the
ID files and Slurm before removing the lock or retrying; never blindly resubmit.
The pilot checks padded prediction parity, deterministic acquisition replay,
continuation from 1284 to 1314 and idempotent completion. Pilot failure prevents
production execution. Each task requests one CPU, 16 GB and three hours.

## Scientific scope

- Tasks 0–15: all eight existing 500-event BO/random pairs (including the seed-29
  sensitivity cases), continued to 1884 labels.
- Tasks 16–23: new 1000-event catalogue-200/201 pairs, both views and methods,
  trained to 1884. The fixed 1884 ceiling permits a clean matched comparison;
  checkpoints at 1284 are also scored.
- Tasks 24–91: 68 missing 10/50/100/200-event trajectories, trained to 1284.
- The remaining high-count matrix and four predictive examples are deferred
  until this stage is assessed. No automatic extension to 2484 or 2994.
- Acquisition never reads reference designs. The initial design, priors, kernel,
  transform, fixed noise, three-label batches and warm refits are unchanged.
- Save every 30 labels. Score 504/534, 744/774, 1254/1284 and, for longer runs,
  1314, 1554/1584 and 1854/1884. Report first **tested** passing checkpoint and
  its confirmation cost, not an exact minimum or operational stopping rule.
- Use the full joint-posterior and numerical-precision gates. Keep inaccurate,
  unresolved and timed-out runs visible. A completed Slurm job is not acceptance.
- Timing inherited from a parent is cumulative training cost; child Slurm time
  includes only this continuation and its offline scoring. Do not add cumulative
  parent and child training times together.

## Inspect and recover

```bash
squeue -j "$(cat "$COMPAS_CAMPAIGN/array_job_id.txt")"
sacct -j "$(cat "$COMPAS_CAMPAIGN/array_job_id.txt")" -X \
  --format=JobID,State,ExitCode,Elapsed
```

Results live in `production/runs/<name>/`: `complete.json` records training,
`history.json` and `status.json` record scientific scoring. For a failed task,
first establish its old allocation has ended, then submit only that array index
with the same frozen files/environment, `COMPAS_STAGE=run`, and `run.slurm`.
No result files need deletion for normal checkpoint recovery. Retain the old
Slurm record and the resubmission ID.

## Reproduce the completed manuscript figures

From the repository root, with the downloaded input and result bundles:

```bash
MPLCONFIGDIR=/tmp/compas-mpl JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/ozstar/gp4/manuscript_figures.py all
```

Modes `summaries`, `corners` and `predictive` can also run separately. The script
uses frozen final checkpoints without refitting. The four corner checkpoints
from stage one are stored under the stage-two `paper_figures/corner_checkpoints/`
directory. Six earlier 50-event BO histories are under `earlier_histories/`.
Predictive and corner caches retain joint GP weights or their intensity draws;
remove the corresponding cache before changing reconstruction settings.
Source hashes and numerical checks are saved alongside the caches. Figure 2
and the conditional-amplitude appendix use `plot_diagnostics.py`.

The primary acquisition figure retains every censored case. Width and corner
figures use final states rather than selecting only strict passes. Predictive
summaries retain the 1200-event perfect-view KL near miss (0.012632).

### Initial versus final GP figures

`manuscript_figures.py corners` now uses `paper_figures/early_checkpoints/`
plus the four final `corner_checkpoints/` files. It fits illustrative 25/50-label
prefixes using the frozen campaign fitting routine, without changing production.
It caches predictions on two uniform four-dimensional Sobol designs under
`paper_figures/gp_only_evolution/`. Its KL curves compare earlier and final BO
GPs, omit the final self-comparison and never load exact likelihoods or a direct
posterior. These curves measure endpoint-relative stability, not correctness.
Figure 4 uses all final widths, median/error bars, and actual full-run budgets;
reference-free stopping counts are not yet measured. Figure 3 retains the old
external-reference benchmark pending the broader manuscript revision.
