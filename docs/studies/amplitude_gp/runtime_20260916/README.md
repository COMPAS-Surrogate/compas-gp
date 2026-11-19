# Four-input GP runtime investigation — 16 September 2026

## Measured findings

- CPU benchmark on OzSTAR `dave1`, one CPU, JAX 0.10.2, x64, using saved checkpoints from `count_n100_perfect_c200_s17_bo`.
- Explicit JAX synchronization includes execution time. First and repeated calls are recorded separately in `profile_results.json`.
- Hyperparameter fitting is the dominant late-run cost. The 20-step fit grows from 1.66 s at 504 labels to 62.91 s at 2994 labels.
- Extrapolating 62.91 s to the production 250 steps gives about 786 s (13.1 min), before acquisition and conditioning. This is an extrapolation from a shorter fit, consistent with the independently observed 806–934 s final checkpoint intervals; it is not a newly timed 250-step fit.
- The one-step control takes 3.92 s, versus 62.91 s for 20 steps at 2994 labels. A simple linear model gives about 3.10 s per optimizer step and 0.82 s fixed overhead, predicting 777 s for 250 steps. This supports numerical optimization, rather than compilation, as the dominant cost. The intercept is an estimate of all fixed overhead, not a direct compilation timer.

All times below are seconds, using the repeated call. Conditioning always uses the production capacity of 3008, even at 504 active labels.

| Active labels | 3 direct labels | Acquire 3 | Full conditioning | Experimental append 3 | Fit 20 steps |
|---|---:|---:|---:|---:|---:|
| 504 | 0.013 | 1.77 | 1.49 | 0.036 | 1.66 |
| 1494 | 0.013 | 1.96 | 1.45 | 0.034 | 10.13 |
| 2994 | 0.013 | 2.55 | 1.48 | 0.033 | 62.91 |

## Experimental update and its limits

- `profile_runtime.py` contains an opt-in Schur-complement inverse update. Existing campaign/package conditioning code is unchanged.
- A fixed-shape three-point update takes 0.033 s at 2994 labels versus 1.477 s for full conditioning (about 44 times faster **for this operation only**).
- At that checkpoint, maximum prediction/acquisition differences on 64 fixed random queries are 6.2e-9/7.4e-9. The 504- and 1494-label checks also pass.
- A small synthetic regression test checks ten consecutive updates, floor clipping, prediction/acquisition parity, and invalid inputs.
- This is not yet a long-trajectory numerical-stability guarantee. Rebuild on kernel/transform changes; test accumulated error and selected acquisition points before integration.
- Nine between-refit updates per 30-label interval would save roughly 13 seconds at the measured timings. This alone cannot fix a roughly 13-minute hyperparameter fit.

## Warm-start comparison (one saved catalogue)

- Start from checkpoint 2964 hyperparameters and fit the same 2994 labels. The transform reference is unchanged in this example; apply the target-scale ratio to the mean and variance when constructing the initialization.
- Keep the same Matérn-5/2 model, fixed noise, likelihood labels and final-checkpoint target transformation. Compare with the saved 250-step fresh-start Adam model.
- The independent query set has 128 prior-wide points plus 384 perturbations of the best training point (three scales). These are diagnostic queries, not posterior samples. Of these, 382 lie within 10 log-likelihood units of the best *queried* direct value.
- Centre prediction-minus-truth residuals by subtracting their mean within that subset, since constant likelihood offsets cancel from a normalized posterior. The RMSE below is not a posterior KL or an acceptance gate.

| Model | Refit seconds | Negative GP log marginal likelihood (lower is better) | Centred query-peak LnL RMSE |
|---|---:|---:|---:|
| Saved fresh-start Adam, 250 steps | Not re-timed; about 13 min inferred | 362549 | 0.3825 |
| Warm-start Adam, 20 steps | 64.26 | 303718 | 0.3447 |
| Warm-start L-BFGS-B, 20 iterations / 23 evaluations | 72.48 | 5294 | 0.1532 |

- L-BFGS-B **did not declare convergence**: it hit the 20-iteration limit. Fixed-step Adam likewise has no convergence certificate here.
- Better optimization is a substantial opportunity: the saved 250-step fit leaves a much better training objective reachable, and query-peak errors improve in this example.
- These are different fitted GPs. Maximum prediction changes across all queries are approximately 282 (Adam) and 2879 (L-BFGS-B) log-likelihood units; do not call the optimizer change numerically equivalent to the baseline.
- Neither candidate has passed the independent posterior gates. No claim of a 10x accepted-inference speedup follows from this one-checkpoint comparison.
- Next bounded validation: allow a convergence-based optimizer budget, repeat on representative low/high-count catalogues and both measurement views, check the independent posterior gates and acquisition behaviour, then compare complete warm-start trajectories against the frozen protocol.

## Priority for the next optimization

- Current fitting constructs a fresh GP with initial length scales at 0.3 of the prior width and runs 250 Adam steps every 30 labels. It does not initialize from the previous optimum.
- Compare warm initialization and convergence-based optimization at a saved final checkpoint. Keep the target transform explicit: when the scale changes, the GP mean and kernel variance must be rescaled consistently; reference/floor changes alter the target too.
- Optimizer changes alter the fitted model, unlike algebraically equivalent conditioning. Require independent posterior checks across multiple catalogues/seeds before changing the paper campaign.
- Do not reduce labels or drop tail anchors just to make matrix calculations cheaper. Determine required labels by matched checkpoint scoring.

## Queue handling and provenance

- Slurm rejected both whole-array and pending-range hold commands with `Requested operation not supported on this system`.
- Saved all 612 pending IDs in `paused_pending.txt`, then cancelled **only pending** entries 20–631 of array 16547555. Active entries 16–19 were left running; their frozen code, inputs and outputs are untouched.
- This is a cancellation for later resubmission, not a Slurm hold. `scontrol release` will not restart cancelled jobs.
- First benchmark job: 16578223. Warm-refit comparison job: 16578315; one-step compilation control: 16578318. Original warm benchmark 16578285 failed on a profiling-harness import (after fitting); its output is preserved. Control 16578307 was cancelled while pending because Slurm would not update its failed dependency.
- Remote profiler directory: `/fred/oz303/avajpeyi/studies/compas_gp4_runtime_20260916/`.
- Original benchmark source is retained remotely as `profile_runtime.py`; failed warm benchmark as `profile_refit.py`, and corrected warm benchmark as `profile_refit_v2.py`. JSON records include source/checkpoint hashes.
- Main campaign identities and accuracy classifications are unchanged. The incomplete 200-event perfect BO run still requires checkpoint recovery.
