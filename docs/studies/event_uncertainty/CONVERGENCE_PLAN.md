# Posterior-convergence and 100-catalogue expansion — 2026-09-10

User-directed follow-up: explain candidate coordinates, evaluate training rates
only at selected points, extend BO/random beyond the old 738-label checkpoint,
and increase the catalogue repetitions to 100 per regime. This is post-processing
of the existing COMPAS population, not 100 new stellar-evolution campaigns.

## Scientific stopping rule for this benchmark

At every additional batch of 128 labels, require:

- GP/direct agreement on two independent full-support references: both KL
  directions <0.01 nats, means/endpoints within 0.2 direct SD, 68%/95% widths
  within 10%. These are the original accuracy tolerances.
- Independent numerical reference/GP integrations agree on means/endpoints
  within 0.05 SD and 68% widths within 3%.
- The GP posterior changes little from the preceding checkpoint: both KL
  directions <0.001 nats, means/endpoints within 0.05 SD, 68% widths within 3%.
- All three requirements hold at two consecutive checkpoints. Any unsuccessful
  or numerically unresolved checkpoint resets the consecutive-check count.

The reference agreement prevents us from accepting a stable but wrong posterior.
It also makes this a **reference-assisted benchmark stopping rule**, not a
validated operational stopping algorithm for a problem with no direct reference.
Reference costs are recorded separately and excluded only from clearly labelled
training/acquisition cost comparisons; they remain in total wall time.

738 is an ordinary checkpoint, not termination. A resource guard at 4322 training
points bounds exact-GP memory/time. Hitting it is reported as `resource_limit`,
not convergence, and yields a lower bound on the required label count. No cases
are omitted or turned into successes. If many cases reach the guard, extending
it or changing the model is a separate recorded decision. The primary eventual
table should give median and 10th--90th percentile evaluation requirements,
completion fraction/denominator, and censored counts; keep maxima as a secondary
summary rather than hinging the entire comparison on one difficult catalogue.

## Frozen ensemble

- First rerun original case zero at 100 and 1000 expected perfect detections and
  500 expected continuous uncertain detections. This extension pilot is separate
  from the fresh ensemble, with the same rule, priors, kernel and fit settings.
- Generate 100 fresh catalogues at each of 30/100/300/1000 expected detections
  with perfect measurements, plus 100 each at 50/500 with continuous uncertain
  measurements. Total: 600 fresh catalogues, 1200 BO/random trajectories. Save
  all catalogues before examining acquisition outcomes.
- Truth stays (-0.3,0.35,0.01,4.7), generation/inference use the same 256M operator,
  and catalogue SeedSequence is [202609120, expected events, case]. Counts are
  Poisson. These estimate fixed-injection performance, not prior-predictive SBC.
- Same 128 Sobol + 98 boundary initialization, seed17 and 250-step GP fit.
  Acquisition pool: 4096 Sobol coordinates, seed117, never pre-evaluated. The GP
  scores them and cosmic integration runs only for selected training points.
  Random additions use seed20260910 for perfect and seed20260922 for uncertain
  measurements. Both retain all training points. Initial rate calls count.
- The direct references use the same mixture and weights as earlier studies,
  starting with 65536 generated points/scramble. Refine to powers17/18 only for
  failed numerical checks, before scoring the GP. Preserve each attempt. An
  unresolved final reference is recorded separately from acquisition performance.
- The implementation computes the exact GP mean with one factorization for
  reference scoring, avoiding unused variances; regression-test against the
  original GPJax prediction. GP fitting and candidate variance are unchanged.

## Execution

`run_convergence_campaign.py` creates `convergence_20260910/`, generates all
600 fresh catalogues, runs the extension pilot, and then the frozen ensemble
serially. It refuses existing campaign directories and checks source hashes
between stages. A separate smoke run exercises actual cosmic-integration calls
on one original low-count catalogue. Use one BLAS/OpenMP thread and persistent
logs/status files. This may take substantially longer than the old fixed-budget
study, particularly for large random designs; elapsed costs are measured.

No claim of 100 completed fits, 100% coverage, a validated operational stop, or
new runtime saving is made merely because the catalogues exist or the campaign
was launched. Numerical criteria and all outcomes remain reviewable.
