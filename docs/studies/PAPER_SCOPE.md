# Mock-only GP posterior paper: scope and completion evidence

User scope decision, 2026-09-08: actual LVK observations are not required for this
paper. Demonstrate the surrogate under explicit simulated-data assumptions and
report its limits. Do not expand the project to resolve real-data systematics.

## Proposed central claim

GP-guided acquisition selects informative post-processing parameter evaluations.
A repaired, fixed-noise GP for the amplitude-marginalized COMPAS log likelihood
can reproduce direct parameter constraints on a fixed coarse chirp-mass/redshift
grid for the tested small simulated catalogues, using existing COMPAS populations.
The claim is about approximate inference under the specified model, not measured
astrophysical constraints. The old four-dimensional trained models fail this test.

## Evidence to present

1. Correctness: exact-rate contraction parity, amplitude integration, noise-freeze
   regression, sampler initialization and diagnostic checks. Explain the GP bug
   with a controlled same-training-data comparison.
2. Approximation: held-out likelihood and joint posterior agreement across three
   fresh training seeds; both 32M and 512M targets; no oracle training points,
   no prior truncation and all acquired tail points retained.
3. Population sensitivity: show the 32M/512M discrepancy, exact population overlap,
   disjoint halves and qualified block-bootstrap scatter. Do not present 32M and
   512M as independent simulations or bootstrap pooling as a Bayesian correction.
4. Calibration: independent noisy-catalogue ensemble, direct/GP interval coverage
   and posterior ranks, binomial uncertainty, all failures, numerical reference
   checks and the separate perfect-measurement assumption.
5. Grid and cost: show why mass10 was chosen, retain full mass support, and separate
   rate construction, fitting, acquisition and inference costs. Quantitative
   information-loss and end-to-end speedup claims require additional comparisons.

## Minimum remaining work after the ensemble

- Completed: all 100 refined reference checks pass; GP interval checks also
  pass against that reference. See `joint_pilot/CALIBRATION_100.md`.
- Assemble the figures, frozen configuration and provenance, including the fitted
  study path rather than assuming production CLI defaults are equivalent.
- Report costs against the accelerated direct baseline; conditional amplitude
  recovery still evaluates direct rates. GP usefulness is demonstrated by accuracy;
  a speed advantage over that baseline has not been established.
- User chose to freeze the grid and condition inference on that binned model.
  No further grid comparison is required; do not claim negligible information loss.
- Completed the paired BO/uncertainty KL study and matched space-filling
  baseline: `joint_pilot/ACQUISITION_COMPARISON.md`. No clear overall adaptive
  advantage in this small pilot; report the result without an efficiency claim.

The 100-catalogue cross-population ensemble is a calibration diagnostic under a
small model mismatch (disjoint finite populations). It is not exact-model SBC.
A matched generation/inference population control would isolate internal
calibration if the cross-population experiment shows a problem. A null result
with 100 catalogues cannot certify few-percent coverage accuracy.

## Explicitly deferred

Actual LVK parameter inference; epoch-dependent observing sensitivity and exposure;
event parameter-estimation/selection systematics; modelling high-mass systems
outside the assumed population support; astrophysical model discrepancy; larger
COMPAS simulations; adaptive binning/delayed acceptance; broad GP architecture
searches. The current model's simulated high-mass tail remains included.

Synthetic event measurement uncertainty was authorized on 2026-09-09 and was
tested with a controlled categorical post-detection response; see
[uncertain measurements](joint_pilot/UNCERTAIN_MEASUREMENTS.md). Posterior predictive
checks are included. Larger catalogues remain an optional separate regime test.
Do not imply that passing ideal small-catalogue tests validates these extensions.


## Final comparison scope (2026-09-09)

ANN implementation and head-to-head rate-emulator training are explicitly dropped.
Use Riley & Mandel only as literature context. Our reduced, catalogue-specific
likelihood and their reusable forward emulator have different goals and dimensions;
training locations alone do not establish efficiency or equal accuracy. Validate
our own label budget on fresh data, and report the tested grid and catalogue regime.

User emphasis, 2026-09-09: the intended methodological contribution is choosing
parameters to explore using GP-based BO/active learning. The present experiment
demonstrates this for post-processing an existing population. Choosing new
stellar-evolution COMPAS runs is a proposed application, not an executed result.
Keep training-label efficiency distinct from actual runtime savings against the
accelerated direct baseline. See [direct inference and selection audit](joint_pilot/DIRECT_REFERENCE_AND_SELECTION.md).

User clarification, 2026-09-10: precomputing cosmic-integration rates for reuse
is acceptable as a benchmark convenience. Present the method as sequential
selection of expensive evaluations, applicable when a precomputed parameter
grid is infeasible. Candidate coordinates need only cheap GP predictions;
unselected rate maps are unavailable to acquisition. Keep cache construction
and cost accounting in the validation details rather than making precomputation
part of the method's premise. Existing cached results remain usable.


## 2026-09-09: completed event-count BO comparison and manuscript assembly

See [event-count study](event_count_bo/README.md) for the frozen design and
reproduction commands. Forty catalogues (ten each at 30/100/300/1000 expected
events), matched BO/Sobol trajectories and five checkpoints are complete.
At 738 labels the strict posterior/KL/numerical gate passes 10/10, 10/10, 10/10,
9/10 for BO and 10/10, 8/10, 5/10, 0/10 for Sobol. All direct-reference numerical
checks pass after reference-only refinement. BO marginal constraints pass all
40 cases; the remaining combined failure is KL-only. One injection and one
training seed bound these claims. Existing populations are post-processed;
no new binary-evolution inputs are selected and no online speedup is inferred.

The nested Overleaf draft now contains methods, results, six title-free figures
with descriptions in captions, and numerical-reference details in an appendix.
The central question is BO allocation of population-model parameter evaluations.
Fast suite: 62 passed, 3 deselected, one existing Matplotlib warning. Next priorities
are injection/seed robustness and online cost accounting. LVK and ANN are outside
this paper's executed scope. No commits or remote synchronization were made.
