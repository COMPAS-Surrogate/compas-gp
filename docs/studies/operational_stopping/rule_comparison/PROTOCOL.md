# Comparison of operational stopping rules

This extension preserves all v1 studies and production defaults. The new v1
replication uses all five existing target families and fresh seeds 3000--3004,
with the unchanged 800-call budget, acquisition and tolerances. It writes each
decision before revealing the exact reference. No threshold search is planned.

Five rules are evaluated on prefixes of the same acquisition trajectories:

1. Best observed log likelihood improves by less than 0.01 over three updates.
2. Two consecutive batches pass the existing pre-refit prediction check.
3. Two consecutive updates pass posterior stability and numerical precision.
4. Two consecutive updates pass prediction, stability and numerical precision.
5. The existing v1 combined rule, including its fresh 64-posterior/16-prior audit.

The first four decisions use checkpoint records and training values only. All
prefix decisions are written before loading reference scores. Scores retain the
original KL, marginal-error and numerical criteria. A rule that has not fired
when a recorded trajectory ends is censored: this is not an independently
executed full-budget run of that alternative. Counts include all true labels
already paid for along the recorded path. These are comparisons of stopping
decisions under one acquisition policy, not comparisons of acquisition methods.
Seeds 2000--2004 are exploratory replay; 3000--3004 are a fresh replication.

Targeted improvement: replace the strict new-batch prediction requirement by
posterior stability followed by a fresh direct-likelihood audit. At the first
two-update stability trigger, draw 128 IID indices from a fresh GP-only Sobol
quadrature and 64 independent uniform-prior coordinates. Retain the original
audit tolerances. This **first-trigger screen** only accepts or rejects this
one candidate GP; a rejection is not a completed adaptive learning run. The
audited fit is fixed; all 192 new calls are counted. Selection uses no reference
scores or target family. Fits, coordinates and predictions are frozen before
true labels are requested; references score the decision afterwards. There is
no claim of a uniform mode-discovery guarantee or successful deployment.

An additional first-trigger screen applies those same audit sizes/tolerances
to the two saved COMPAS pilot trajectories, using fresh exact rate evaluations
and their original 2200-call guard. This follows inspection of their operational
stability replay and is exploratory. It does not generate new catalogues or
independently confirm a selected rule. Catalogue coefficients are available to
evaluation; saved reference rates and reference proposals remain scoring-only.

Separately, measure the detection rate of larger broad audits on all five v1
hidden-mode stopped GPs (seeds 2000--2004). Use 32 fresh audit seeds per GP, 64
GP-posterior draws, and nested uniform-prior counts 16, 64, 256 and 1024. These
are conditional audit experiments on five fixed failures, not 160 independent
BO runs. Reference posteriors do not select audit coordinates. Larger audit
budgets are diagnostic; they may exceed the original 800-call run budget.

For IID prior probes, a pre-specified discoverable region of prior probability
v is missed with probability (1-v)^m. Report m >= log(delta)/log(1-v). This
requires that hitting the region would reveal a problem, and says nothing
without a minimum detectable volume assumption. Do not apply the IID formula
to adaptively acquired training points or interpret a prior-volume bound as
a posterior-mass bound.

## Exploratory posterior-impact refinement

The first COMPAS audits passed their posterior residual checks but failed the
pointwise broad-region cutoff (0.53 and 0.66 lnL versus 0.3). This motivates an
additional exploratory score using the same paid audit labels. Let
q = (2/3) p_GP + (1/3) prior; estimate both GP and exact posteriors with common
importance weights on the 128+64 audit sample. The GP normalization uses the
independent GP-only quadrature. Require sample maximum-directed KL <0.005,
mean/interval changes <0.05 GP SD, 68% width change <3%, and both importance
ESS >=64. This assesses posterior impact rather than a largest pointwise tail
error. These finite-sample estimates are diagnostics, not confidence bounds.

This candidate is defined after seeing the two COMPAS residual diagnostics;
their impact results are development evidence. Apply it unchanged to the fresh
3000--3004 toy first-trigger audits, with no threshold search. First-trigger
rejection remains rejection, not a new trained solution. The broad-mode
discovery limitation remains; this refinement is not a coverage certificate.

Chart contract: a static Matplotlib horizontal stacked bar compares outcome
counts for five rules on each 25-case set. Blue indicates an accurate stop,
pink with hatching a premature stop, neutral open fill a censored trajectory;
explicit counts and labels provide a non-color distinction. A second dot/line
plot shows rejection fractions versus broad-audit cost for each fixed failure.
Export title-free PNG/PDF with explanatory captions, following this scientific
repository's figure convention. Inspect exported PNGs before handoff.
