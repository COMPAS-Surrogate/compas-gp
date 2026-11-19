# GP-LnL posterior validation plan

Created: 2026-09-08. Latest: repaired reduced GP passes three-seed 32M/512M
agreement checks and a 20-catalogue pilot; see
[reduced validation](joint_pilot/REDUCED_GP_VALIDATION.md). A separate 100-catalogue
ensemble and refined numerical references now pass; see
[ensemble report](joint_pilot/CALIBRATION_100.md). The original four-dimensional GP remains rejected.

## Objective and scope

Demonstrate a useful approximate posterior for the four COMPAS post-processing
parameters, with errors small compared with the reported parameter constraints.
Start with a small observed catalogue, the 32M population, and a fixed, manageable
Mc-z grid. Establish the regime where this works before extending it.

The minimum result is agreement with a direct binned-likelihood posterior,
evidence that finite-population errors do not materially change those constraints,
and repeated noisy-mock validation. Claims are conditional on the population,
selection, observation, prior, and binning models tested. A direct likelihood
using a finite COMPAS file is a numerical reference, not astrophysical truth.

**Paper scope, clarified by the user on 2026-09-08:** simulated observations,
fixed selection model and existing COMPAS populations. Actual LVK inference is
not a paper requirement. Observing-epoch sensitivity/exposure, event measurement
systematics, selection calibration and the interpretation of high-mass systems
belong to future real-data work. Retain the full simulated mass support in the
current experiment; excluding LVK does not justify cutting its high-mass tail.
The paper must not claim constraints on the real Universe from these mock tests.

Deliberately exclude delayed-acceptance MCMC, adaptive pixel selection during
sampling, a new GP architecture, and a comprehensive hyperparameter sweep from
the minimum result. Coarse-to-fine training is one bounded follow-up experiment.
Larger catalogues and measurement uncertainty extend the claim only after they
pass their own checks. Failure in a difficult regime can define the scope of a
finished study rather than trigger unlimited method development.

## Questions and evidence

| Question | Required evidence | Hold fixed when comparing |
| --- | --- | --- |
| Is the COMPAS population large enough? | Stability across population sizes and independent realizations or qualified system-level resamples | Catalogue, grid, prior, likelihood implementation |
| Does the GP reproduce the likelihood? | Direct-reference agreement in marginal constraints and joint structure; held-out raw-lnL errors | COMPAS file, catalogue, grid, prior |
| Does sampling explore the GP posterior? | Reliable chain diagnostics, dispersed starts, checks for missed modes | Frozen trained GP and target |
| Is coarse binning useful? | Lower finite-population error and/or cheaper validated inference, with quantified information loss | Catalogue, physical domain, population and inference settings |

## Corrections to earlier evidence

- `subset_training/test_coverage_gate.py` centres each synthetic likelihood at
  the injected parameter; it does not simulate noisy observations. Its reported
  100% coverage cannot establish conservative or calibrated intervals. Even a
  too-narrow posterior centred exactly on the injection can pass. Treat it as a
  surface-approximation experiment, not SBC. This supersedes the coverage
  interpretation in the historical investigation notes in `AGENTS.md`.
- That experiment supplies points near the known peak and does not validate
  discovery by the complete active-learning procedure.
- Entry points differ: `SurrogateWorkflowConfig` in `workflow.py` defaults to
  square-root compression, while the older `LnLSurrogate` training factory call
  leaves compression at its default. Freeze one explicit configuration and test
  the same path that will produce the reported result.
- The reviewed NUTS `marginal` target assumes linear propagation of GP variance.
  This is not valid under square-root compression. Start with the deterministic
  inverse-transformed GP prediction (`target="mean"`), assessed against direct
  lnL. This target is a plug-in prediction, not generally the expectation of raw
  lnL. Do not claim uncertainty marginalization or use mean/marginal agreement as
  proof of accuracy. Repairing uncertainty propagation is optional unless needed.
- Keep all acquired far-tail training points, following the existing subsetting
  findings. Do not use earlier scoring tables without checking their definitions.

## Execution rules that limit cost

1. Benchmark before choosing run counts or promising completion dates. Separate
   file loading, integration, event likelihood, GP fitting/acquisition, compilation,
   and posterior sampling. Coarser output bins may not speed cosmic integration.
2. Cache binned rate matrices at evaluated parameter vectors where practical.
   Reuse them for catalogue likelihoods and exact nested aggregation; include file
   identity, parameters, grid, integration settings and code revision in cache keys.
   Repeated catalogue inference must still retrain without access to injected truth.
3. Use small pilots before expanding. The initial 32M case carries most validation;
   use 512M for targeted checks before committing to full 512M chains.
4. Change one source of approximation at a time. Freeze GP choices before the
   confirmatory catalogue ensemble; keep tuning and validation seeds separate.
5. Record failed fits, bad chains, excluded runs, direct-call counts and wall time.
   Never silently drop failures from a coverage denominator or promote a run merely
   because a sampler wrote samples.
6. Keep the scientific prior unchanged. Concentrate acquisition locally with
   broader exploration retained; do not truncate to a coarse credible box or use
   the coarse posterior as a prior for the same data. Final draws use the final
   resolution only.

## Stages and completion criteria

### S0 — Establish a trustworthy, timed baseline

- [x] Audit the current training and sampling paths; make explicit the kernel,
  transform, clipping, prior, acquisition schedule, seeds and sampler target.
  First joint pilot choices frozen in `grid_resolution/joint_pilot.toml`.
- [x] Check inverse-transform consistency and initial-point handling. Verify
  convergence statistics with a standard implementation of rank-normalized split
  R-hat and bulk/tail ESS. Physical-start conversion fixed; rank/folded split
  R-hat and bulk/tail ESS now use NumPyro estimators; exact-target sampling passes.
- [x] Replace the misleading coverage gate with a cheap generative Gaussian toy:
  draw parameters, simulate noisy data, compare with a known posterior. Include
  deliberate location shifts and interval shrinkage to test failure detection.
  1,000 genuinely noisy simulations; deliberate narrowing and bias detected.
- [x] Benchmark representative 32M evaluations and a small GP/sampler baseline.
  32M costs ~0.68 s/evaluation; synthetic baseline 2.64 s excluding imports.
  One joint seed's 356 calls cost ~4 minutes before GP overhead and direct-chain
  reference costs. Measure the latter before expanding the campaign.
- [x] Record 32M file provenance and physical settings in the grid manifest.
- [x] S3 completed the 512M memory/timing and overlap audit: 32M is a subset
  of 512M; the streamed loader and cached operators support disjoint-block tests.

**Exit:** basic correctness checks pass, actual configuration is reproducible,
and a measured budget for S1–S2 is recorded. Fix only blocking issues first.

### S1 — Choose a fixed grid using physics and data-free population probes

- [x] Define nested grids over identical Mc-z bounds (five tested).
  Freeze edges before confirmatory validation. Keep integration accuracy separate
  from observational pixel size; do not arbitrarily thin the integration grid.
- [x] Verify rate conservation, bin-edge handling and correlated system weights
  under redshift aggregation on small synthetic fixtures.
- [x] Verify explicit-grid count likelihoods and synthetic uncertain-event integral
  bookkeeping (unequal areas, PE prior, out-of-domain samples). Legacy observation
  loading and actual LVK use remain unvalidated. For uncertain events,
  derive the event integral with the proper densities, bin areas and prior
  correction; summing both weight arrays and multiplying is not generally valid.
- [x] Plot rates at eleven parameter settings spanning fiducial, one-at-a-time
  prior endpoints and two joint extremes, using 32M and no observed data.
- [x] Compare system-level pixel effective sizes and retained normalized model
  differences, keeping the underlying integration fixed.
- [x] Select a provisional grid using physics and the measured trade-off:
  `mass10` (~10% fractional mass widths, dz=0.1; 870 pixels). See
  [grid study](grid_resolution/README.md) for the evidence and limitations.
- [ ] Validate this choice in the later posterior and finite-population checks.

**Exit:** one fixed candidate grid, justified by physics and population probes.
The data-free screen is complete; it does not demonstrate adequate finite-population
accuracy. Posterior agreement in later stages is still required. Aggregation
leaves the total-rate Poisson term unchanged and cannot fix its sampling error.
Redshift contributions from the same binary are not independent draws.

### S2 — Demonstrate GP versus direct posterior agreement

- [x] Generate one fixed low-duration Poisson mock catalogue, initially with
  perfect measurements. Reuse the same events for candidate/native grid comparisons.
  Completed: 81 perfect binned events, 0.1 yr, seed 20260908; explicit grid arrays
  in `grid_resolution/results/pilot_catalogue.npz`. Conditional amplitude reference
  passes; this is not the joint reference required below.
- [x] On the fixed catalogue and S1 grid/population, obtain a converged direct-likelihood
  reference with broad/dispersed initialization. A mode match or 1D slice is not
  a reference posterior. Choose a sampler supported by the actual direct target;
  do not assume the COMPAS integrator is differentiable for NUTS.
  Completed: exact sparse contraction + analytic amplitude marginalization; four
  Metropolis chains, max R-hat 1.00142, min bulk ESS 3185. Independent 8192-point
  full-prior Sobol integration agrees in shape-parameter means within 0.027 SD.
- [x] Train the frozen GP configuration using three independent training seeds.
  Use four dispersed chains for the cheap GP target; inspect joint structure and
  conduct broader mode searches as well as chain diagnostics.
  Repaired reduced model: three fresh seeds pass on 32M and 512M using broad
  quadrature validation; separate four-chain 512M NUTS passes after reducing its
  step size. Historical four-dimensional failures remain in the joint report.
- [x] Make held-out direct evaluations in the bulk, interval boundaries,
  degeneracies and broader regions. Start with roughly 100 points if S0 timings
  permit; these are independent validation points, not training labels.
- [x] Compare means, interval endpoints/widths, correlations and joint contours.
  Subtract a constant offset when scoring raw-lnL residuals. Importance reweighting
  is a supplementary check with weight ESS/overlap diagnostics; it cannot prove
  absence of modes never visited by the proposal.

**Provisional tolerances:** location and interval-endpoint changes below 0.2
reference marginal standard deviations; 68%/95% widths within 10%; no material
joint-structure discrepancy. Aim for centred posterior-region lnL RMSE near
0.1 and inspect large residuals. These are proposed engineering criteria, not
theorems. Freeze them after the pilot and before confirmatory runs. Reference
Monte Carlo uncertainty must be appreciably smaller than the comparison tolerance.
Near-boundary/skewed cases also require quantile and joint-region assessment.

**Sampler gate:** rank-normalized split R-hat <1.01, adequate bulk and tail ESS
(initial floor 400 per parameter), no unresolved divergences or missed-mode
concerns. Passing diagnostics alone does not validate the target likelihood.

**Exit:** all three training seeds meet posterior tolerances. If not, permit one
focused acquisition/budget adjustment and repeat; record an unsupported regime
if the bounded attempt fails, rather than launching a kernel search.

### S3 — Establish finite-population stability

- [x] Compare 32M/512M direct posteriors on the same catalogue and selected grid.
  Completed, including the disjoint 480M remainder: sigma shifts ~0.7 SD and its
  68% interval widens ~26%. See [population study](population_stability/README.md).
- [x] Use disjoint blocks and block resampling to assess conditional realization scatter.
  Sixteen 32M blocks, two 256M halves, and 128 full-size block-bootstrap replicates
  completed; mean scatter 0.014–0.088 posterior SD. Individual-system resampling
  and independently verified simulation campaigns remain untested. Recompute
  correlated contributions and normalization consistently when resampling systems.
- [x] Escalate to direct/reference posterior comparisons when residuals or
  importance-weight overlap do not support a reliable constraint comparison.
- [ ] Quantify independent-realization uncertainty where data permit. Bootstrap
  checks are conditional on represented systems; they cannot reveal missing
  population support. If independent runs are unavailable, state that limitation.
- [ ] Verify interval and location stability against the S2 tolerances, including
  per-event Mc-z contributions rather than the total-rate diagnostic alone.
  Current result **fails**: sigma endpoint changes ~0.97 SD; total rate changes
  only ~0.1%, demonstrating why rate stability alone is insufficient.

**Exit:** finite-population uncertainty is below the claimed constraint accuracy,
or the result is explicitly restricted to the finite-file target. If necessary,
coarsen the grid or increase the population and revalidate affected stages.

### S4 — Calibrate the frozen procedure on noisy catalogues

- [x] Run a 20-catalogue pilot to catch gross failures and measure ensemble cost.
  All 20 completed; GP/direct widths within 1.7%, 91.5 s main run.
- [x] Run a fresh 100-catalogue frozen-procedure ensemble after the pilot passes.
  Completed with disjoint 256M halves: all 100 GP/direct interval checks pass;
  two 32768-point direct references agree on every coverage decision.
  Draw parameters from the stated inference prior and simulate random counts and
  events, retraining the full procedure for each dataset without oracle points.
- [x] Report posterior ranks and 68%/90%/95% coverage with binomial uncertainty,
  separately by parameter; do not count correlated parameters as independent
  simulations. Account for posterior draw autocorrelation in rank checks.
- [ ] Include representative difficult-region and direct-reference spot checks.
  Fixed-parameter repeated-data coverage is a separate diagnostic; it need not
  equal nominal Bayesian coverage at every parameter value.
- [ ] Test shared finite-population internal consistency separately from mocks
  generated using independent/larger populations. Describe which claim each tests.

**Exit:** calibrated toy controls and no unresolved material undercoverage or
posterior mismatch in the stated regime. At 100 simulations, nominal 68% coverage
has roughly 4.7 percentage points standard error, so this cannot certify few-percent
calibration. Report the uncertainty; increase towards 400–500 only if stronger
precision is needed. Non-rejection alone is not a calibration guarantee.

### S5 — One bounded coarse-to-fine efficiency experiment

- [ ] Compare direct fine-grid GP training with coarse-guided fine-grid training,
  using the same final grid, prior, catalogue and population, and three matched
  seeds. Use the validated manageable grid as the final target if appropriate.
- [ ] Warm-start acquisition with coarse information, but recompute final-resolution
  labels and retain broad exploration. Preserve far-tail observations at each level.
- [ ] Charge all coarse-stage work, loading and compilation to the total cost.
  Compare cost to reach the S2 posterior tolerances, not cost per iteration alone.

**Exit:** quantify any wall-time/direct-call saving without a loss of accuracy.
If there is no useful saving, retain fixed-grid inference and report that result.
Delayed-acceptance MCMC is not required to complete this stage or the project.

### S6 — Finish the scoped result; extend only when justified

- [ ] Assemble configuration/provenance manifest, direct-versus-GP corner plots,
  population/grid sensitivity summary, calibration plot and timing/call-count table.
- [ ] State supported catalogue size, grid, measurement model and failure boundaries.
- [ ] Optional: noisy event measurements and a larger catalogue with 512M. Start
  each with a small pilot and repeat applicable accuracy/calibration gates. Do not
  transfer the small-catalogue conclusion to the sharper regime automatically.

## Tracking and next action

Update this file after each stage with evidence paths, actual resource costs,
decisions and remaining limitations. Check boxes mean demonstrated completion,
not that a script exists. Changes to acceptance criteria must be dated and justified.

| Date | Stage | Evidence / decision | Next action |
| --- | --- | --- | --- |
| 2026-09-08 | Planning | Agreed direction recorded; experiments remain unrun | S0: audit configuration, repair cheap validation, measure costs |
| 2026-09-08 | S1 data-free screen | User prioritized physics-led grid selection. Five grids x eleven parameter probes; 10.54 s loading/integration/aggregation, excluding imports/plots. `mass10` chosen provisionally; no production defaults changed. See `grid_resolution/README.md`. | Complete S0 correctness work and event-grid consistency before S2; retain LVK selection/measurement checks |
| 2026-09-08 | Mass tail + event/grid pilot | 13 systems above 40; retain full support, no further coarsening. 81-event Poisson pilot and exact conditional-amplitude comparison pass. Synthetic event-integral tests pass; 24 fast tests passed. | Finish remaining S0 GP/sampler correctness checks, then joint direct/GP pilot; no joint validation claimed |
| 2026-09-08 | S0 correctness baseline | Fixed constrained starts and convergence diagnostics; reject nonlinear marginal correction; replaced old gate. Exact noisy controls, 4-chain NUTS and 1D GP checks pass; 35 fast tests pass. | S2: first joint 32M seed using frozen TOML; measure direct-reference cost; then compare constraints |
| 2026-09-08 | S2 first pilot rejected | Seed 17 and one local refinement both pass sampler checks but fail fresh direct-lnL checks (RMS 2.640/2.641 versus ~0.1). 456 total direct calls, 310.5 s measured work. No direct reference or additional seeds run. | Establish an efficient direct reference before more GP tuning; consider exact amplitude marginalization |
| 2026-09-08 | S2 direct reference and reduced GP probe | Exact contraction agrees within 4.6e-13 lnL at 23 probes; evaluations ~253x faster. Four direct chains and independent full-prior integration pass consistency checks (98 s total). Refined 4D GP 68% widths are 0.283/0.336 of direct widths for alpha/sigma. A bounded amplitude-marginalized 3D GP probe still has RMS 0.785 lnL at reference-posterior locations. | Use accelerated direct likelihood for S3 finite-population checks; defer further GP tuning until its cost advantage is demonstrated. |
| 2026-09-08 | S3 population comparison | 32M is a verified subset of 512M. Streamed loading and batched contraction enable 32M/512M/disjoint-480M posterior comparisons in 143.9 s. Sigma shifts ~0.7 SD and its 68% width grows ~26%; 40 fast tests pass. This fails the provisional population stability criterion. | Compare disjoint population blocks, starting with two 256M halves; estimate realization scatter before calibration or further GP tuning. |

| 2026-09-08 | S3 existing-budget bootstrap | Sixteen block operators reproduce 512M to 9.7e-16 relative. Half-population means differ from full by <0.063 SD. Full-size bootstrap mean scatter 0.014–0.088 SD; 68% widths vary up to ~7% over the central 90% of resamples. 80.9 s main run; 41 fast tests pass. | Proceed to a bounded noisy-catalogue calibration pilot using disjoint generation/inference subsets; preserve finite-population sensitivity as an error budget. No new COMPAS simulations. |

| 2026-09-08 | GP implementation diagnosis | optimise_noise=False did not freeze noise: actual SD ~5 versus configured 0.00316. Repaired and regression-tested. Same-point 250-step fit removes most artificial narrowing; +128 points gives means within 0.05 direct SD and widths within 6.2%. Fresh lnL RMS remains 0.58–0.66; 42 tests pass. | Reassess the reduced 3D target with the freeze repaired; keep fresh validation and seed checks before calibration. |

| 2026-09-08 | S2 repaired reduced GP passes tested accuracy checks | Exact amplitude marginalization, fixed noise and 98 boundary anchors; 738 retained points. Three independent training seeds on both 32M and 512M; independent quadrature confirms 512M RMS 0.049–0.055 lnL and 68% widths within 1.2%. See `joint_pilot/REDUCED_GP_VALIDATION.md`. | Freeze this protocol; retain direct likelihood as reference. |
| 2026-09-08 | Noisy-catalogue calibration pilot | 20 prior-drawn truths, disjoint 256M generation/inference halves, 0–125 events. All 20 GP/direct agreement gates pass; widths within 1.7%, means within 0.012 SD. Coverage containment identical, but only 20 trials; not precise calibration. 91.5 s main run. | Controlled event measurement/selection test, followed by a small LVK pilot only after direct agreement. |

No large simulation campaign, commit or push has been performed. S2 now has a
converged direct reference and a passing reduced GP for the tested small-catalogue
regime. The original four-dimensional GP results remain rejected. The reduced
model analytically marginalizes amplitude and uses direct conditional amplitude
reconstruction; no end-to-end GP speed advantage is claimed.

**Current direction (user, 2026-09-08):** freeze mass10 and proceed with BO on
simulated datasets. Further grid comparisons are optional and not on the paper's
critical path; condition the inference claim on the chosen binned model.

Completed the first paired BO/uncertainty study: three simulated catalogues,
three seeds, 226–738 points, both KL directions on two independent quadrature
grids. Mean-target median forward KL falls from 0.0493 to 0.00243 nats; exact
expected likelihood through the sqrt transform gives 0.0474 to 0.00238. This is
a small uncertainty effect, not a demonstrated calibration improvement. See
[BO uncertainty study](joint_pilot/BO_UNCERTAINTY.md).

**Completed acquisition baseline:** fixed Sobol prefixes were compared with the
same initial points, candidate pool, budgets, catalogues and seeds. All matching
audits pass. At 738 points mean-target median forward KL is 0.00243 for BO and
0.00234 for Sobol; BO wins 5/9 paired trajectories. Uncertainty propagation has
little effect on this conclusion. This pilot establishes no clear overall BO
advantage. See [acquisition comparison](joint_pilot/ACQUISITION_COMPARISON.md).

**Latest user-directed follow-up:** minimizing expensive training evaluations is
now explicit. A likelihood-variance acquisition rule passes the exploratory KL/
shape-constraint gate for all nine trajectories at 482 labels, versus 610 for
both earlier policies (21% fewer supplied labels at the tested checkpoints).
This is exploratory, not a wall-time saving or a validated online stop. See
[label efficiency](joint_pilot/LABEL_EFFICIENCY.md).

**Completed (2026-09-09):** fresh 100-catalogue confirmation of the frozen
482-label likelihood-variance rule: 100/100 posterior and numerical gates pass;
99/100 pass both KL directions. One borderline KL failure remains reported.
An alpha 68% coverage deficit shared with the direct likelihood persists in a
matched-population control on those 100 catalogues, but not in an independent
1000-catalogue direct-only control (68% coverage 67.1–68.9% across parameters).
This larger control does not independently calibrate the GP or population mismatch.
See [confirmation](joint_pilot/CONFIRM_482.md) and
[direct control](joint_pilot/DIRECT_RANK_CONTROL.md).

**Next unit:** assemble the mock-only paper evidence, retaining the original
coverage deficit and separate population-sensitivity limitations. A 482-label
budget is supported in the tested regime, not as a universal stopping criterion.

**Scope extension and completion (2026-09-09):** user authorized uncertain
measurements and posterior predictive checks, emphasizing GP-guided parameter
exploration. Completed a three-case paired pilot and 100 fresh matched-population
catalogues with a fixed correlated categorical measurement response. All 100
posterior/KL/numerical gates pass; 68% widths within 1%, 95% within 1.3% of direct.
P–P plots and first-three-case predictive checks are retained. Individual sigma
coverage deficits shared largely with direct motivate one independent 1000-case
direct-only control; no GP retuning or seed search. See
[uncertain measurements](joint_pilot/UNCERTAIN_MEASUREMENTS.md).

The uncertain direct-only 1000-case control is complete: all 68/90/95% coverage
levels are compatible with nominal within individual exact 95% binomial
intervals; original sigma deficits did not persist. Retain the original 100
GP cases separately. 60 fast tests pass. Next, measure online cost with the
frozen method against the accelerated direct baseline, then assemble the paper's
parameter-exploration, accuracy, P–P and posterior predictive figures.

**ANN implementation is dropped entirely.** Riley & Mandel will be discussed as
literature context for training designs, not as a matched efficiency benchmark.
Do not claim fewer irreducible expensive evaluations or similar accuracy from
training-row counts alone. No ANN weights, architecture reconstruction or rate-
emulator training is required. The grid, LVK exclusion and COMPAS budget remain
fixed. A 1-year simulated case is a separate regime test if the paper includes it.

## Latest completion record

2026-09-08: user excluded LVK inference from the paper. Completed 100 fresh
cross-population mock fits in 346.9 s and refined reference checks in 197.8 s.
All GP interval and reference numerical gates pass. Original smaller quadrature
flagged 22/100 cases; these are preserved, with the resolved refinement documented.
45 fast tests pass. No commit, push or additional COMPAS simulation.

## Budget constraint and uncertainty direction (2026-09-08)

Treat the existing 512M population as the available simulation budget. Larger
COMPAS runs are not a prerequisite for continuing this project. Failure of the
32M/512M stability comparison demonstrates population dependence for one mock;
it does not establish that the existing 512M run is unusable or measure its own
remaining error.

Completed the first conditional test using sixteen disjoint blocks and a
full-size block bootstrap. An individual-system bootstrap remains optional; it
is not the method used for the recorded results. Respect the simulation sampling design and normalization. Apply the same
resampling weight to every redshift contribution from a binary and across every
parameter evaluation; assess likelihood differences, preserving their covariance.
Subpopulation scatter is not directly the full-512M uncertainty.

If full-population uncertainty is material, test a positive latent rate-map model
with simulation-informed uncertainty, propagating that uncertainty into the event
likelihood. A bootstrap posterior ensemble is initially a sensitivity diagnostic,
not automatically a Bayesian marginal posterior or proof of coverage. Validate
any correction using disjoint mock-generation/inference populations and noisy
catalogues. Bootstrap cannot supply astrophysical channels absent from the run.

Keep this separate from GP interpolation error. Increasing GP training noise or
adding pointwise variance to log likelihood is not an accepted correction. In
particular, Essick & Farr (2022), https://arxiv.org/abs/2204.00461, show why ignoring
cross-parameter correlations in approximate Monte Carlo uncertainty marginalization
can introduce bias. Continue with the fast direct target for this diagnostic;
return to a GP only with a separately validated mean and uncertainty model.


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
