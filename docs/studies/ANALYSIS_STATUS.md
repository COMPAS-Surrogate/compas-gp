# Mock-only surrogate analysis status — 2026-09-11

## Dense reference and physics revision — 2026-09-12

Current PDF: `overleaf/output/dense_reference_review/ms.pdf`. The four corners
compare neutral BO curves with actual, internally validated 2000-label GP
references. Stopping panels are monochrome. Physical posterior widths versus
expected event count now share the existing evaluation-cost figure, retaining
three main figures and five overall. Figure 5 now shows expected likelihood
loss for measured 32M/128M/256M populations relative to 512M, with 12 count
locations and an explicit distinction between exact exposure scaling and
independent runs. All 149 fast tests pass. See
`docs/studies/paper_completion/DENSE_REFERENCE_AND_PHYSICS.md` for construction,
results and limitations. Entries below describe earlier revisions.

## Corner-plot redesign — 2026-09-11

The current PDF is `overleaf/output/corner_review/ms.pdf` (18 pages). Four BO-only
corner plots now compare smaller/larger catalogues with perfect/uncertain
measurements, grouped into the same two figures. Aligned stopping panels use one
audit score, a unit threshold, stability markers and the recorded stop. Numerical
reference comparisons remain internal, with a concise reference description in
the validation appendix. See `docs/studies/paper_completion/CORNER_REDESIGN.md`
for provenance and checks. Earlier entries below describe preceding revisions.

## Presentation revision — 2026-09-11

The current manuscript uses three main figures: paired perfect/uncertain corner
plots with actual BO stopping diagnostics, followed by an ensemble evaluation
summary. Five figures remain overall, with cosmic integration and population-size
sensitivity in the appendix. The main likelihood and mock-data generation remain
in the main text. The review PDF is `overleaf/output/presentation_review/ms.pdf`
(18 pages including appendices). See `docs/studies/paper_completion/PRESENTATION.md`
for the narrative, source data and reproducibility checks (paths relative to the
package repository). Earlier entries below describe preceding revisions.

## Completed manuscript campaign — 2026-09-11

The fresh confirmation and population-size runs are complete. BO has 23 accurate
stops and one premature stop in the fresh 24-case confirmation; random has
three accurate stops and 21 non-stops. This supersedes any inference of a
zero-failure guarantee from the development replay below. The 600-catalogue
benchmark has 1,197 successful outcomes and three retained random resource
limits after independent reference refinement. All 36 population-size posterior
checks resolve; the new main figure measures parameter sensitivity to COMPAS
size versus observed catalogue size. Full results, provenance and figure changes:
`docs/studies/paper_completion/RESULTS.md` (relative to the package repository).
The fast test suite passes 142 tests. Earlier entries below are historical.

## Periodic audit repair — completed development replay, 2026-09-11

Periodic audits independent of the prediction counter, with two resolved
joint-posterior stability passes required for acceptance, give 24/24 accurate
BO stops on the already-examined extension cases. Both fixed-96 and adaptive
96/192/384 audits have zero premature, unresolved or non-stops. Their median
training counts are both 358; median total costs are 646 and 856, respectively.
All three original BO non-stops are resolved. The fixed-96 policy is preferred
for this benchmark because adaptive expansion costs at least as much in every
case. This is development evidence, not new-catalogue confirmation.

The preceding audit-only candidate is preserved: fixed/adaptive audits had
one/two premature BO acceptances. Those checkpoints failed joint-posterior
stability. Both revisions were replayed on all 48 BO/random trajectories with
decisions frozen before new reference scoring, and all four physical marginal
posteriors checked at accepted stops. All 24 random trajectories now receive
audits but none accepts within 658 training labels. Five final random fits
meet the looser independent accuracy target and still fail the tighter audit
criteria; non-stopping must not be equated with posterior inaccuracy.

Results, exact costs, plots and source provenance:
`operational_stopping/gpry_compas/stable_scheduled_v2/report/RESULTS.md`.
The rejected audit-only candidate remains under `scheduled_v1/`.
No new parameter-space reference grid or GPry installation was needed; audits
query selected coordinates through the fast fixed-population rate evaluator.
All original training trajectories and stopping decisions are preserved.
138 fast tests pass, three heavy/network tests are deselected. Manuscript
methods and the new development subsection report both candidates explicitly.

## COMPAS stopping and matched random comparison — 2026-09-11

The new COMPAS-only extension is complete: 48 trajectories, three injections,
100/1000 expected events, and paired perfect/uncertain views of 12 independent
latent catalogues. All decisions were frozen before direct-reference scoring.
At 658 training labels, BO passes all accuracy checks in 24/24 likelihood cases,
versus 5/24 for IID random continuation after identical initialization. BO has
lower final KL in all 24 pairs; the median paired random/BO KL ratio is 108.
These are training-budget comparisons, not runtime speedups.

The posterior-mass audit uses the first injection for development and the other
two for confirmation. On the 16 confirmation cases per method, BO gives 13
accurate stops, zero premature stops and three non-stops; the pointwise audit
gives 10 accurate stops and six non-stops. The mass rule's median accepted cost
is 388 training plus 96 audit calls (484 total). These conditional medians do
not compare identical accepted subsets. The refined counter without guards
has one premature BO stop in confirmation. Random reaches no guarded audits;
its non-stops also reflect the shared counter and stability requirements.

All final numerical checks pass, including exact reconstruction of all four
physical marginal posteriors. The GP has three inputs because amplitude is
analytically marginalized. Acquisition and stopping already evaluate rates on
demand without a parameter-space grid. Removing validation caches also requires
counting the total-rate calls used to reconstruct amplitude at posterior points.
The finite study does not establish universal stopping, a four-input GP,
continuous-sensor calibration or new COMPAS evolution results.

The reproducible report is `docs/studies/operational_stopping/gpry_compas/report_v2/RESULTS.md`
from the repository root. `extension_v2/verification.json` replays all 1008
checkpoints, pairing and audits. Numerical reference refinements, the scoring
path fix and its resumed outputs are preserved under `extension_v2/offline/`.
The fast suite passes 129 tests, with three heavy/network tests deselected.

The manuscript adds `sections/bo_stopping_methods.tex`, the generated
`output/bo_study/stopping_results.tex` and
`figures/bo_study/bo_random_stopping_convergence.pdf`. The compiled review copy
is `overleaf/output/stopping_extension_draft/ms.pdf`. No commits or remote Overleaf
synchronization were performed. Older entries below describe separate studies.

## Latest user-directed revision — 2026-09-10

The reference-free stopping candidate has now been implemented and tested with
thresholds frozen before confirmation. It is **not ready for general use**.
Across 25 toy runs (five families, five seeds each), it produced 11 accurate
stops, 5 premature stops and 9 resource-limited runs without a stop. All five
narrow-secondary-mode cases stopped while missing substantial posterior mass;
all nine non-stopping cases already met the external accuracy target at their
last checkpoint. References were accessed only after decisions were saved.
Two separate COMPAS pilots also completed: the perfect-measurement case stopped
at 2098 exact calls (reference KL 0.000247), while the uncertain-measurement case
did not stop at its 2146-call final checkpoint despite reference KL 0.000239.
Both reference numerical checks passed. These are implementation pilots, not
a new calibration ensemble. See `operational_stopping/report_v1/RESULTS.md` and
the frozen `operational_stopping/PROTOCOL.md`; 87 fast tests pass, 3 deselected.
No confirmation thresholds were retuned and the existing benchmark campaign
was left unchanged. Earlier proposed-rule statements below are historical.

Clarification after the convergence extension: a direct-reference accuracy
threshold is an offline benchmark criterion, not a deployable stopping rule.
The methods and Table 1 now make that distinction explicit. See
`event_uncertainty/OPERATIONAL_STOPPING_PLAN.md` for the original candidate pre-refit
prediction checks plus GP-only posterior stability and a small fresh audit.
That candidate failed the subsequent validation described above; the frozen
benchmark campaign is unchanged. Table 1 summarizes the completed 100-catalogue groups
at 30 and 100 expected detections, with median and 10th--90th percentiles.
The old fixed-budget Table 2 has been removed from the manuscript, with raw
records retained. Plot/table labels now say "expected detections" explicitly.
The original 307-event corner illustration was separately extended: BO met the
reference-assisted criterion at 994 points, random at 1122. Its corner inset
shows both posterior change and reference error through the BO endpoint.
78 fast tests pass; 3 heavy tests are deselected.

The essential campaign is complete (refine, sweep, robustness). The original
500-event uncertain case now passes reference checks after independent
refinement; its random fit still misses the posterior accuracy target at 738.
Raw original and refined records are retained separately. The width sweep has
one remaining numerical-reference issue at twice nominal errors; the robustness
study has one BO accuracy miss in the larger-exposure group. These are retained,
not silently removed when changing the experiment's stopping rule.

User requested simpler abstract/prose, explicit candidates/Sobol/rate scaling,
lighter zero-rate maps, identifiable panels, 100 catalogues per regime and
evaluation requirements determined by posterior convergence rather than 738.
The new frozen design is `event_uncertainty/CONVERGENCE_PLAN.md`. It generates
600 fresh catalogues (100 in each of six regimes), runs a separate extension
pilot, and then compares BO/random with selected-point-only training evaluations.
Its status is `event_uncertainty/convergence_20260910/campaign_status.json`.
Reference-assisted stopping is explicitly distinguished from an operational
stop. Resource guards and unresolved references remain reportable outcomes.
75 fast tests pass, 3 heavy tests deselected; the four-checkpoint smoke run
reproduces the original likelihood up to a data-only constant and uses exactly
one rate evaluation per supplied training label. No new COMPAS evolution runs.

The first extension-pilot catalogue (original perfect-measurement case zero,
100 expected/96 observed events) reached the two-checkpoint convergence rule at
866 BO and 1378 random training evaluations. This is one pilot catalogue, not
the fresh-ensemble result or a replacement for the all-catalogue budget table.
All 600 fresh catalogues have been generated; their inference remains queued
behind the extension pilot. Consult the campaign status for subsequent progress.

Earlier records below describe their state at the time and may be superseded.

## Current work — 2026-09-10

The continuous-event pilot is documented in `event_uncertainty/README.md`.
The user authorized manuscript cleanup and essential follow-ups. The frozen
design is `event_uncertainty/ESSENTIAL_RUNS.md`: reference-only refinement of
the unresolved original case, paired half/double measurement-width experiments,
and three-injection/three-design-seed robustness with BO/random/Sobol.

The local serial campaign was launched with `run_essential_campaign.py`.
Read `event_uncertainty/essential_20260910/campaign_status.json` and the phase
logs for actual progress; launch is not completion. Original results are retained.
The separate smoke run completed; 70 fast tests pass, 3 heavy tests deselected.
Continuous-sensor prior-drawn calibration is the next stage after reviewing and
freezing the supported procedure, and has not been launched as part of this stage.

The manuscript now separates continuous noisy-event accuracy from categorical
calibration, with categorical controls and the full corner plot in the appendix.
No prospective result has been inserted into the paper. No commits or remote
Overleaf synchronization were performed. Entries below are prior evidence/history.

The ANN implementation comparison and actual LVK inference are out of scope.
The paper can study training requirements and posterior accuracy under the fixed
binned COMPAS model, with a qualified literature discussion. Its scientific claims
must distinguish surrogate accuracy from calibration of the finite-population target.

| Question | Current evidence | Boundary of the claim |
| --- | --- | --- |
| Is the likelihood implementation trustworthy? | Sparse contraction parity, analytic amplitude checks, GP noise-freeze regression and sampler fixes; 56 fast tests | Exact for the tested numerical model, not a validation of astrophysical assumptions |
| Can the GP reproduce direct constraints? | Fresh 100-catalogue 482-label ensemble: all posterior gates pass; 68% widths within 2.1%, 95% within 3.8% | Mean target, ideal observations, fixed prior/grid/selection; conditional amplitude uses direct rates |
| Is the likelihood surface accurate? | Median forward KL 0.00158; 99/100 meet both KL <0.01 | One retained borderline failure; no universal 482-point guarantee |
| Is numerical integration adequate? | Both direct and GP references pass independent 16384-point checks on all 100 cases | Finite quadrature remains an approximation; one GP 90% containment decision is sensitive |
| Does adaptive acquisition help? | Three-catalogue matched pilot: likelihood-variance policy reaches all-case gate at 482 versus 610 labels | Exploratory discrete checkpoints; not a wall-time benchmark or validated online stopping rule |
| Is the finite COMPAS population adequate? | 32M/512M differences are material; disjoint blocks and bootstrap characterize conditional sensitivity | Separate simulated campaigns unavailable; bootstrap cannot repair missing support |
| Are intervals calibrated? | Fresh alpha68 is 57% direct /56% GP; independent 1000-catalogue matched-population direct control gives 67.1–68.9% coverage at 68%, compatible with nominal | Original deficit retained; larger control is direct-only and does not calibrate 1000 GPs or population mismatch |
| Is ordinary sampling implemented correctly? | Correct starts and diagnostics; prior 512M four-chain check passes after smaller step size | Fresh ensemble uses quadrature, not 100 independently validated NUTS runs |
| Does GP agreement survive uncertain measurements? | Fresh 100-catalogue categorical-response experiment: all posterior/KL/numerical gates pass; 68% widths within 1%, 95% within 1.3% | Controlled response conditional on detection; not realistic LVK PE; individual sigma coverage deficit remains reported |
| Do posterior predictive catalogues resemble the observations? | First three uncertain cases: direct/GP count and mean-bin predictive distributions agree; observed summaries are not extreme | 1000 replicate catalogues per method/case; selected summaries are not proof of model adequacy |

Authoritative fresh-ensemble report: [CONFIRM_482.md](joint_pilot/CONFIRM_482.md).
Original failed models/results are retained as diagnostic history and must not be
mixed with current validation results. The earlier 738-label calibration does not
transfer automatically to a changed acquisition rule.

The deficit did not persist in the larger independent direct-only control; see
[control results](joint_pilot/DIRECT_RANK_CONTROL.md). Next, assemble the paper's
accuracy, training-budget and population-sensitivity evidence with these scope
limits. Adding a 1-year catalogue would extend the regime and needs
its own direct-reference check; it is not already supported by these results.
Do not expand to an ANN implementation, more pixel-grid optimization or real LVK
systematics to resolve this targeted mock-calibration question.

User scope update: emphasize GP-guided selection of parameter evaluations for
population-synthesis analyses. The executed demonstration varies post-processing
parameters, not binary-evolution inputs requiring new COMPAS runs. See
[uncertain-measurement results](joint_pilot/UNCERTAIN_MEASUREMENTS.md) and
[direct-reference/selection audit](joint_pilot/DIRECT_REFERENCE_AND_SELECTION.md).

The uncertain-measurement direct-only follow-up completed on 1000 independent
catalogues: 68% coverage 67.2–69.1%, 90% 89.9–91.7%, 95% 94.8–95.5%; all nominal
levels within individual exact 95% binomial intervals. The original sigma deficit
did not persist. This is a separate matched-model control, not 1000 GP fits.
Current fast suite: 60 passed, 3 deselected, one existing Matplotlib warning.
Next: bounded online cost comparison and paper assembly, keeping the measured
post-processing scope explicit.


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


## 2026-09-10: continuous event-level uncertain-measurement pilot completed

See [frozen design and results](event_uncertainty/README.md). Twenty paired
catalogues at 50/500 expected events produce 80 BO/random trajectories and 400
checkpoints. This uses continuous log-coordinate measurement likelihoods, with
analytic within-bin integration and a separate two-prior PE-recycling check.
It is distinct from the earlier categorical-response calibration ensemble.

At 738 points, perfect-measurement BO/random successes are 10/10 versus 10/10
at low counts and 10/10 versus 0/10 at high counts. Uncertain-measurement results
are 10/10 versus 10/10 and 9/10 versus 0/10. The remaining high-count uncertain
BO case satisfies accuracy checks, but its reference endpoints differ by
0.0725 SD (tolerance 0.05 SD); it remains unresolved and visible in the results.
For low-count uncertain data, all BO cases pass by 354 points versus 610 for
random acquisition. This supports matched-budget acquisition efficiency within
the declared mock model, not a general speedup or validated LVK analysis.

The manuscript includes model equations, convergence bands and two preselected
posterior-predictive examples. Fast suite: 66 passed, 3 deselected. Next priorities
are a reference-only precision check for the unresolved catalogue, robustness
across injections/design seeds, and cost accounting. A broader event-level
coverage ensemble has not been executed. No new COMPAS runs or LVK inference.
