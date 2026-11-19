# Matched BO versus space-filling acquisition

The purpose is to distinguish improvement from more training labels from an
advantage specifically due to adaptive acquisition. This follow-up uses the same
three simulated catalogues (43,95,6 events), seeds (17,29,43), initial 226 points,
checkpoint budgets, transformed GP and validation grids as `BO_UNCERTAINTY.md`.
The grid and prior are frozen. No LVK analysis is included.

BO uses the existing mixture of expected-improvement and variance-weighted
acquisition, with uniform exploration. The baseline instead takes successive
128-point prefixes of the exact same 4096-point scrambled Sobol candidate pool.
It does not use candidate labels or validation results to choose points. The
initial 128 Sobol points and 98 anchors remain identical between methods; every
subsequent training set contains all earlier training points. Budget includes
anchors. The Sobol pool has its own scramble, independent of the initial design;
it is not a single continued Sobol sequence starting at the initial design.

Both strategies fit both mean and expected-likelihood posterior targets at
226,354,482,610,738 points. Within each strategy, these targets share the fitted
GP. Both KL directions are scored on the same two independent full-prior
quadrature scrambles. Average those numerical estimates within a trajectory
before reporting medians over the nine catalogue/seed combinations; two numerical
scrambles do not double the number of independent training experiments.

The comparison script verifies identical starting locations/labels and scores,
nested training sets, unique points and the exact prescribed Sobol prefixes.
It fails on audit errors rather than silently presenting an unmatched comparison.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/bo_uncertainty_curve.py --strategy sobol
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/compare_acquisition.py
```

The original BO outputs are preserved. Baseline outputs are in
`sobol_uncertainty_results/`; matched summaries and PNG/PDF figures are in
`acquisition_comparison_results/`. All candidate labels are cached offline for
both procedures, but only acquired labels enter training. Report supplied-label
budgets separately from offline reference/cache costs. These runs do not provide
a controlled end-to-end speed benchmark.

## Results

All matching audits pass. There are nine paired catalogue/seed trajectories,
not eighteen. Mean-target median forward KL (nats), after averaging the two
numerical scrambles within each trajectory:

| Training points | BO | Sobol | BO lower KL, out of 9 |
| --- | --- | --- | --- |
| 226 | 0.04931 | 0.04931 | tied |
| 354 | 0.02103 | 0.01675 | 4 |
| 482 | 0.00986 | 0.00687 | 3 |
| 610 | 0.00470 | 0.00323 | 4 |
| 738 | 0.00243 | 0.00234 | 5 |

For the expected-likelihood target at 738 points, medians are 0.002375 (BO)
and 0.002304 (Sobol), again 5/9 paired BO wins. Reverse KL is also comparable:
mean-target final medians 0.002429/0.002315, with BO lower in 6/9 trajectories.
The median paired BO/Sobol forward-KL ratio is 0.889 even though the ratio of the
two pooled medians exceeds one: these summarize different orderings. Neither
statistic alone establishes dominance. Inspect the paired values and facets.

The six-event catalogue favours BO at the final budget; the 43- and 95-event
catalogues show mixed results. One catalogue per event count cannot establish a
causal event-count trend. Both methods reduce posterior error with added labels;
this pilot does not establish a general adaptive-acquisition advantage. It also
does not prove equivalence. The uncertainty-aware target changes these comparisons
only slightly.

Baseline main runtime was 114.9 s, including offline cache generation, versus
118.0 s in the earlier BO run. These separate runs are not a controlled timing
benchmark. Do not translate this small wall-time difference into a speed claim.
51 fast tests pass (3 deselected, one existing Matplotlib warning). The figure
was visually inspected. No production defaults changed and no commit/push.

## Decision

Keep the grid frozen and retain the previously validated inference configuration.
The 100-catalogue validation used a different, frozen training recipe; this small
pilot does not automatically transfer that validation to pure Sobol or the new
batched BO recipe. Include the paired curves as an exploratory acquisition
comparison, with no claim that BO is required for accurate small-catalogue
posteriors. No further acquisition tuning is needed for the scoped paper.

Next, assemble the mock-only method/results narrative and evidence table:
GP correctness and posterior agreement, finite-population sensitivity, ensemble
coverage, and this bounded acquisition/uncertainty comparison. Any stronger claim
about acquisition optimality, speed or other catalogue regimes needs its own
validation and is optional future work.


User-directed follow-up: [posterior-focused label efficiency](LABEL_EFFICIENCY.md) improves the tested all-trajectory passing budget from 610 to 482. The earlier conclusion remains specific to the EI/variance policy. [Riley–Mandel comparison protocol](RILEY_MANDEL_COMPARISON_PLAN.md) distinguishes rate emulation from catalogue-specific lnL emulation.
