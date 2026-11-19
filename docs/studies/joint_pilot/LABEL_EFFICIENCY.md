# Posterior-focused acquisition and training-label efficiency

2026-09-08. User prioritized reducing expensive evaluations and a comparison
with Riley & Mandel (2023). The earlier result concerns one EI/latent-variance
policy, not whether adaptive acquisition is useful in general.

## New bounded experiment

Use the same three simulated catalogues, three training seeds, 226 initial points,
five budgets and two validation scrambles. Replace the acquisition score with
Var_GP[L(theta)], computed from exact first and second likelihood moments through
the fitted square-root transform. Sample 128 candidates per batch in proportion
to this variance, with 10% uniform exploration. Keep all old points and the full
prior. This is a pointwise likelihood-variance heuristic, not an integrated
variance-reduction optimizer. It does not use direct labels until acquired.

Both mean and expected-likelihood posterior targets are still assessed using the
same fitted GP; the acquisition change is separate from uncertainty propagation
in the posterior. The Gaussian process noise remains fixed. The pointwise score
can over-select correlated candidates in a batch; no diversity optimizer was added.

## Retrospective diagnostic

For each catalogue/seed trajectory, require both KL directions <0.01 nats,
shape-parameter means within 0.2 direct SD, and shape 68% widths within 10%, on
both numerical scrambles. Count the first tested budget satisfying these checks.
This explicitly exploratory gate is not the full four-parameter coverage gate,
a validated online stopping rule, or an exact-KL guarantee. Training design and
numerical audits verify matching initial data, unique points and retained prefixes.

Mean-target results:

| Labels | Old EI/variance passes | Sobol passes | Likelihood-variance passes | New-rule median forward KL |
| --- | --- | --- | --- | --- |
| 226 | 0/9 | 0/9 | 0/9 | 0.04931 |
| 354 | 3/9 | 3/9 | 7/9 | 0.00785 |
| 482 | 5/9 | 6/9 | 9/9 | 0.00214 |
| 610 | 9/9 | 9/9 | 9/9 | 0.000697 |
| 738 | 9/9 | 9/9 | 9/9 | 0.000288 |

The first tested budget passing all nine trajectories falls from 610 to 482:
128 fewer supplied training labels, approximately 21%. These are discrete
checkpoints, not the exact minimum required budget. At 738 labels the median KL
is about eight times smaller than the old policy/Sobol, but fewer labels to an
accuracy target is the more relevant cost metric. No catalogue or seed was dropped.

This is an exploratory result on already-used catalogues. It needs confirmation on
fresh catalogues before changing the validated pipeline. The 100-catalogue gate
from the earlier recipe does not automatically transfer to this new acquisition.
Full rate caches and direct validation grids were evaluated offline, so this run
does not demonstrate a 21% reduction in total executed direct calls or wall time.
Charge validation, fitting and acquisition costs when testing deployment savings.

## Evidence and next step

```
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/bo_uncertainty_curve.py --strategy posterior_variance
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/compare_label_efficiency.py
```

`posterior_variance_uncertainty_results/` holds all curves and checkpoint training
data. `label_efficiency_results/` holds paired summaries, first-passing budgets
and `label_efficiency.png`/`.pdf`. Runtime 116.0 s including cached direct rates,
excluding imports. 56 tests pass, including independent integration tests of both
likelihood moments. The figure was visually inspected; production defaults unchanged.

Freeze this acquisition and test the 482-label design on fresh catalogues with
full posterior checks. Develop an online stopping audit only after its reliability
and direct-call cost are measured. The user subsequently dropped ANN implementation entirely. The
[Riley–Mandel comparison plan](RILEY_MANDEL_COMPARISON_PLAN.md) now contains only
literature context with no matched efficiency claim. Fresh-catalogue confirmation
is the active analysis task.
