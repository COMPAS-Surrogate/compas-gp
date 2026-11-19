# Fresh-catalogue confirmation of the 482-label design

2026-09-09. This experiment freezes the likelihood-variance acquisition developed
on three catalogues and tests it on a new, independent set of mock-catalogue draws.
The ANN implementation comparison is explicitly dropped; literature discussion
will compare stated training designs without an efficiency-superiority claim.

## Frozen configuration

One hundred prior-drawn truths, random seed 20261020, perfect binned Poisson
observations at 0.1 yr. Generate from blocks 0–7 and analyse with blocks 8–15 of
the existing 512M population (disjoint 256M halves from one campaign). Fixed
mass10 grid, full mass support, original prior and selection approximation.

For every catalogue use training seed 17, 128 initial Sobol points plus 98 boundary
anchors, then two 128-point batches sampled in proportion to exact GP likelihood
variance with 10% uniform exploration. Refit at 226,354,482 points; retain all
points. Matérn-5/2, square-root transform, genuinely fixed noise, 250 optimizer
steps per fit. Inference uses the deterministic transformed-GP mean target.
No failed fit receives extra labels, no prior narrowing, and no hyperparameter
retuning is allowed in this ensemble. One training seed across these catalogues
complements, rather than replaces, the earlier three-seed pilot.

## Independent checks

Use two 16384-point full-prior Sobol integrations (seeds 20261021 and 20261022),
and exact conditional amplitude CDFs and moments for both GP and direct targets.
This removes conditional-amplitude Monte Carlo noise. Direct rate evaluations
remain part of amplitude reconstruction; this is still a validation hybrid.

Report three separate gates:

- Posterior agreement: all four means/endpoints within 0.2 direct SD and 68%/95%
  widths within 10%, on both numerical scrambles.
- KL: both directions below 0.01 nats, on both scrambles. This is stricter than
  parameter-constraint agreement and is not combined with it to hide failures.
- Numerical agreement: GP and direct means/endpoints change by <0.05 SD and
  68% widths by <3% between scrambles.

Coverage and posterior ranks are prior-averaged cross-population diagnostics,
not exact-model SBC or fixed-parameter frequentist coverage. Binomial intervals
are individual intervals, not a simultaneous certification. Do not pool the old
pilot/100-catalogue results with this fresh ensemble as if the procedure were
unchanged. Failures remain in the attempted-catalogue denominator.

## Reproduction

```
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/confirm_482.py
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/summarize_confirm_482.py
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/plot_confirm_482.py
```

`confirm_482_results/` preserves the catalogue arrays, every fitted model/scaler
and training set, per-case metrics and summaries, configuration, source hashes and
runtime versions. No original results are overwritten. The benchmark evaluates
shared candidate/reference caches offline; training-label count is not total
executed direct-call cost. This does not establish an online stopping rule or
end-to-end speedup.

## Fresh-ensemble results

All 100 catalogues evaluated successfully; none were excluded. Counts span 0–159
(four empty catalogues). All 100 pass the full posterior-constraint gate and all
100 pass numerical agreement. The 68% interval-width ratios span 0.9793–1.0094,
and 95% ratios 0.9622–1.0063. Maximum mean shift is 0.0272 direct SD; maximum
endpoint shift is 0.1429 SD. Median forward KL is 0.001577 nats.

99/100 pass the separate KL gate. Case index 30 (123 events) has maximum forward
KL 0.0100093, just above the frozen 0.01 threshold; it remains a recorded failure
of that criterion. The threshold was not rounded or changed after seeing this case.
Maximum reverse KL over the ensemble is 0.007364.

| Parameter | Direct 68% | GP 68% | Direct 90% | GP 90% | Direct 95% | GP 95% |
| --- | --- | --- | --- | --- | --- | --- |
| alpha | 57/100 | 56/100 | 93/100 | 93/100 | 97/100 | 97/100 |
| sigma | 70/100 | 70/100 | 92/100 | 92/100 | 96/100 | 96/100 |
| a_SFR | 67/100 | 67/100 | 87/100 | 87/100 | 93/100 | 93/100 |
| d_SFR | 62/100 | 61/100 | 90/100 | 87/100 | 95/100 | 95/100 |

**Do not call this a clean nominal-coverage pass.** Alpha's 68% coverage is below
nominal for both methods: exact individual 95% binomial intervals are approximately
[46.7%,66.9%] for direct and [45.7%,65.9%] for GP. The discrepancy is mainly shared
by the direct target; accurate emulation alone does not establish calibration of
that target. Multiple parameters/levels were inspected, so neither one interval
nor non-rejection elsewhere supplies a simultaneous conclusion.

All direct coverage containment decisions agree between numerical scrambles.
GP 68%/95% decisions also agree; one d_SFR 90% decision changes. We run a direct-only
matched-population control on these same catalogues, using their generation
population for inference, to assess whether finite-population mismatch explains
the observed coverage deficit. This control does not involve GP retuning.

Main work took 322.2 seconds including 89.2 seconds of shared rate-cache work,
excluding imports. 56 fast tests pass, 3 deselected; one existing Matplotlib
warning. The calibration PNG/PDF was inspected. No commit/push, ANN work, new
COMPAS populations or LVK data.


## Matched-population follow-up

The direct-only matched-population control completed on the same 100 catalogues.
At 68%, coverage is [57,69,62,62] for alpha,sigma,a_SFR,d_SFR. Alpha therefore
remains 57/100 when population mismatch is removed. Switching halves does not
explain its deficit in this ensemble. This is not evidence that the discrepancy
has been resolved, nor that one finite ensemble proves an inference bias.

A separate direct-only rank-calibration control with 1000 fresh prior-drawn
catalogues, fixed seed 20261025, is used to assess whether the deficit persists.
It uses the same population for generation and inference and two independent
16384-point integrations. It does not retrain a GP or choose a successful seed.

This control completed: 68% coverage is [67.1%,67.2%,68.9%,67.3%]. All four
parameters' 68%, 90% and 95% coverage values are compatible with nominal within
individual exact 95% binomial intervals. The earlier deficit did not persist in
this independent direct-only experiment; the original 100-catalogue result is
retained above. This is evidence for internal calibration of the matched direct
target, not a 1000-catalogue GP validation or a resolution of population-systematic
uncertainty. Full method and results: [DIRECT_RANK_CONTROL.md](DIRECT_RANK_CONTROL.md).
