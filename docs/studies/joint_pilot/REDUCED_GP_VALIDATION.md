# Repaired reduced GP validation — 2026-09-08

The repaired three-dimensional GP reproduces the direct posterior in the tested
small-catalogue, fixed-grid regime. This supersedes the rejected fixed-noise
experiments, which did not actually freeze the GP noise. It does not validate
the old saved models, GP predictive variance, larger catalogues, or LVK inference.

## Frozen procedure and controlled checks

Use the existing mass10 grid and full prior. Analytically marginalize the linear
SFR amplitude under its uniform prior; fit the remaining alpha, sigma and d_SFR.
Use Matérn-5/2, sqrt scaling, genuinely fixed noise, and 250 optimization steps.
Start with 512 full-prior Sobol points; add 128 GP-guided points with a 10% uniform
exploration component. Retain every training point. Add 98 points on the faces
of a 5x5x5 prior grid, giving 738 points total. These boundary anchors were added
after independent validation located residual errors near prior boundaries;
they are a documented exploratory change, not a pre-registered choice.

The matched 585-point reduced fit, with only the noise freeze repaired, achieved
fresh direct/GP posterior RMS errors 0.076/0.077 lnL at 250 optimizer steps.
Increasing to 2000 steps did not improve it. Independent fresh training designs
use no direct-reference posterior samples. Seeds 17, 29 and 43 were checked on
8192 full-prior quadrature points, then a second independent scramble.

On 512M, the independent validation scramble gives posterior-weighted lnL RMS
errors 0.0506, 0.0543 and 0.0488 after removing an irrelevant constant offset.
Across seeds, 68% widths differ from direct by at most 1.18%; means by 0.006 SD;
interval endpoints by 0.014 SD; correlations by 0.005. Direct quadrature ESS is
3288. The 32M anchored fits also pass (RMS about 0.05–0.06).

These are joint posterior quadrature checks, not three independent MCMC runs.
The separate four-chain 512M sampler result is recorded in
`reduced_seed_results_512M_anchored/seed_17/sampler_summary.json`. The initial
0.9 target acceptance run had one divergence and failed its gate; its summary
and chains are preserved with `accept90` suffixes. A 0.99 rerun addresses this
sampler issue without changing the GP or target. The rerun passes: zero
divergences, maximum R-hat 1.0018, minimum bulk/tail ESS 2378/2128, and
means within 0.012 direct SD. It took 72.0 seconds.

Final fast suite: 42 passed, 3 deselected (11.95 s); one existing Matplotlib
deprecation warning. No commit or push.

## Twenty-catalogue pilot

Generate 20 independent prior-drawn truths and Poisson catalogues at 0.1 yr
using blocks 0–7 of the existing population (256M). Analyse using disjoint blocks
8–15 (256M). These are disjoint subsets of one simulation campaign, not verified
independent campaigns. Refit the frozen 738-point procedure for each catalogue,
using one training seed. Evaluate direct and GP posteriors with common 8192-point
full-prior quadrature and exact conditional amplitude draws.

All 20 cases evaluated; no failures were excluded. Counts range from 0 to 125
(two zero-event catalogues). The recorded accuracy gate requires mean shifts
below 0.2 direct SD and 68% widths within 10%; all pass. Observed maximum mean
shift is 0.0112 SD; width ratios range 0.9839–1.0066; posterior-weighted lnL RMS
ranges 0.0044–0.0874. Direct quadrature ESS ranges 703–3514. The small consistent
narrowing in d_SFR remains measurable and should not be described as exact parity.

| Parameter | Direct 68% coverage | GP 68% coverage | Direct 95% coverage | GP 95% coverage |
| --- | --- | --- | --- | --- |
| alpha | 15/20 | 15/20 | 19/20 | 19/20 |
| sigma | 16/20 | 16/20 | 19/20 | 19/20 |
| a_SFR | 14/20 | 14/20 | 19/20 | 19/20 |
| d_SFR | 11/20 | 11/20 | 18/20 | 18/20 |

The containment outcomes agree case by case. Twenty trials have wide binomial
uncertainty: these counts are a failure-screening pilot, not precise calibration
or proof of nominal coverage. In particular, 11/20 is not evidence of established
68% coverage for d_SFR. Both methods share quadrature error; independent scrambles
were checked for the 81-event reference, not all twenty mocks. The pilot took
91.5 seconds excluding interpreter/import overhead.

## Evidence and reproduction

- `repaired_reduced_gp.py`: matched noise-repair experiment.
- `reduced_seed_check.py --boundary-anchors`: independent 32M training seeds.
- `reduced_seed_check.py --population 512 --boundary-anchors`: 512M transfer.
- Add `--evaluate-only --validation-seed 20260926` for independent quadrature.
- `sample_reduced_final.py`: four-chain 512M check.
- `reduced_calibration_pilot.py`: noisy-catalogue pilot, cases and manifest.
- `plot_reduced_validation.py`: `reduced_calibration_results/validation_overview.png`.

Run with the repository virtual environment and MPLCONFIGDIR pointing to a
writable temporary directory. Existing population operators are required; see
`../population_stability/README.md`. No additional COMPAS simulations were run.

## What is and is not ready

This supports an approximate posterior proof of principle for the tested ideal
binned catalogues. The conditional amplitude reconstruction still uses exact
rates, so this is a validation hybrid with no demonstrated end-to-end GP speedup.
The accelerated direct likelihood remains the reference and a viable fallback.
Do not add GP variance to account for finite population uncertainty: predictive
variance calibration and population uncertainty are separate unresolved tasks.

Scope update (user, 2026-09-08): actual LVK inference is excluded from this paper.
The next required step is the fresh 100-catalogue mock ensemble with the procedure
frozen and numerical integration checked independently. Changing observing-run
sensitivity/exposure, event measurement systematics and interpretation of high-mass
systems are future real-data work. Keep the full simulated mass support. A
controlled uncertain-event demonstration remains optional; it is not required to
finish the mock-only proof of principle.


Completed follow-up: [100-catalogue frozen-procedure ensemble](CALIBRATION_100.md), including refined direct references. All 100 interval comparisons pass; actual LVK work is outside this paper.
