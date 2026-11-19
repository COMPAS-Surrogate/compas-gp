# Frozen-procedure mock ensemble — 2026-09-08

One hundred fresh catalogues pass GP-versus-direct posterior tolerances under the
specified ideal-observation model. This extends the 20-catalogue exploratory pilot;
the new truths and observations use a separate random seed, with no GP retuning.
Actual LVK inference is outside the paper scope.

## Design

Draw 100 truths independently from the full four-parameter prior. Generate Poisson
counts at 0.1 yr from the first 256M systems; infer with the disjoint last 256M.
Use the existing full-support mass10 grid and fixed selection model. Both halves
come from one COMPAS campaign; this is a cross-population calibration diagnostic,
not exact-model SBC or a comparison of verified independent COMPAS campaigns.
Counts range from 0 to 152 (median 33; six empty catalogues). These conclusions
are prior-averaged, not a precision calibration at a fixed 81-event count.

Freeze the repaired reduced GP protocol before this ensemble: Matérn-5/2, sqrt
scaling, fixed noise, 250 fit steps; 512 Sobol + 128 guided + 98 boundary points,
all retained; training seed 17 for every mock. The guided points depend on the
observations, never the injected truth. Amplitude is analytically marginalized;
the initial posterior comparison recovers it with exact conditional draws.

Use common 8192-point full-prior quadrature for GP/direct comparisons and another
independent scramble for direct-reference checks. The full posterior agreement
gate uses mean and endpoint shifts <0.2 direct SD and 68%/95% widths within 10%.
This supplements the pilot's simpler mean/68%-width flag without relaxing it.
The independent numerical gate is stricter: mean/endpoint changes <0.05 SD and
68%-width changes <3%. Keep numerical failures separate from GP-fit failures.

## Initial ensemble result

All 100 cases evaluated, all 100 pass the full GP/direct agreement gate. No failed
case was omitted. Maximum mean shift: 0.0138 SD; maximum interval/median endpoint
shift: 0.0491 SD. The 68% width ratios range 0.9757–1.0127; 95% ratios range
0.9878–1.0055. Centred posterior-weighted lnL RMS has median 0.0547 and maximum
0.1130. Thus the ~0.1 likelihood-error aim is not a universal per-case bound,
although all parameter-constraint tolerances pass.

| Parameter | Direct 68% | GP 68% | Direct 90% | GP 90% | Direct 95% | GP 95% |
| --- | --- | --- | --- | --- | --- | --- |
| alpha | 66/100 | 66/100 | 91/100 | 90/100 | 94/100 | 94/100 |
| sigma | 69/100 | 69/100 | 94/100 | 94/100 | 97/100 | 97/100 |
| a_SFR | 69/100 | 69/100 | 90/100 | 90/100 | 95/100 | 95/100 |
| d_SFR | 68/100 | 67/100 | 92/100 | 91/100 | 98/100 | 98/100 |

Exact binomial 95% intervals are saved in `summary.json`. For GP 68% coverage,
they are approximately alpha [55.8%,75.2%], sigma and amplitude [59.0%,77.9%],
and d_SFR [56.9%,76.1%]. Every reported nominal level lies in its corresponding
interval, but these are individual intervals, not a simultaneous guarantee.
Non-rejection is not proof of perfect calibration or immunity to systematic error.

The initial direct quadrature ESS spans 198–4000. Only 78/100 pass the strict
independent numerical gate: the worst endpoint change is 0.243 SD. Shared-grid
GP/direct agreement alone cannot resolve this error. The initial coverage table
and figure must be read alongside the higher-resolution reference check below.

## Numerical refinement

`check_ensemble_quadrature.py` computes two new 32768-point direct references and
integrates the amplitude conditional CDF and moments analytically. It does not
refit or improve the GP. Results are stored in `quadrature_refinement.json` and
`direct_reference_20261003.json` / `direct_reference_20261004.json`.

**Completed: all 100 numerical checks pass.** Across the two larger references,
maximum mean differences are 0.0019 SD, endpoint differences 0.0098 SD and 68%
width differences 0.35%. Every 68%/90%/95% coverage containment decision agrees
between these independent references. Refinement took 197.8 seconds.

Comparing the stored GP intervals against the refined direct reference, all 100
still pass: maximum endpoint shift 0.131 SD, 68% width ratios 0.9404–1.0343 and
95% width ratios 0.9774–1.0229. Use these conservative, independently referenced
bounds for the paper: **68% widths within 6%, 95% widths within 2.3%**, rather than
claiming the initial shared-grid 2.4% bound applies to an exact reference. The GP
intervals retain 8192-point integration error; they were not re-integrated or
retuned. This comparison tests the delivered approximation including that error.

One initial direct coverage entry changes: alpha 90% coverage becomes 90/100
(previously 91/100). All other table entries remain unchanged. The authoritative
refined counts are in `reference_assessment.json`; the refined figure is
`calibration_refined.png` / `.pdf`. Neither numerical checks nor matching coverage
counts establish calibration more precisely than the binomial intervals allow.

Final checks: 45 fast tests passed, 3 deselected, one existing Matplotlib warning.
New tests cover empty-catalogue Poisson normalization and the analytic conditional
amplitude moments/quantiles against independent numerical integration.


## Reproduction and assets

```bash
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/reduced_calibration_pilot.py --catalogues 100 --seed 20260930 --validation-seed 20261001 --check-quadrature --outdir reduced_calibration_100_results
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/check_ensemble_quadrature.py
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/assess_ensemble_reference.py
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/plot_calibration_ensemble.py
```

Outputs: `reduced_calibration_100_results/`. Initial ensemble main work took
346.9 seconds, excluding interpreter/import overhead. The calibration figure
exports PNG and PDF; posterior-rank shading is a 95% pointwise uniform-reference
band, not a simultaneous band. The lower panels use exact binomial intervals.
No new COMPAS runs, real observations, commits or pushes.

## Interpretation for the paper

Report a mock-only proof of principle conditional on the specified grid, prior,
population and observation/selection model. Keep GP interpolation accuracy,
finite-population sensitivity and numerical-reference precision separate.
A speedup over the accelerated direct likelihood is not established. An adjacent
finer-grid comparison is required for any claim of negligible information loss;
otherwise explicitly condition the inference result on the chosen binning.
