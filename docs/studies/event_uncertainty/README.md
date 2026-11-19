# Continuous event-level measurement uncertainty

The additive essential campaign is specified in [ESSENTIAL_RUNS.md](ESSENTIAL_RUNS.md).
Its local status is in `essential_20260910/campaign_status.json`. It preserves
all original results below and does not count smoke checks or running jobs as
completed scientific evidence. The width sweep and injection/design robustness
are separate from the future continuous-sensor calibration ensemble.

Completed on 2026-09-10: 20 paired catalogues, 80 acquisition trajectories and
400 checkpoints. The inference/validation runner took 894.5 seconds on this
machine; this is not an end-to-end speedup measurement.

The frozen design has ten catalogue realizations at each of 50 and 500 expected
detections. Each realization supplies both perfect and uncertain versions of the
same latent events. Counts remain Poisson-distributed. A fixed interior injection,
fixed blocks0-7 COMPAS population and the existing O3 selection approximation are
used throughout. This is an event-level conditional-on-detection mock sensor,
not strain parameter estimation or an LVK analysis.

Within each nonzero finite population bin, latent chirp mass and redshift are
uniform in linear coordinates. Observed log coordinates follow independent
Gaussians centered at the true log coordinates. Per-event widths are
(0.03, 0.3)*12/rho, with rho drawn from p(rho) proportional to rho^-4 above 12,
independently of the hyperparameters. No observed measurements are clipped or
filtered to the latent population domain. These assumptions are explicit rather
than inherited from the legacy mock posterior-cloud generator.

The event likelihood is integrated analytically across each population bin.
This resolves measurement errors narrower than a coarse bin. Zero-rate bins are
excluded only when the fixed operator has no entries there at any hyperparameter;
the full finite active support, including high masses, is retained.

Before the GP comparison, mock PE posteriors for every event in the first small
catalogue are generated under two normalized linear-coordinate analysis priors:
constant density and density proportional to M*z^2. Both yield independent
truncated Gaussian draws in log coordinates. Inverse-prior and bin-area weighting
must recover population-posterior summaries from analytic event integrals.
8192 and 32768 samples per event check sample-count dependence; the final check
requires means/endpoints within 0.05 reference SD and 68% widths within 3%.
Samples are used to validate likelihood recovery; analytic event integrals are
then used for GP training, separating measurement uncertainty from PE Monte Carlo
error. This is not a study of finite-PE-sample inference across all catalogues.

The paired comparison uses identical GP settings and the common 226-point
initialization, followed by BO or IID uniform acquisition at budgets
226,354,482,610,738. All points are retained. Two independent full-support
importance integrations use 32768 generated proposal points per exposure and
scramble; truth/Fisher information is used for validation proposals only.
Accuracy and independent-reference precision use the previous study's tolerances.
All failures must remain reported. Twenty fixed-injection catalogues are a pilot,
not simulation-based calibration or a precise coverage study.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/event_uncertainty/run_study.py
```

The runner refuses to overwrite an existing result directory. Raw catalogues,
reference maps, training checkpoints, recycling checks and original source hashes
are retained. GP/acquisition overhead and offline validation costs are not a
claimed end-to-end speedup. This work does not run new COMPAS populations.

Validation completed: 66 fast tests pass (3 heavy tests deselected; one existing
Matplotlib deprecation warning).
The full first-small-catalogue prior-recycling test passes with both priors at
both 8192 and 32768 samples per event. At 32768, the largest mean discrepancy
is 0.00056 reference SD, the largest interval-endpoint discrepancy is 0.00256 SD,
and the largest 68% width discrepancy is 0.030%. These numbers validate this
specific mock measurement/prior calculation, not the whole GP study or LVK PE.


## Pilot results

At 738 training points, validated successes out of ten are:

| Expected events | Measurements | BO | Random |
| --- | --- | --- | --- |
| 50 | Perfect | 10 | 10 |
| 50 | Uncertain | 10 | 10 |
| 500 | Perfect | 10 | 0 |
| 500 | Uncertain | 9 | 0 |

The remaining 500-event uncertain BO case (catalogue 8, 474 observed events)
passes both reference accuracy checks, but the reference interval endpoints
differ by 0.0725 SD, exceeding the 0.05 SD numerical tolerance. It remains
unresolved and is not counted as a validated success. All cases stay in the
KL curves. There are no other final numerical failures.

For uncertain measurements, all ten low-count BO cases first pass at 354 points,
versus 610 for random acquisition. Median final forward KL is 0.000250 versus
0.00268 at 50 expected events, and 0.00144 versus 0.0297 at 500. Medians include
the unresolved case. These are matched-budget comparisons after a common
initialization, not comparisons of independent random initial designs.

Median paired uncertain/perfect 68% reference-width ratios, in alpha, sigma,
a_SFR, d_SFR order, are [1.085, 0.997, 1.000, 1.001] at 50 expected events and
[1.133, 1.085, 1.050, 1.010] at 500. Twenty fixed-injection catalogues cannot
establish nominal credible-interval coverage.

Two preselected predictive examples contain 56 and 515 observations. Each uses
1000 replicated catalogues with amplitude, count and measurement variation.
The figures show measured-coordinate empirical CDFs with pointwise 90% bands;
they are not simultaneous acceptance regions or population-wide calibration.

After inference completes, regenerate the summary, figures and manuscript input:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/event_uncertainty/publish_results.py
```

Raw results and source hashes are in `results/`. The generated manuscript input
is `overleaf/output/bo_study/event_uncertainty.tex`, and figures are in
`overleaf/figures/bo_study/event_uncertainty_*.{pdf,png}`.
