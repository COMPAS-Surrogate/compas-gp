# Controlled uncertain measurements and posterior predictive checks

Frozen design, 2026-09-09, before scoring the pilot. This extends the mock-only
paper and keeps the 482-label likelihood-variance acquisition unchanged.

## Generative model

Draw population parameters from the existing full prior. Generate a 0.1-year
Poisson catalogue with the existing detected rate operator, using blocks 0–7
of the 512M population. For every detected event apply a categorical measurement
response R[measured bin, latent bin]. Its offset weights follow a correlated
discrete Gaussian with sigma=1 bin, rho=0.6 and offsets from -3 to +3 bins on
each axis. Boundary offsets saturate in the edge category; this is an explicit
toy sensor definition, not clipping of physical PE samples. All domain bins,
including the high-mass guard, remain present. The response is column stochastic,
so it preserves total rate and each paired catalogue's event count.

Inference uses the same population and response. Measured rates are R times
latent detected rates; the count likelihood therefore exactly sums over unknown
latent event bins. This is a controlled categorical measurement experiment, not
a realistic Gaussian error model in physical mass/redshift or an LVK PE analysis.
It is conditional on the existing detection model: no further selection is
applied to the noisy measurements. The response does not change the amplitude's
linear scaling, so analytic amplitude marginalization remains valid.

## Experiments and fixed checks

1. Three paired perfect/uncertain catalogues, seed 20261030, to check the extension.
2. If the pilot passes, 100 fresh uncertain catalogues, seed 20261101, without
   changing response, prior, GP, acquisition or 482-label budget.
3. Two independent 16384-point full-prior Sobol references per experiment,
   seeds catalogue_seed+1 and catalogue_seed+2, with exact amplitude summaries.
4. Same posterior, KL and numerical gates as CONFIRM_482.md. Retain all failures.
5. P–P plots from parameter posterior CDF ranks for direct and GP inference;
   finite-ensemble bands, separate from the earlier cross-population study.
6. For the first three cases, 1000 posterior predictive replicated catalogues
   for each method, including amplitude uncertainty and fresh Poisson counts.
   Compare event count, mean measured mass-bin and redshift-bin indices, and
   marginal measured count distributions. These are posterior predictive checks,
   not uniform p-values or a replacement for parameter calibration.

Rate caches are shared across catalogues for validation. GP acquisition sees
candidate labels only after selection, but the benchmark precomputes their rate
maps. Training-label counts must not be reported as total executed direct calls
or an online wall-time speedup. Conditional amplitude and predictive recovery
still use direct rates.

## Reproduction

```sh
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/uncertain_measurements.py --catalogues 3 --seed 20261030 --paired --outdir uncertain_pilot_results
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/uncertain_measurements.py --catalogues 100 --seed 20261101 --outdir uncertain_100_results
```

Output directories refuse overwrite and retain source hashes, response matrix,
latent/measured catalogues, all trained models and acquired points, summaries,
independent numerical checks and replicated-data diagnostics.

## Paired pilot results

All six fits (three perfect and three uncertain) pass the posterior, KL and
numerical gates. The three catalogues have 34, 0 and 49 events. Under uncertain
measurements, the maximum forward KL is 0.001295 and the largest mean shift is
0.00363 direct SD; 68% widths differ from direct by at most 0.74%.

The direct uncertain/perfect 68% alpha-width ratios are 1.046, 1.000 and 1.374.
Measurement uncertainty changes the constraints, while the GP follows the new
target. Individual realized intervals need not all widen. The empty catalogue's
direct posterior is unchanged, as required by rate conservation. This three-case
pilot does not establish calibration. It passed without retuning, so the frozen
100-catalogue confirmation proceeded with the predeclared seed.

Predictive plots retain the first three cases, including the empty catalogue.
Mean bin indices for empty replicates are defined as zero. In pilot case 0 the
observed mean redshift-bin index is near the upper predictive tail for both
methods. Do not describe all predictive summaries as perfectly centered or use
posterior predictive tail probabilities as uniform calibration p-values.

Generate aggregate diagnostics and four PNG/PDF figures with:

```sh
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/summarize_uncertain_measurements.py --outdir uncertain_100_results
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/plot_uncertain_measurements.py --outdir uncertain_100_results
```

## Fresh 100-catalogue confirmation

All 100 fits evaluated successfully, with 0–141 events and nine empty catalogues.
All 100 pass the posterior, KL and numerical gates, without extra training labels.
The 68% GP/direct width ratios span 0.99044–1.00944; 95% ratios span
0.98746–1.00578. Maximum mean shift is 0.01189 direct SD and maximum endpoint
shift 0.04576 SD. The median casewise maximum forward KL is 0.000565 nats;
maximum forward/reverse KL are 0.00523/0.00509 nats. Total runtime was 326.9 s,
including 93.1 s of rate-cache work, excluding imports and plotting.

| Parameter | Direct 68% | GP 68% | Direct 90% | GP 90% | Direct 95% | GP 95% |
| --- | --- | --- | --- | --- | --- | --- |
| alpha | 76/100 | 76/100 | 88/100 | 88/100 | 93/100 | 93/100 |
| sigma | 67/100 | 67/100 | 82/100 | 82/100 | 90/100 | 89/100 |
| a_SFR | 68/100 | 67/100 | 89/100 | 89/100 | 97/100 | 97/100 |
| d_SFR | 66/100 | 66/100 | 88/100 | 88/100 | 94/100 | 94/100 |

The direct and GP rank CDFs remain inside the 95% per-panel DKW envelopes
(maximum deviation <0.090, envelope threshold 0.13581). These simultaneous
CDF bands do not mean every individual coverage check passes: sigma's 90%
coverage excludes nominal in its individual exact 95% binomial interval for
both methods, and sigma's GP 95% coverage also narrowly excludes nominal.
Retain these deficits, rather than calling the ensemble a clean calibration pass.

For the first three cases (21, 92, 39 events), the direct and GP replicated-data
distributions agree visually. Observed count/mean-bin summaries have predictive
upper-tail probabilities from 0.166 to 0.812 across methods. These chosen summaries
do not show an extreme discrepancy in these examples; that is not a global
model-adequacy claim. Marginal count plots retain pointwise, not simultaneous,
90% predictive bands, including finite Poisson scatter.

## Predeclared direct-only follow-up

After seeing the sigma deficit, fixed one independent 1000-catalogue control,
seed 20261105 with quadrature seeds 20261106/20261107. Same response, prior,
duration and matched population. Generate measured Poisson counts directly
from R times detected rates, equivalent by Poisson splitting to the paired
sensor construction. No GP is fitted, no seeds are screened, and this result
will be retained whether or not it is compatible with nominal coverage.

```sh
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/uncertain_rank_control.py
```

The larger direct-only experiment does not increase the number of GP validation
catalogues beyond 100 or test finite-population/model discrepancy.

The follow-up completed in 109.3 s, retaining all 1000 cases:

| Parameter | Direct 68% | Direct 90% | Direct 95% |
| --- | --- | --- | --- |
| alpha | 672/1000 | 900/1000 | 948/1000 |
| sigma | 681/1000 | 912/1000 | 949/1000 |
| a_SFR | 691/1000 | 917/1000 | 953/1000 |
| d_SFR | 675/1000 | 899/1000 | 955/1000 |

All nominal levels lie within their individual exact 95% binomial intervals.
At most one containment decision per parameter/level changes between numerical
scrambles (0.1 percentage point). The earlier sigma deficit did not persist in
this independent direct-only control. It is retained in the 100-case report;
the follow-up supports matched-model internal calibration without proving it
exact or validating new GP fits.

## Validation and paper scope

60 fast tests passed, 3 deselected, one existing Matplotlib warning. New tests
check response normalization, count/rate conservation, identity response,
explicit event-integral parity, simulated response including boundaries, and
posterior predictive mean/variance against analytic amplitude integration.
The four confirmation PNGs were visually inspected; PDF counterparts are saved.
Source hashes and runtime versions are in `evidence_provenance.json`.

The supported extension is a known correlated categorical sensor applied after
detection. Realistic physical measurement uncertainties, unknown selection,
changing observing epochs and new stellar-evolution simulations remain separate
claims. The next efficiency task is a bounded online timing comparison against
the accelerated direct baseline, recording setup, true evaluations, fitting,
acquisition and inference separately. Do not equate offline reference-cache
work with the GP training budget. No ANN implementation or LVK analysis was added.
