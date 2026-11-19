# What the direct reference and selection model actually do

Code audit, 2026-09-09. This distinguishes the validated mock path from the
legacy observation-loading path and potential future LVK inference.

## Why direct inference became tractable

The earlier integrator recomputed binary/detection/redshift contributions for
each post-processing parameter point. `FixedPopulationRates` precomputes their
parameter-independent sparse linear map, then contracts it with the changing
metallicity distribution and star-formation history. It preserves the original
discretized rate calculation. The recorded matched timing in the joint-pilot
README is 0.593 seconds versus 0.00234 seconds per evaluation (about 253 times),
not including one-time operator construction. These are historical local timings,
not a fresh timing benchmark for every population or measurement configuration.

The amplitude a_SFR scales all rates linearly. With its bounded uniform prior,
we integrate it analytically and recover its truncated-Gamma conditional. The
remaining three parameters are integrated using uniform-prior Sobol points,
weighted by exp(log likelihood). The recent reference uses two independent
16384-point scrambles and checks their agreement. No GP predictions enter the
direct weights. This is numerical posterior quadrature, not exact integration
and not a full expensive MCMC run for every catalogue.

Rate maps at the same parameter points can be reused across catalogues. The
large direct-only ensemble amortizes that cost. GP-assisted amplitude recovery
and posterior predictive simulation also use direct rates in these studies.
Thus neither 482 training labels nor a short ensemble runtime measures the
end-to-end cost of an online GP-only analysis.

## Selection present in the tested code

`fixed_population.py` calls `FindDetectionProbability` and includes its result,
comoving shell volumes, and source-to-observer time conversion in the operator.
`ratesSampler.py:from_compas_h5` constructs an O3 sensitivity selection model.
The fixed operator must be rebuilt if selection, cosmology or the population
changes. The current tests use a single selection model/exposure, not an
observing-epoch mixture.

For perfect measurements of detected sources, the Poisson-bin likelihood uses
detected rates, including the expected detected count. This is consistent with
the tested generative model. In the new synthetic measurement experiment,

    lambda_measured[y] = sum_x R[y,x | detected] lambda_detected[x]

and each column of R sums to one. Detection has already happened; the response
is explicitly conditional on it. This preserves the expected count and is a
consistent, deliberately simplified uncertain-observation likelihood.

## What has not been established for LVK posterior samples

For ordinary strain-data likelihoods, the standard Poisson-process expression is

    L(theta) proportional to exp(-mu_detected(theta))
        product_i integral lambda_intrinsic(x | theta) p(d_i | x) dx.

Selection enters the expected count. One cannot substitute detected intensity
for intrinsic intensity in the event integrals unless the event likelihood is
also expressed consistently conditional on detection. Ordinary PE posterior
samples require division by the PE prior in the correct parameter measure,
including coordinate Jacobians. Multiplying a standard event likelihood by an
extra detection probability is generally not the same model.

The explicit `posterior_grid_coefficients` helper handles inverse PE-prior and
bin-area factors, but explicitly does not handle coordinate transforms or
selection. `LVKObservation` itself is an empty subclass. The legacy
`core_ln_likelihood` consumes supplied weights against a detected rate map; it
does not establish their physical likelihood/selection convention. Per-event
normalization is only a parameter-independent constant when those coefficients
are already correct; it does not perform missing prior or selection corrections.
No claim of a fully validated LVK ingestion/selection pipeline follows from the
mock tests. No legacy LVK behavior was changed in this audit.

Methodological reference: [Mandel, Farr & Gair (2019)](https://arxiv.org/abs/1809.02063).

## Framing the contribution

Demonstrate that GP-guided acquisition can select informative parameter
evaluations and reproduce the posterior at a controlled training-label budget.
Show comparisons with space-filling acquisition at matched budgets. Distinguish
this measured post-processing experiment from the proposed extension to choosing
new population-synthesis runs, where simulation cost, stochastic realizations
and additional evolutionary parameters would require separate validation.

Compare costs to the accelerated direct baseline, including fitting/acquisition
and setup, before claiming actual speedup. An accuracy benchmark can use many
direct evaluations offline without giving their labels to the acquisition rule.
