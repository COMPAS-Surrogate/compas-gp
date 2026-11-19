# Model notation and implementation audit

The active manuscript starts with Bayes' theorem in `overleaf/sections/bayesian_model.tex`.

| Symbol | Meaning | Implemented quantity |
| --- | --- | --- |
| lambda = (alpha, sigma, a_SFR, d_SFR) | Population hyperparameters | Arguments to FixedPopulationRates.__call__ |
| theta | Individual-binary parameters, including chirp mass, mass ratio and redshift | COMPAS masses, delay times and detection integration |
| D = {n_b} | Simulated observed catalogue counts | catalogues.npz; not LVK posterior samples |
| pi(lambda) | Independent uniform hyperprior | LOW and WIDTH in run_study.py |
| r_b(lambda) | Detected rate per observer year | FixedPopulationRates contraction, including detection, volume and time dilation |
| nu_b = T r_b | Expected observed count | Rate rows scaled to the experiment duration |
| ell(lambda) | Binned Poisson log likelihood | sum n log nu - nu, omitting data-only constants |
| R_yb | Measurement response conditional on detection | binned_response.py; column-normalized |
| phi = (alpha, sigma, d_SFR) | Shape coordinates modeled by GP | SHAPE = [0,1,3] |
| ell_m(phi) | Log amplitude-marginalized likelihood | vector_marginal |
| p(a | phi,D) | Conditional amplitude posterior | Truncated Gamma, same for GP and reference |

Metallicity uses natural logarithms: FindZdistribution computes log Z and sigma,
with fixed zero skewness and no redshift dependence of sigma in this experiment;
the finite log-metallicity grid is normalized numerically. The SFR law and four
variable parameters follow Riley and Mandel (2023), building on Neijssel et al.
(2019). Cosmic integration reweights an existing binary population; changing
lambda does not run new stellar evolution.

Sources reviewed:
- https://arxiv.org/html/1809.02293v9 : Bayes notation, posterior recycling and selection caveats.
- https://arxiv.org/pdf/1906.08136 : metallicity-specific star formation and delay-time integration.
- https://arxiv.org/pdf/2303.00508 : equations 1-5 and four-parameter post-processing setup.

The event-posterior likelihood in the appendix is explanatory, not executed in
this study. Event posterior samples must be divided by their parameter-estimation
prior, with consistent coordinates and selection convention. Do not identify
posterior samples with raw data or insert them directly into our count likelihood.
