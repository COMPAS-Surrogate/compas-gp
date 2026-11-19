# BO acquisition and GP uncertainty pilot

2026-09-08. The grid is frozen: additional grid comparisons are not required for
this paper's conditional binned-model claim. User priority is now posterior error
versus acquired likelihood points and a matched mean/uncertainty comparison.

## Define the two targets before sampling

For a GP on raw log likelihood, f(theta) ~ Normal(mu(theta), s(theta)^2), compare
p_mean proportional to prior * exp(mu), and p_expected proportional to prior *
E_GP[exp(f)] = prior * exp(mu+s^2/2). Zero variance means the deterministic limit.
Both are deterministic posterior targets. Drawing a fresh random log likelihood
on every ordinary MCMC call is not this calculation; an appropriate augmented
or pseudo-marginal method would require a separately justified algorithm.

This is the expected-likelihood marginal approximation, as distinguished from
random surrogate posteriors in [Stuart & Teckentrup, section 4.2](https://arxiv.org/html/1603.02004v3).
Normalizing the expected unnormalized likelihood is generally different from
averaging separately normalized random posterior densities. The latter needs
coherent GP function draws and their cross-parameter covariance. We test the
former here. Neither construction automatically guarantees conservative intervals.

Our fitted Gaussian lives in transformed space: F ~ Normal(m,v), with
raw lnL = R - max(scale*F,0)^2 and R = reference + median. Therefore raw lnL is
not Gaussian and applying a delta-method sigma^2/2 correction is not exact.
For M=scale*m, V=scale^2*v, the actual expectation is

```
E[exp(lnL)] / exp(R)
  = Phi(-M/sqrt(V))
    + exp(-M^2/(1+2V)) / sqrt(1+2V)
      * Phi(M / sqrt(V*(1+2V))).
```

The V=0 branch is exactly exp(-max(M,0)^2). The implementation evaluates this in
log space, uses latent GP variance (not observation noise), and is tested against
independent rescaled numerical integration. It is a study helper only; the
production sampler's rejection of invalid nonlinear `marginal` remains intact.

## Matched design

Use the first three catalogues from the saved 100-catalogue ensemble, selected
by index before examining BO outcomes: 43, 95 and 6 events. Generation/inference
populations remain the disjoint 256M halves. Fixed grid, prior and selection.

Three training seeds (17,29,43), each beginning with 128 full-prior Sobol points
plus 98 boundary anchors. Four batches add 128 points each: 64 sampled using
expected improvement for minimization of the transformed target, 64 using latent
variance; each probability distribution includes 10% uniform exploration. Within
each batch the fit is frozen; candidates are chosen without replacement and all
old points are retained. This is batch BO/active learning, not 128 sequential
refits and not the previously validated 512+128 training recipe.

Both posterior targets use exactly the same fit at 226,354,482,610,738 total
training points. This isolates propagation of uncertainty from acquisition.
Acquisition itself is common to both targets; we have not optimized a separate
acquisition policy for each posterior interpretation.

At every checkpoint, compare with direct amplitude-marginalized likelihood on
two independent 8192-point full-prior Sobol grids. Compute both KL directions in
nats using normalized quadrature weights, plus shape-parameter widths/means and
latent uncertainty calibration. The KL is for the three-dimensional marginal.
It is also the four-dimensional joint KL if both posteriors use the same exact
conditional amplitude distribution. It does not assess an approximate amplitude
surrogate or rely on histogram estimates of multidimensional KL.

Candidate rates/labels were cached for efficiency, but unacquired labels are not
used in fitting or acquisition. The truth parameters never guide acquisition.
Total cache generation must be charged in any actual cost claim. The horizontal
axis counts supplied training labels, including anchors, not every offline cache
evaluation or the direct-reference validation calls.

## Result

Median forward KL over the three catalogues, three seeds and two numerical grids:

| Total training points | Mean target | Expected-likelihood target |
| --- | --- | --- |
| 226 | 0.04931 | 0.04742 |
| 354 | 0.02103 | 0.02069 |
| 482 | 0.00986 | 0.00966 |
| 610 | 0.00470 | 0.00464 |
| 738 | 0.00243 | 0.00238 |

At 738 points, forward KL ranges 0.000180–0.004640 for mean, and
0.000179–0.004573 for expected likelihood. Reverse KL shows the same trend.
Maximum forward-KL difference between the two numerical scrambles is 0.000129
at the final checkpoint; numerical scrambles are not extra independent training
runs. Final shape 68% widths are within 2.4% of direct on the common grids.
These common-grid interval figures are exploratory, not a new precision gate.

Including GP variance makes a small difference; it is not consistently better
for every trajectory. Additional training points produce the dominant reduction
in error. These curves do not establish an advantage of BO over a matched
space-filling design: no such baseline has yet been run.

Final posterior-weighted latent one-standard-deviation coverage ranges
0.625–0.803, median 0.738. This is pointwise transformed-function coverage, not
parameter credible-interval coverage or proof that the GP process is calibrated.
Up to 0.39% of direct posterior mass is above the training reference, where the
inverse transform caps raw lnL. That limitation remains visible rather than being
fixed by silently altering the transform.

## Reproduction and evidence

```
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/bo_uncertainty_curve.py
MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/joint_pilot/plot_bo_uncertainty.py
```

`bo_uncertainty_results/curves.json` contains all 180 checkpoint records;
per-trajectory NPZ files preserve training locations/labels and acquired indices.
`summary.json` records the script hash and timing. `kl_curves.png`/`.pdf` shows
seed trajectories and medians, after averaging the two scrambles within a seed.
Main runtime: 118.0 s including 71.6 s cache generation, excluding imports.
51 fast tests pass (3 deselected, one existing Matplotlib deprecation warning).
No production defaults changed; no commit/push or new COMPAS simulations.

## Next bounded comparison

Add a matched space-filling acquisition baseline at the same budgets and seeds
before attributing the decreasing KL specifically to BO. Keep the mean versus
expected-likelihood contrast paired. Expand to more simulated catalogues only if
the paper needs a stronger uncertainty-propagation conclusion. Do not add LVK
systematics or more grid comparisons to this experiment's critical path.


Completed follow-up: [matched space-filling baseline](ACQUISITION_COMPARISON.md). The tested BO policy shows no clear overall advantage across the nine paired trajectories.
