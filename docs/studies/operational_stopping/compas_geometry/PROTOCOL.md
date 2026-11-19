# COMPAS-focused stopping and autodiff checks

This supersedes the generic shape benchmark as the focus of ongoing stopping
work. All scientific test cases here use the fixed COMPAS population, detected
mass/redshift rate grid, and catalogue likelihood already used by this project.
No new arbitrary target shapes or training-set subsetting are introduced.

Preselected cases: ensemble catalogues 0, 1, 2 at expected counts 30, 100, and
1000 with perfect measurements; the separate pilot catalogue at expected count
500 with uncertain measurements. These ten saved BO paths are reference-assisted
and may end early. New decisions use only their training prefixes and GP-only
calculations. Their references are accessed after saving decisions for scoring.
This is a COMPAS checkpoint-screening experiment, not an independently executed
new adaptive campaign or a calibrated rare-failure-rate measurement.

GP fits preserve the source's physical coordinates, Matérn-5/2 kernel, square-root
transform, fixed observation noise, and all training labels. The amplitude is
analytically marginalized, leaving alpha, sigma and SFR exponent for this study.

Candidates:

1. GPry-inspired CorrectCounter: all labels in each new batch must agree with
   the previous GP within 0.1 + 0.01*(best observed LnL - actual LnL), twice.
   These are explicit experimental thresholds, not GPry's dimensional defaults.
2. CorrectCounter plus Gaussian KL <0.02 between successive GP posteriors,
   with two consecutive passes and a resolved GP-only integration check.
3. Posterior stability: both directed joint KLs <0.001, means and interval
   endpoints change <0.05 SD, 68% widths change <3%, twice.
4. At the first stable posterior trigger, test the current GP with a new local
   derivative audit and a direct evaluation at its gradient-optimized maximum.
   Also monitor whether GP score and curvature changes are <0.1 in posterior
   SD coordinates. Derivative stability is assessed as a potential additional
   condition, not presumed to establish correctness.

Use two independent GP-only importance designs, each with 8192 base coordinates:
25% uniform-prior plus 75% broad Student-t, centered using training labels only.
The proposal does not alter the prior. No true LnL grid is evaluated. Numerical
agreement of both GP designs is required; unresolved checks cannot trigger a stop.

JAX autodiff supplies derivatives of the inverse-transformed GP mean. Cache the
GP's linear solve once per fit. Score changes are whitened by the GP posterior
covariance; curvature differences use C^(1/2)^T Delta H C^(1/2). The current exact
COMPAS evaluator is NumPy/SciPy, so its derivatives are checked with fresh exact
labels at matched finite-difference coordinates, without porting its physics.

The audit evaluates the full gradient and mixed Hessian near the GP posterior
mean, at step sizes 0.25 and 0.125 posterior SD. In three dimensions this needs
37 points; one additional exact evaluation checks the GP maximum, located by
bounded L-BFGS-B using autodiff gradients and multiple GP-only starts. Compare
the exact-minus-GP residual's derivatives on identical stencils: slope norm
<0.1, Hessian spectral norm <0.1, and refinement changes <0.03 / <0.05.
Require mode-relative residual <0.2 LnL. Check the prior boundaries explicitly;
do not clip audit coordinates into a different design. Costs include all 38
new labels. A rejected audit is a rejected checkpoint, not a retrained solution.

The score tolerance corresponds approximately to <0.1 SD displacement and the
curvature tolerance to roughly 5% width error for a locally quadratic posterior.
Those interpretations are local approximations to test against the COMPAS
results, not assumptions that all posteriors are Gaussian. Intervals and joint
posterior stability are retained independently.

Sources: https://gpry.readthedocs.io/en/latest/module_convergence.html and
https://gpry.readthedocs.io/en/latest/module_gp_acquisition.html (CorrectCounter,
GaussianKL, training-mode alignment and gradient-based acquisition optimization);
https://docs.jax.dev/en/latest/higher-order.html (higher-order autodiff).

## Development adjustment after the single-case smoke

The smoke resolved the code path but showed numerical precision failures in the
training-centered GP integration proposal. Refine that proposal once using
GP-only mean/covariance estimates, then use two fresh 16384-coordinate designs.
Save precision diagnostics and ESS explicitly. This increases only surrogate
evaluation work; it adds no exact likelihood grid or prior restriction.

The smoke also showed that changes of local GP derivatives can be appreciable
even when integrated posterior changes are small. The comparison therefore
retains the derivative audit as one candidate and adds a posterior-scale
prediction check as another. Trigger both at two numerically resolved Gaussian
KL <0.02 updates (the GPry-style moment threshold). In addition to the original
38 points, evaluate twelve points at +/-1 and +/-2 posterior SD along the
covariance factor columns. Rays reaching a prior boundary stop at 99% of the
available distance. Score their errors and the optimized-mode error relative
to the shared centre, with tolerance 0.1+0.01*(best audit LnL-actual LnL).

This matched experiment costs 50 labels per candidate, of which a standalone
mode/profile audit needs only 14 (centre, mode, twelve profile points). Report
the actual 50-label cost in the experiment and the 14-label design separately.
No threshold was tuned against a reference score. All source paths, including
unresolved or rejected cases, remain in the report. The derivative and profile
audits use the same frozen candidate, rather than selecting whichever agrees
with the reference after seeing it.
