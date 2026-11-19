# COMPAS experiment: GPry-style acquisition and online stopping

Implement the mathematical ideas from arXiv:2211.02045 in the existing JAX
runtime; do not install or vendor GPry. This is an adaptation to our existing
Matérn-5/2 GP and square-root-compressed LnL, not a reproduction of GPry's SVM
and RBF model. Keep all training points and the full scientific prior.

## Frozen scientific choices

- The fixed COMPAS blocks 0–7 (256M), mass10 output grid, original parameter
  bounds, and analytic amplitude marginalization are unchanged.
- Four regimes: expected counts 100 and 1000, each perfect and uncertain
  measurements. Three fresh catalogues per regime. Perfect/uncertain cases
  share their underlying mock catalogue. Fixed injection, not prior-drawn SBC.
- Catalogue seed family 2026091101; BO seeds 5011, 5012, 5013. A separate seed
  9011 smoke is excluded. Initial design: 32 scrambled Sobol points plus the 26
  boundary locations of a 3×3×3 lattice (58 labels).
- Three-label Kriging-believer batches. GP hyperparameters and target scaling
  refit after every 30 new labels (250 Adam steps, fixed noise), with exact
  conditioning after every batch. Fit in the original physical coordinates.
- Maximum 658 labels per run, including initialization. Record proposed stops
  but continue all paths to the same budget for validation and fair bands.
  Labels after each candidate stop are validation continuation costs, not
  required by that stopping decision. Budget exhaustion is not convergence.

## Acquisition and stopping

GPry Eq. 15 form: maximize 2*zeta*(plug-in LnL - reference) + log(expm1(s_LnL)),
with zeta = dimension^(-0.85). Obtain s_LnL from the analytic second/fourth
moments of the Gaussian transformed prediction under the existing clipped sqrt
inverse. This retains the nonlinear transform; it does not assume calibrated
GP uncertainty. Preserve GPJax's numerical jitter; add no observation-noise
draw to acquisition. Optimize with JAX gradients and bounded multistart L-BFGS-B,
seeded by a broad Sobol pool and training-centred proposals. Log numerical
fallbacks and exclude duplicate points. Temporary mean labels change conditional
covariance without changing the GP mean or fitting hyperparameters.

Monitor three candidate stops on the same trajectory:

1. Paper Eq. 16 CorrectCounter: abs(prediction-actual) < .0352674038 +
   .01*abs(best_before-prediction), four consecutive labels. Predictions and
   best_before are frozen before the whole batch is evaluated. Any failure
   resets the counter, including a later failure within that batch.
2. The counter plus Gaussian posterior KL <.02 on two consecutive, numerically
   resolved 30-label checkpoints. Require the counter still to pass at stop.
   Use maximum directed Gaussian KL, an explicitly conservative convention.
3. Existing joint-posterior stability: maximum directed joint KL <.001, means
   and 68%/95% endpoints change <.05 SD, 68% widths change <3%, twice, with
   resolved GP-only integration. Check every 30 labels for comparable windows.

GP-only integration uses two independent scrambled full-prior/Student-t mixture
designs (16384 base locations each). Refine the proposal once from GP moments.
ESS >=100 and agreement within the same summary tolerances are required for
both current and preceding GPs. No reference grid informs acquisition or stopping.

## Offline assessment and figure contract

Save decisions and their hashes before loading any reference arrays. A separate
process then uses existing direct COMPAS reference rates (two independently
scrambled importance designs, power16 for count100 and power17 for count1000).
Recompute catalogue-specific LnLs from those rates. These references are scoring
data only; record numerical precision failures rather than declaring accuracy.

Plot 0.5*(KL_direct_to_GP + KL_GP_to_direct), matching the paper's symmetric KL,
versus actual likelihood training/continuation labels. Use per-regime panels,
median curves and interquartile shading across the three runs, and individual
curves as a companion. Mark median candidate stopping counts and their spread;
show how many runs stopped. Do not impute a stop for a censored run or carry
its last value forward. All curves continue to the common budget; no smoothing.
The shaded band is between-run spread, not a confidence interval. Title-free
standalone scientific PNG/PDF plots use Matplotlib and are visually inspected.

Accuracy is scored separately: both directed KLs <.01, mean/endpoint shifts
<.2 SD, 68%/95% widths within10%, and resolved reference/GP integration. Report
accuracy at each proposed stop, late improvement, failures, and total runtime.
Do not call the fixed-injection experiment calibration or a rare-failure bound.

Sources: https://arxiv.org/html/2211.02045v2 (Sections4.1–4.3, AppendixA/B),
https://gpry.readthedocs.io/en/latest/module_convergence.html. Eq.16 in the paper
uses the prediction in its relative tolerance; current GPry documentation uses
the actual label. This experiment explicitly follows the paper's formula.
