# Frozen COMPAS stopping and random comparison, 2026-09-11

This extension tests generalization; it cannot certify all COMPAS cases. The
previous twelve trajectories are development evidence and are excluded here.

## Experiment

Three injections in (alpha, sigma, a, d): (-.3,.35,.01,4.7),
(-.46,.15,.006,4.35), (-.12,.52,.013,5.05). At each injection use exposures
giving 100 and 1000 expected events, two independent Poisson catalogues, and
paired perfect/continuous uncertain measurements. Thus 24 catalogues and 48
BO/random trajectories. Catalogue SeedSequence is [202609112, truth index,
expected count, case]. Initialization/acquisition seeds are 6011 and 6012.
Do not tune rules or remove cases after examining extension reference scores.

The fixed 256M population, mass10 output bins, full prior, sqrt transform,
Matérn-5/2 GP, 250 optimization steps and exact amplitude marginalization are
unchanged. Fit all training points. Both acquisition methods use the same 58
initial coordinates and labels, condition every three new labels, refit every
30, and continue through 658 training labels. Random means IID uniform prior
draws after initialization, not MCMC. BO uses the preceding continuous GPry
acquisition with JAX gradients and three-point Kriging-believer batches.
Only selected coordinates request exact rates. No parameter grid is required;
existing exact rate caches may be reused for independent validation.

## Frozen stopping monitors

1. Original four-correct counter plus two Gaussian-KL <.02 checkpoint passes.
   Retain the original fixed 2^14 integration monitor as a comparator.
2. Precision-refined counter plus Gaussian stability. At each checkpoint,
   increase two independent GP-only integration designs from 2^14 through
   2^16 points if numerical agreement fails. This uses no exact labels.
3. Guarded stop: refined counter/Gaussian condition, two joint-stability
   passes (maximum directed KL <.001 and stable moments/intervals), followed
   by a fresh 96-label audit before accepting the current GP. Draw 64 points
   from its numerical posterior, 16 from its tempered posterior (temperature
   two), and 16 IID uniform prior points. Audit RNGs are separate from training.
   Remove a constant residual offset; require posterior RMS <=.07 lnL,
   95th absolute residual percentile <=.15, correction-weight ESS fraction
   >=.99, and maximum relevant broad-point residual <=.3. A broad point is
   relevant when exact or offset-corrected predicted lnL exceeds the best
   training label minus 12. These finite audits do not certify mode discovery.
   Retry no earlier than 60 additional training labels, at most four audits.

Audits remain held out and never influence the shared training trajectory.
This isolates acquisition from validation and permits paired training-budget
comparison. All audit values are saved, including failures. Count their 96
exact calls per attempt in operational stopping costs. Training curves exclude
held-out audit calls and are labelled accordingly. No speedup is inferred from
training counts. Original and improved monitors run on the same trajectory.
All decisions and checkpoints are hashed before revealing direct references.

## Independent accuracy and four-dimensional reconstruction

After decisions are frozen, use existing exact-rate importance designs near
each injection, with two independent scrambles and their actual proposal
densities. Recompute the new catalogue likelihoods. Numerical reference and
GP precision gates are unchanged: mean/endpoints <.05 SD, 68% widths <3%,
ESS >=100. Refine references alone if they fail; never acquire training points
from reference information. Both directed KLs must be <.01; all four marginal
mean/endpoints <.2 SD and 68/95% widths within 10% at claimed stops and final
fits. Joint four-dimensional KL equals shape KL when the same exact amplitude
conditional is used. Compute amplitude conditional moments/CDF analytically,
using reference rates only for scoring. This reconstruction still requires
on-demand total rates at posterior points in a grid-free application; its cost
is additional to training. A GP over four explicit input coordinates has not
been tested by this experiment.

## Plots and manuscript

Question: does BO reach a given posterior accuracy with fewer training labels,
and does the guarded rule reject inaccurate stable fits? Matplotlib PDF/PNG
line plots have 21 checkpoints per trajectory: blue solid BO and orange dashed
random, median and descriptive interquartile bands, no smoothing. Show all
individual trajectories separately. A paired stopping-cost dot plot retains
failures and censored runs. Tables provide exact counts. Figures are title-free,
with inward mirrored ticks on all four spines; panel definitions and scope go
in the manuscript captions. Preserve earlier benchmark results and clearly
identify the changed acquisition and stopping procedures.
