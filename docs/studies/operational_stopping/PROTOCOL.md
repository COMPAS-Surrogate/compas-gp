# Frozen v1 operational-stop validation

The operational learner must make its decision without reference coordinates,
reference scores, true mode locations or family labels. It receives a counted
point-evaluation callable and the unit-square prior. Its immutable decision is
written before constructing any reference. External scoring verifies its hash.

## Scope and design

Two-dimensional deterministic toy likelihoods: correlated Gaussian, curved
banana, separated two-mode mixture, narrow Gaussian, and a broad mode plus a
narrow secondary mode containing 30% of the untruncated probability. Centres
and orientations vary with seed. Box truncation is normalized by quadrature.
This tests interpolation and stopping, not noisy generative calibration or
COMPAS finite-population adequacy. The hidden-mode family is a deliberate
failure-mode test, not a guarantee that a finite audit can discover every mode.

- Development smoke: correlated seed1000; inspect execution and gate behaviour.
- Frozen confirmation: all five families, seeds2000--2004 (25 runs). No tuning
  based on their reference scores. Retain unsuccessful stops and resource limits.
- Same square-root target transform, all training points, Matérn-5/2 exact GP,
  fixed noise and 250 fitting steps as the current pipeline. These toys are 2D.
- Initialize at 32 scrambled Sobol plus 16 boundary points. Add batches of 32
  from 2048 cheap Sobol candidate coordinates, weighted 90% by GP likelihood
  variance and 10% uniformly. Only selected coordinates receive exact labels.
- Two separate 16384-coordinate scrambled Sobol quadratures sample the GP
  posterior. These evaluate only the surrogate and are independent of truth.
  Require numerical agreement and effective sample size >=100 in each.
- Prediction check: compare each new batch with predictions saved before
  refitting, in original log-likelihood units. Remove the likelihood-weighted
  median residual (a constant offset), then scale absolute residuals by
  0.1 + 0.01*(best new log likelihood - local new log likelihood). Require
  90th percentile <=1 and maximum <=3. All data are still used for training.
- Stability: maximum of both directed successive-posterior KLs <0.001;
  mean/interval-endpoint changes <0.05 preceding SD, 68% widths change <3%.
  Both GP-only quadratures must agree to these marginal tolerances.
- Require prediction, stability and numerical checks for two consecutive fits.
  Then audit 64 fresh GP-posterior coordinates and 16 independent uniform-prior
  coordinates. For posterior residuals after median-offset removal, require
  RMS <=0.1, 95th percentile absolute error <=0.2 and normalized importance ESS
  >=0.95. A broad point relevant under either predicted or exact likelihood
  (within 12 lnL of the best training value) must have residual <=0.3.
- Score audits before training on them. Accepting an audit stops on the current
  GP. Rejected audits are added to training and reset the consecutive count.
  Count every initialization, acquisition and audit label. The guard of 800
  calls bounds this experiment; reaching it is unresolved, never convergence.

## Hidden reference and reporting

After the durable operational decision, evaluate the exact toy and each saved
GP on two independent 131072-coordinate reference quadratures. Report maximum
directed KL on both, marginal changes, numerical checks, and the first checkpoint
meeting the external target. The target is both KLs <0.01, means/endpoints
within 0.2 direct SD and 68% widths within 10%. This is an offline accuracy
assessment, not input to the stopping rule. Numerical failures are unresolved
scores, not evidence of either a correct or a premature stop.

Publish all trajectories with operational stop markers and the hidden-reference
KL revealed afterwards. Summarize premature stops, valid stops, unresolved runs,
evaluation requirements including audits, and any delay relative to reference
accuracy. Report small denominators explicitly; 25 examples cannot establish a
universal or precise rare-failure guarantee. No threshold retuning on confirmation.

The rule is inspired by pre-refit prediction checks (GPry section4.3), with our
own offset treatment, batch thresholds and fresh audit. It is not GPry itself.
https://arxiv.org/pdf/2211.02045
