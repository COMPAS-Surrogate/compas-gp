# Four-input GP follow-up — frozen before training

This follow-up retains the failed large-batch pilot and changes the acquisition
schedule explicitly. It is not a new-catalogue confirmation, SBC or NUTS test.

- Same two latent catalogues, 100/1000 expected detections, perfect/uncertain
  views, initial seeds 17/29/43 and initial 144 labels from pilot_20260914.
- 24 fresh continuations: 12 BO and 12 IID random. Acquire three labels per
  batch, condition all labels immediately, refit GP hyperparameters/transform
  every 30 labels. Use the historical optimized GPry-form acquisition with
  dimension 4, fantasy covariances, full-prior Sobol candidates and three local
  optimization starts. No arbitrary 25/75 split is used in this follow-up.
- Retain the entire scientific prior and every tail point. No analytic amplitude
  term enters GP predictions. Matérn-5/2, sqrt transformation, fixed-noise fitting
  and 250 optimizer steps remain unchanged. Fixed conditioning capacity 768.
- Save checkpoints at 144, 414 and 744 labels. 744 is the fixed training cap;
  no stopping decision or reference-driven expansion. Save pre-training batch
  predictions to assess prediction quality without recycling training residuals.
- Finish each trajectory before independent scoring. The scorer never alters
  training points. Reuse the two validated shape-reference designs and exact
  rates. Replace the original one-random-amplitude-per-shape scoring with 16
  stratified amplitude draws per shape, balanced equally between uniform and
  conditional truncated Gamma proposals. Use their exact mixture density.
  For unresolved results, one bounded scoring-only refinement uses 64 draws.
- Direct/GP numerical gates, joint KL and four-parameter accuracy tolerances
  remain those of PROTOCOL.md. Also require the two directed KL estimates to
  agree between scrambles within max(0.001 nats, 10% of their magnitude).
  Preserve initial and refined outcomes and all numerical failures.
- Score 414 and 744 labels. The former is a near-budget diagnostic relative to
  the historical 400 labels, not an exactly matched budget claim. Compare BO
  and random at identical budgets; test the larger budget within each trajectory.
- Output source snapshots/hashes, timing components, checkpoints, independent
  numerical checks, all outcomes and compact plots/report. No production or
  manuscript change, commit or remote write.

Run one first trajectory to establish implementation/timing. Then complete the
declared matrix with unchanged settings. Scientific failures are outcomes, not
reasons to drop jobs. This pilot cannot certify global mode recovery.
