# Posterior-mass audit variant, frozen before direct-reference scoring

The pointwise audit refused stops on the first injection because some broad
points had large absolute residuals at low likelihood, even when its posterior
RMS was small. This motivated the following fixed variant. Injection index 0
is development evidence. Indices 1 and 2 (16 paired likelihood cases, 32
trajectories) are confirmation for this variant. No direct-reference score
has been inspected when defining it. This does not erase the original frozen
pointwise-audit experiment or its rejected stops.

Use the same counter/Gaussian/joint-stability eligibility and retry schedule as
the pointwise audit, on its unchanged training trajectory. Draw an independent
96-point audit with 64 numerical GP-posterior draws, 16 temperature-two draws
and 16 IID uniform prior draws. Use the actual mixture fractions 2/3, 1/6, 1/6
and GP-only normalizing integrals to evaluate its mixture density. The two
independent GP normalizers must agree within .02 log units; increase GP-only
integration resolution through 2^16 if necessary. The evidence estimate divides
by generated proposal count, including rejected outside-prior points.

Keep posterior-only RMS <=.07, 95th residual percentile <=.15 and correction
ESS fraction >=.99. Replace the pointwise broad veto by self-normalized
importance estimates of both directed KLs on the entire audit mixture. Require
their maximum <.003 and its 95th bootstrap percentile <.005 (256 stratified
resamples preserving component counts), plus corrected importance ESS >=32.
The bootstrap describes the finite observed audit; it is not a rigorous
sequential confidence guarantee or a bound on undiscovered modes.

Audit coordinates, GP predictions and proposal densities are frozen before
exact likelihood calls. Audit RNG SeedSequence [202609115, injection, expected
count, case, measurement-is-uncertain, method-is-random]; bootstrap seed is
202609116 + training count. Retry after >=60 additional training labels,
at most four audit attempts. Audits stay held out. Count 96 exact calls per
attempt; no costs from the alternative pointwise audit are charged to this
separate rule. Freeze and hash all mass-audit decisions before revealing any
direct-reference score. Independent scoring checks the original and mass-audit
decisions without changing either. Report development and confirmation apart.
