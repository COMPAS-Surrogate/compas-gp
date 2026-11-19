# Periodic posterior-audit development experiment

This is a new candidate, motivated by the three known non-stops in extension_v2.
All 24 likelihood cases (48 BO/random trajectories) are development evidence
because their reference results have already been examined. Fresh audit draws
are independent of training, but do not turn known catalogues into held-out
confirmation. Preserve the old frozen files and outcomes.

Use the existing checkpoints at 238, 298, 358, 418, 478, 538, 598 and 658 training
labels. Always attempt the final checkpoint. Neither the prediction counter,
successive GP stability, nor pointwise RMS/q95 can veto an audit or acceptance.
They remain diagnostic fields. Audit proposal normalizers must still be
numerically resolved using two independent GP-only importance designs, with
maximum log-normalizer difference <0.02 and GP posterior summaries satisfying
the established numerical agreement checks. Refine powers 14, 15, 16 as needed.
If these fail, record a numerical failure and move to the next checkpoint.

Draw a fresh 4:1:1 mixture of the numerical GP posterior, its temperature-two
version, and IID uniform-prior points. Use 96 total points initially. If the
result is inconclusive, add independent samples at the SAME frozen GP to reach
192, then 384. All increments keep the same mixture and normalizers. Persist
coordinates, GP values and mixture densities before evaluating exact likelihoods.
The rule reads no reference grid, reference scores, or injected parameters.

Acceptance requires maximum directed KL <0.003, 95th stratified-bootstrap
percentile <0.005, and GP/direct importance ESS >=32. Use 256 stratified bootstrap
replicates. If acceptance fails, enlarge while below 384 if ESS is inadequate or
the 5th bootstrap percentile is below 0.003. Otherwise continue training. RMS
and q95 are recorded but cannot veto acceptance. Finite bootstrap percentiles
are not rigorous sequential bounds or guarantees of unseen-mode discovery.

Replay both a fixed-96 audit and the adaptive-384 audit on shared initial draws,
recording costs for each policy separately. Audits are held out of the saved
training trajectories. This isolates the scheduling/acceptance repair; feeding
failed audits into subsequent GP training is not tested in this replay. Continue
each policy to first acceptance or the 658-label cutoff. Budget exhaustion is
not successful convergence. Numerical failure uses zero exact audit calls.

Audit RNG seed: SeedSequence [202609121, injection index, expected count,
catalogue index, uncertain flag, random flag, checkpoint]. Bootstrap seed:
202609122 + checkpoint. Freeze this protocol and implementation before running
new audits. No threshold changes based on the results; retain failed candidates.

After all decisions are frozen, score against cached direct-rate references.
Reuse existing scores wherever available; at new stopping checkpoints compute
all four physical marginal checks as needed. Keep both directed KLs <0.01,
mean/endpoint errors <0.2 SD and 68/95% width errors <10%, with reference and GP
numerical agreement. Use existing rate caches with fresh selected rates computed
on demand. The physical model is four dimensional, with amplitude marginalized
analytically and a three-input GP. No GPry installation is needed.

First run the three formerly non-stopping BO cases as an execution pilot, then
complete all 48 trajectories without altering policy. Report successes, premature
acceptances, unresolved cases, non-stops and total costs. Update standalone KL
figures to show the new measured audit statistics and stopping locations.
