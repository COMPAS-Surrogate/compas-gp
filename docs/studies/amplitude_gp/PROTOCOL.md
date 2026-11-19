# Explicit SFR amplitude pilot — 14 September 2026

Keep existing production settings, studies and manuscript unchanged. This is a
bounded validation experiment, not SBC or a new astrophysical result.

1. Conditional test: fixed shape, Poisson counts drawn at expected counts
   10, 100 and 1000 using a frozen seed. Fit 33-label one-dimensional GPs
   with training seeds 17, 29 and 43. Compare normalized posteriors against
   the exact bounded-uniform-prior truncated Gamma, including prior effects.
2. Joint test: reuse count-history catalogue case 200 at 100 and 1000 expected
   counts, in both perfect and uncertain views. Same 256M blocks0–7 population,
   mass10 grid, selection, full prior and measured events. Three seeds 17/29/43.
   Train amplitude as a fourth GP coordinate, with no analytic amplitude term
   supplied to the GP. Initial 64 Sobol points plus 80 boundary anchors; retain
   every label. Checkpoints 144, 272, 400 labels; no extension based on results.
3. Compare IID uniform continuation with a new explicitly specified batched
   acquisition pilot: each 128-point batch has 32 fresh uniform points and
   96 sampled from a fresh 4096-point Sobol pool using the existing GPry-form
   score (zeta=4**(-.85)). Sampling uses Gumbel top-k, without replacement.
   This tests this batched policy, not the historical three-member optimized
   acquisition, and does not establish an optimal 25/75 ratio.
4. Finish and freeze all training before loading cached reference rates. Score
   the final checkpoint on two independent existing full-support shape designs,
   with an independent 50/50 conditional-Gamma/uniform amplitude proposal.
   Underflowing Gamma intervals use uniform proposals with the correct density.
   Check direct reference parity against the existing marginalized-plus-exact-
   conditional reference and agreement of both integrations for direct and GP.
5. Accuracy gates: both directed KLs < .01; mean and interval endpoint shifts
   < .2 direct posterior SD; 68% and 95% width changes <10%. Numerical gates:
   independent mean/endpoint shifts < .05 SD and 68% widths <3%, ESS >=100.
   Any unresolved integration is reported as unresolved, never as a pass.
   Also retain posterior-weighted centered log-likelihood RMSE and correlations.

Report counts, failures, actual elapsed times and source/input hashes. Rates
are evaluated only at selected training coordinates. Reference caches are
offline scoring work; this is not an end-to-end speedup measurement. Stop at
this bounded pilot and assess evidence before broad calibration, grid changes,
or further campaigns. No commits, pushes or manuscript replacement.
