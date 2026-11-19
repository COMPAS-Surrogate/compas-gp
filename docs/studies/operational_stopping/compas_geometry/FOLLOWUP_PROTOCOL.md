# Later audit trigger: COMPAS development follow-up

After examining the ten-case first screen, repeat the unchanged 50-label audit
at the first saved `posterior_stability` decision, instead of the earlier
Gaussian-moment stability decision. This is a development comparison on the same
catalogues, not independent confirmation. Do not tune any audit tolerance.

Use only the selected checkpoint to refit and select fresh audit locations.
Save all decisions before attaching the already available reference scores.
The stricter trigger still requires two consecutive numerically resolved GP-only
posterior comparisons. Count 50 new labels per case; the profile-only design
would use 14 of these locations. Audits do not modify the saved training path.

The purpose is to measure whether a later stability trigger reduces unnecessary
audit rejection. It does not measure the outcome of retraining after rejection,
the probability of missing another mode, or online stopping performance.
