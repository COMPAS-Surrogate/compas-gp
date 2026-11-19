# Paper completion — 2026-09-11

1. Finish and publish the existing 600-catalogue campaign without changing its
   frozen inference settings. Keep resource limits and numerical failures.
2. Fresh fixed-96 stopping confirmation: see gpry_compas/CONFIRMATION_PROTOCOL.md.
3. Independent reference-only refinement of the original double-width failure
   and high-count ensemble reference; preserve originals and bounded failures.
4. Main figures: physical rate response, acquisition efficiency, paired
   perfect/uncertain accuracy, physical information versus measurement width,
   population-size sensitivity, confirmed stopping accuracy versus total cost.
5. Compact BO/random/Sobol robustness and continuous posterior prediction in
   appendix. Remove redundant initialization, categorical predictive and
   stopping-development figures from paper while retaining study archives.

Population-size figure protocol: three fresh Poisson catalogues at each of
50, 300 and 1000 expected detections, generated with the full 512M population
at the paper's fiducial injection. Nominal continuous measurement model.
Infer each identical catalogue with nested 32M, 128M, 256M and 512M populations,
using unchanged mass/redshift grids, selection and priors. Compare direct
four-parameter posteriors; no GP approximation enters. Check two independent
full-support importance integrations; initial 2^15, bounded refinement 2^17.
Report maximum absolute parameter mean shift in 512M posterior SD and maximum
68% width change across parameters. Retain numerical/support failures. These
are paired finite-population sensitivity measurements relative to 512M, not
absolute bias, independent-population variance, or a safe catalogue-size limit.
RNG SeedSequence [202609214, expected_events, case], cases 0--2.
