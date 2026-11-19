# Fresh fixed-96 stopping confirmation — frozen 2026-09-11

Use the unchanged compare_random likelihood, population, prior, GP, acquisition,
58-point initialization, three-point batches, refit/stability schedule and
658-training-label resource limit. Use case identifiers 100 and 101 in the
existing deterministic catalogue and RNG functions: these produce new latent
catalogues and new initialization/acquisition/audit draws, distinct from all
case 0/1 development data. Three injections, 100/1000 expected events, paired
perfect/uncertain measurements, two cases and BO/random give 48 trajectories.
These are 12 independent latent catalogues and 24 paired likelihood cases.

Freeze the stable_scheduled_v2 fixed-96 rule: audits from 238 training labels,
every 60 labels and at the final checkpoint, 64 posterior +16 tempered +16
prior draws; unchanged KL, ESS, bootstrap and joint-stability conditions. Only
96-point audits are used. Existing training-only diagnostic monitors are saved
by the reused runner but do not define the confirmation endpoint. Their audit
calls, if any, are comparator overhead and excluded from the fixed-96 policy
budget; all actual computation is separately retained in timing records.

Save all trajectories and audit decisions before accessing independent reference
scores. Score the accepted stop and final checkpoint with the existing two
independent cached reference designs, refining only direct numerical failures
through a bounded 2^18 points per design if needed. Test all four parameter
marginals and both directed KLs. Unresolved references remain unresolved;
no-stop cases remain censored. No thresholds, seeds or cases may be changed
based on confirmation outcomes. No universal stopping or calibration claim.
