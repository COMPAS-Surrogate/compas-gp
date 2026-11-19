# Periodic audit with posterior stability: second development candidate

The audit-only candidate scheduled_v1 stopped all 24 BO trajectories but had
one premature fixed-96 acceptance and two premature adaptive-384 acceptances.
Both problematic checkpoints had zero consecutive joint-stability passes.
Preserve that candidate as a failed experiment, including its reference scores.

Retain the SCHEDULED_AUDIT_PROTOCOL.md schedule, audit mixture, RNG, numerical
checks, KL thresholds, size-refinement policy and cost accounting unchanged.
Change acceptance only: the audit must pass AND the saved GP posterior must
have passed two consecutive numerically resolved joint-stability checks.
These are the existing checks from extension_v2: successive-posterior KL <0.001
and stable moments/intervals, checked every 30 training labels. Do not impose
the prediction counter or pointwise residual checks. Audits still run on the
fixed schedule even if posterior stability has not yet passed.

Use identical audit seeds at matching checkpoints to isolate this change;
continue to new fresh audits where the old candidate stopped prematurely.
Both fixed96 and adaptive384 use the same stability requirement. A bootstrap
pass at an unstable GP continues training and cannot establish convergence.

Freeze this additional requirement before rerunning the complete 48-trajectory
development comparison. Do not select stable checkpoints using reference KL.
Validate all four marginal posteriors at accepted stops. This is still known-
catalogue development evidence, not independent confirmation or a universal
convergence guarantee. No new parameter-space reference rate grid is generated.
