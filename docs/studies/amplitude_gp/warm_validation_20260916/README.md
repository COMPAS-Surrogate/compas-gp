# Bounded OzSTAR warm-fit validation — 16 September 2026

## Submitted scope

- **8 saved-design checks:** 100/500 expected events, perfect/uncertain views, BO/random, catalogue 200, seed 17. Refit and independently score 504, 534, 744 and 774 labels.
- **32 fresh trajectories:** 50/500 expected events, perfect/uncertain views, BO/random, catalogues 200–202 at seed 17; additionally catalogue 200 at seed 29. Matching methods share catalogue, prior, initialization and reference designs.
- Four validation tasks may run concurrently. Each requests one CPU, 16 GB and two hours. Previously running paper jobs are independent and may finish; cancelled paper entries are not resubmitted.
- A separate compute-node smoke test trains to 174, resumes to 204, verifies padded/active prediction parity and deterministic acquisition replay from restored RNG state. The full array depends on its successful exit. It is an implementation gate, not scientific validation or a production runtime benchmark; representative large-checkpoint timings were measured earlier.

## Frozen training policy

- Four inputs `(alpha, sigma, a, d)`, Matérn-5/2, square-root target, fixed noise, all labels retained, same priors and likelihood as the original campaign.
- 64 scrambled Sobol points plus 80 boundary anchors. Initial 144-label GP uses the existing 250-step Adam fit.
- Three-point BO batches, or three IID random points; condition immediately with full exact conditioning. The experimental incremental inverse update is not introduced in this campaign.
- Warm-start L-BFGS-B refits every 30 labels: maximum 100 iterations, 10 line-search steps, `ftol=1e-9`, `gtol=1e-5`. Transform metadata is refreshed consistently; previous kernel mean/variance are rescaled for the new target scale.
- Every optimizer's termination, objective and timing are retained. Hitting the iteration cap is not optimizer convergence. Nonfinite or worsened objectives fail the job instead of silently accepting a broken fit.
- Fixed ceiling **1284 labels**. Save raw labels, GP parameters, RNG and timing every 30 labels; immutable run identities support checkpoint recovery.
- Training does not access posterior reference designs. No automatic early stopping or prior narrowing is introduced.

## Scoring and scale-up gates

- After a fresh trajectory completes, score checkpoints 504/534, 744/774 and 1254/1284 against the same two independent direct-reference designs and paired amplitude proposals.
- Keep the existing full joint-posterior gates: both directed KLs below 0.01 nats, mean/endpoint differences below 0.2 reference SD, 68%/95% width discrepancies below 10%, plus independent numerical-precision checks.
- Increase amplitude integration from 16 to 64 replicas only when unresolved. Report unresolved numerical checks separately from inaccurate GP fits.
- Report first **tested** pass confirmed 30 labels later, with the confirmation cost. This is a retrospective reference-based comparison, not a deployable stopping rule or exact minimum required budget.
- Compare BO/random on the same checkpoints, preserve all failures, and measure actual acquisition/fitting/conditioning/likelihood costs. Report full training and separate scoring time; do not infer an end-to-end speedup from a single kernel operation.
- Do not restart the 632-fit campaign automatically. Review recovery failures, timing, numerical resolution and scientific passes by count/catalogue/measurement/seed first. Extend or revise only failed/unresolved cases with a recorded reason.

## Files and recovery

- Remote root: `/fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916/`.
- Original frozen code and input sources remain under `compas_gp4_paper_20260915/`.
- `manifest.json` enumerates every task; `checksums.json` freezes the validation driver, scoring helper, manifest and this protocol.
- `pilot_success.json` records the implementation gate; `lower_budget/` contains fixed-design checks; `trajectories/runs/` contains fresh training histories and independent scores.
- `complete.json` means training completed; `status.json` plus the history records show whether scoring completed and passed. Scheduler state must also be checked for timeouts or failures.
- Re-execute the same task index with identical source/input identities to resume; never patch the frozen deployed source in place.
