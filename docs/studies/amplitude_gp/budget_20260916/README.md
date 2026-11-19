# Four-input GP budget investigation — 16 September 2026

## Scope and decision

- Test saved designs for catalogue 200, seed 17, at 100 expected events, with paired perfect/uncertain measurement views. These views share one latent catalogue; they are not independent replications.
- Keep all four GP inputs and every training label. No new training likelihood evaluations or BO trajectory were run.
- **The BO designs already contain enough information at 1254 labels:** refitting their hyperparameters passes the independent posterior gates at both 1254 and 1284 labels for both measurement views.
- This establishes sufficiency of those fixed designs in this case, **not a universal budget or the minimum required**. Optimized fits below 1254 have not been tested here.
- The original fresh-start Adam fits fail at 744, 774, 1254 and 1284. The improved fit can therefore remove an apparent need for more training points.

## Method

- Baseline screen: restore frozen checkpoints at 744, 774, 1254 and 1284; perform the existing full posterior comparisons, not merely likelihood residual checks.
- Use the same two independent shape-reference designs and the same amplitude-proposal seeds for BO/random and for every checkpoint of a given catalogue.
- Apply the existing 16-replica amplitude design, increasing to 64 only if numerically unresolved. Accuracy requires both directed KLs and parameter-summary tolerances, alongside reference and GP numerical-precision gates.
- Refit control: retain exactly the same likelihood labels, Matérn-5/2 kernel family, fixed noise and square-root target transformation. Initialize hyperparameters from the previous saved checkpoint, rescale for the current target scale, and use L-BFGS-B for at most 100 iterations.
- Refit outputs are separate candidates, not replacements for the frozen campaign. Candidate checksums and optimizer terminations are recorded.
- Offline scoring removes algebraically inactive GP padding; a regression test checks agreement with padded predictions.

## Original fits: maximum directed posterior KL (nats)

| Labels | BO perfect | BO uncertain | Random perfect | Random uncertain |
|---:|---:|---:|---:|---:|
| 744 | 0.3360 | 0.2515 | 0.5714 | 0.4848 |
| 774 | 0.3064 | 0.2467 | 0.5372 | 0.4487 |
| 1254 | 0.0378 | 0.0457 | 0.2907 | 0.2409 |
| 1284 | 0.0263 | 0.0448 | 0.2733 | 0.2199 |

All 16 are numerically resolved and inaccurate; the KL threshold is 0.01, together with the parameter-summary gates.

## Refitted fixed designs

BO passes at 1254 and 1284 for both measurement models. Random fails at both budgets under the **same** refit protocol. All eight comparisons are numerically resolved.

| Labels | BO perfect | BO uncertain | Random perfect | Random uncertain |
|---:|---:|---:|---:|---:|
| 1254 | 0.00311 | 0.00320 | 0.04763 | 0.04858 |
| 1284 | 0.00310 | 0.00300 | 0.03545 | 0.03609 |

This is a matched fixed-budget advantage for BO in this catalogue, not a general efficiency ratio. No confirmed random budget has been found in the tested refit checkpoints; do not infer `>2994` for the improved optimizer.

- Each BO refit takes about 30–32 seconds; the complete refit-plus-reference score takes about 90 seconds at these sizes. This is not the cost of generating the full training design.
- All eight optimizations (BO and random) reach the 100-iteration cap: **the optimizer did not declare convergence**, even though the independent posterior-accuracy gates pass. These are different notions of success.
- `first_confirmed_tested = 1254` means the fit is accurate there and again at 1284. It requires 1284 acquired labels to have both checkpoints available, and excludes the separate offline reference-evaluation cost.
- This retrospective, reference-based confirmation is not an operational stopping rule. The actual saved training runs continued to 2994; their sunk computation has not been recovered.
- Improved fixed-design fits do not prove an entire warm-start BO trajectory would select the same points or attain the same result.

## Comparison with the three-input case

For the same count, catalogue and both measurement views, the previous three-input audit-selected stop used 298 training labels plus 192 audit calls (490 total). Its 1258-label endpoint was a diagnostic continuation. The baseline analytically marginalizes amplitude; these new candidates learn amplitude as a fourth GP input. This is a method comparison, not a matched initialization or wall-time benchmark.

## Next bounded validation

1. Test optimized BO fits below 1254 to bracket a smaller sufficient budget; separately extend optimized random fits if their current checkpoints fail.
2. Repeat on other catalogue realizations, seeds and event counts; do not generalize from this single latent catalogue.
3. Run a small complete warm-start BO/random comparison with independent posterior scoring and measured wall time.
4. Validate a training-time stopping policy before resubmitting the full paper campaign. Keep the 612 cancelled pending entries available for resubmission.

## Reproduction and provenance

- Runner: `../score_checkpoint_history.py`.
- Remote root: `/fred/oz303/avajpeyi/studies/compas_gp4_budget_20260916/`.
- Job arrays: 16578410 (original fits), 16578447 (BO refit), 16578458 (matched random refit).
- `manifest.json` records scopes, budgets and source hashes. Remote original and refit scripts are kept separately because the latter adds the optional optimizer control.
- `retrieved/screen/` and `retrieved/refit/` contain per-checkpoint scores, reference identities, candidate checkpoints, optimizer diagnostics and completion states.
- `fixed_design_audit.json` verifies all eight candidates against the frozen originals: identical four-dimensional coordinates and likelihood labels, unchanged noise/jitter and target-transform parameters.
- `summarize_retrieved` verifies outcome consistency, candidate hashes and common proposal seeds before writing `summary.json`.
