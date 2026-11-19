# Four-input GP

This is the active development study. The inputs are alpha, sigma, SFR
amplitude and SFR d. Passing acquisition checks is distinct from passing
joint-posterior accuracy checks.

## Current evidence

- [`paper_campaign_20260915/`](paper_campaign_20260915/README.md): submitted
  OzSTAR campaign for 632 four-input paper fits; submission is not validation.
- `continued_bo_20260914/results.json`: all twelve BO runs at 1,494 labels;
  every joint KL improved, but no run passes the accuracy gate yet.
- `continued_bo_20260914/acquisition_audit.json`: saved acquisition histories
  and all 9,000 added labels pass integrity checks.
- `ozstar_setup_20260914/RESULTS.md`: environment setup and compute-node smoke test.
- `small_batch_20260914/RESULTS.md`: matched BO/random baseline and fitting ablations.

## Scripts to use

| Script | Purpose |
|---|---|
| `paper_campaign.py` | Prepare frozen paper inputs, train one array task, and score it |
| `continue_bo.py` | Existing bounded 744→1494 continuation; not a generic resumable runner |
| `audit_continued_bo.py` | Verify the completed continuation from the repository root |
| `conditional_slices.py` | Known-answer conditional amplitude checks |
| `small_batch_followup.py` | Baseline training, state loading and reference-scoring helpers |

`run_pilot.py` and `stratified_reference.py` are still imported by the current
runner. Keep them until those shared helpers are extracted.

The other scripts reproduce earlier fixed-label optimizer/transform ablations,
rescore the old large-batch pilot, or summarize and verify those results. They
are supporting diagnostics, not alternative production pipelines. The protocols
record their budgets and evidence boundaries. See `scripts/ozstar/` at the
repository root for environment setup; do not launch old pilots merely because
the scripts are present.
