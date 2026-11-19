# Four-input paper: stage two started

- Pilot **16623041**, training array **16623042**, diagnostic array **16623043**.
- Forty remaining primary runs: eight at 500 events, twenty at 700 and twelve at 1000. These complete the planned 140-trajectory matrix when combined with the earlier 100; computational completion does not imply every fit passes.
- Four predictive examples retain the original 30/1200-event catalogues and both measurement views. The 30-event fits use 1284 labels; all other new fits use 1884.
- Training settings and the original accuracy criteria are unchanged. The existing KL=0.0105 case remains a documented narrow miss; no extra run is launched solely to cross that threshold.
- Twelve diagnostic tasks use completed catalogue-200 BO GPs across counts 10/50/100/200/500/1000 and both views. Each tests conditional amplitude at the injection shape and the direct-reference posterior mean shape. Actual observed counts determine the Gamma shape; exact rates determine its rate. These are checks of the four-input GP, not separately fitted one-input surrogates.
- Four of those tasks (100/1000 events, both views) also evaluate withheld four-input likelihood predictions and inverse-transformed predictive intervals on two reference designs. Raw arrays, coverage, residuals, ESS, maximum weight and training-distance checks are saved. Between-design scatter remains to be reviewed.
- Training runs four at a time; diagnostics two at a time. Both arrays depend on the fresh-run/resume pilot succeeding. No new stellar-evolution simulations or changes to prior bounds are involved.
- Local validation: **213 passed, 3 deselected**, plus shell syntax and whitespace checks. Source/input hashes and prior completion were verified before submission.
- Reproducible workflow: `scripts/ozstar/gp4/`. Frozen submission record: `docs/studies/amplitude_gp/manuscript_stage2_20260917/`. Remote root: `/fred/oz303/avajpeyi/studies/compas_gp4_manuscript_stage2_20260917`.
- Next after completion: review outcomes, use joint four-input posterior draws for rate maps and predictive catalogues, generate diagnostic/width/corner plots, then replace corresponding manuscript figures and prose. No figure or scientific pass is claimed by this submission.

## Retrieval — 18 September 2026

All stage-two allocations completed successfully. `RESULTS.md` and
`combined_trajectory_summary.csv` contain the complete main-campaign summary;
`results_snapshot.json` preserves full scoring records. Download raw training
artifacts without modifying remote originals with:

```bash
rsync -a --stats -e 'ssh -o BatchMode=yes' \
  --include='/production/***' --include='/logs/***' --include='/pilot/***' \
  --exclude='*' \
  ozstar:/fred/oz303/avajpeyi/studies/compas_gp4_manuscript_stage2_20260917/ \
  docs/studies/amplitude_gp/manuscript_stage2_20260917/
```

The retrieved NPZ checkpoints retain every training point, fitted parameters
and RNG state. Acquisition batches, optimizer reports, timings, scoring records
and logs remain alongside them. `download_verification.json` records the final
local checkpoint verification when retrieval completes.
