# Stage-one submission — 17 September 2026

- Reproducible runner and submission files: `scripts/ozstar/gp4/`.
- OzSTAR root: `/fred/oz303/avajpeyi/studies/compas_gp4_manuscript_stage1_20260917`.
- Pilot **16611906**; array **16611907**, 92 tasks, maximum four concurrent.
- Initial verified state: pilot pending; all 92 array tasks pending its successful completion. No scientific result claimed by submission.
- 16 existing 500-event continuations to 1884; eight new 1000-event runs to 1884; 68 missing lower-count runs to 1284. Matched methods share budgets and scoring checkpoints. The fixed 1884 high-count budget includes scoring at 1284 rather than requiring a separate mid-job extension submission.
- Remaining high-count and predictive runs remain deferred pending review. No automatic extension beyond 1884.
- Local tests: 210 passed, 3 deselected; shell syntax and diff whitespace checks passed. Upload hashes and parent provenance verified remotely before submission.
- `manifest.json`, `checksums.json`, `upload_checksums.json` and `submission.json` preserve the exact deployed identities and Slurm IDs. `initial_status.json` is a timestamped submission snapshot, not a live monitor.
