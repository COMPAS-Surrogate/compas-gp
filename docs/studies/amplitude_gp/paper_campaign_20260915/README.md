# Four-input paper campaign

This campaign starts four-input GP replacements for the GP-dependent results
in the current `overleaf/ms.tex` and included sections. It reuses the original
catalogues and fixed 256M COMPAS population. It does not rerun stellar evolution.

| Group | Fits | Purpose |
|---|---:|---|
| Count histories | 140 | 70 original paired cases, BO and random |
| Predictive examples | 4 | Perfect/uncertain views at expected counts 30 and 1200 |
| Error-width sweep | 160 | 20 latent catalogues × four widths × BO/random |
| Robustness | 108 | Three injections, two exposures, two catalogues, three seeds and three strategies |
| Space-filling comparison | 120 | 40 original catalogues × BO/random/Sobol |
| Categorical control | 100 | Original prior-drawn categorical-response catalogues |

Every fit starts with 64 Sobol points plus 80 boundary anchors in four
coordinates. All labels are retained. Three-point acquisition, 30-label refits,
square-root target, fixed noise and Matérn-5/2 follow the four-input development
configuration. The fixed budget is 2,994 training labels; this changes the
historical budgets/initializations and does not inherit their stopping claims.
Compact checkpoints every 30 labels permit later stopping-policy replay.

Training never reads the separate reference designs. Final scoring compares
against exact amplitude-integrated/reconstructed references with two independent
shape designs and stratified amplitude proposals (16, then 64 replicas if
unresolved). Existing source designs may be inadequate for some catalogue
locations: numerical failures remain unresolved, not inaccurate or successful.
Posterior uncertainty calibration and prior-wide mode discovery are not certified.

A compute-node pilot first verifies input hashes, portable exact-rate parity,
and a 174-label high-count BO run. The full array depends on its successful
Slurm exit. Four jobs run concurrently, each allocated one CPU, 16 GB and eight
hours. Wall-time failures can resume from the last complete checkpoint with
an identical source/input identity. `complete.json` means training completed;
`score.json` reports scientific accuracy. Keep failed and unresolved jobs in
all denominators.

Direct COMPAS likelihood-loss, pixel-support, physical-rate and event-weight
illustrations do not use a GP and remain unchanged. Posterior predictive/rate
maps, conditional amplitude checks, and reference-free stopping replay require
post-processing the new states. No manuscript figures or claims are replaced
by submission of this campaign.

Remote root:
`/fred/oz303/avajpeyi/studies/compas_gp4_paper_20260915/`.
See `campaign_manifest.json`, `source_manifest.json`, and `submission.json` for
inputs, source fingerprints and Slurm IDs. Runner: `../paper_campaign.py`.
