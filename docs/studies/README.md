# Research studies

Start with the four-input GP. Other folders remain because they supply
reference calculations, physics checks or helpers imported by that workflow.
Generated results are local evidence, not additional supported entry points.

## Active entry points

| Folder | Role |
|---|---|
| [`amplitude_gp/`](amplitude_gp/README.md) | Four-input GP development, continuation and validation |
| `grid_resolution/` | Mc–z discretisation checks |
| `population_precision/`, `population_stability/` | Finite-population checks and rate data |
| `generate_data/` | COMPAS rate-generation Slurm scripts |

## Reference and supporting studies

| Folder | Why it remains |
|---|---|
| `operational_stopping/` | Existing BO machinery, stopping work and three-input reference results |
| `event_uncertainty/` | Catalogue generation and independent posterior references |
| `event_count_bo/`, `joint_pilot/` | Shared experimental helpers and baseline comparisons |
| `paper_completion/`, `rate_heatmaps/` | Existing manuscript results and figure construction |

Some scripts import helpers across these folders. Moving those scripts without
first extracting the shared helpers would break the current workflow. Their
presence does not imply that every old experiment needs rerunning.

`ANALYSIS_STATUS.md`, `PAPER_SCOPE.md` and `POSTERIOR_VALIDATION_PLAN.md` retain
historical context. For the latest four-input evidence use the amplitude-GP
index; older three-input conclusions do not validate four-input inference.

## Archived studies

The following standalone studies were removed from the active checkout on
2026-09-15:

- `finite_sample_bias`: early finite-sample derivations and exploratory plots.
- `lnl_1d`: old saved one-dimensional scan.
- `parse_ilya_posteriors`: one-off posterior parsing.
- `simulation_study`: historical simulation plotting scripts.
- `subset_training`: early GP transform, acquisition and training-set experiments.

Local archive:
`/Users/avi/Documents/projects/compas/cosmic_integration_archive/legacy_studies_20260915/`.
It preserves 23 Python scripts and their associated results, including existing
uncommitted edits. `manifest.json` records original paths and source hashes;
`VERIFIED` records the move checks. No test datasets were moved or removed.

These scripts are historical snapshots. To rerun one, restore its directory
from `docs/studies/NAME` inside the archive to the same path in a suitable
checkout; several scripts depend on their original relative paths. Old links
in historical reports or project notes may refer to those archived paths.
The far-tail training-point findings recorded in AGENTS.md still apply.

## Versioned study material

Study source, reports, figure PDFs/PNGs and compact text/JSON/CSV summaries are
versioned with the package. Large datasets, NumPy arrays, posterior caches,
per-step optimizer dumps and copied runtime environments remain local. This
source release is not a self-contained data archive: reproducing the numerical
experiments still requires the population data and run inputs named by each
study. The manuscript is maintained in the separate `overleaf/` checkout and
project notes in the separate `wiki/` checkout.
