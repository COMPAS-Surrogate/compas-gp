# COMPAS cosmic integration and likelihood surrogates

Compute population rates, evaluate observational likelihoods, and train a
Gaussian-process surrogate using Bayesian optimisation.

## Start here

| Location | Purpose |
|---|---|
| [`src/cosmic_integration/`](src/cosmic_integration/) | Reusable package code and command-line tools |
| [`tests/`](tests/) | Automated tests; large local datasets are ignored |
| [`docs/studies/`](docs/studies/README.md) | Current research workflows and their dependencies |
| [`scripts/ozstar/`](scripts/ozstar/README.md) | Locked environment setup and Slurm smoke test |
| `overleaf/` | Separate manuscript checkout |

The current research target is the **four-input likelihood GP**, including
SFR amplitude. Its development runs are in
[`docs/studies/amplitude_gp/`](docs/studies/amplitude_gp/README.md).
The analytical calculation remains the accuracy reference.

## Install and test

```bash
uv sync --locked --python 3.12.12
uv run --locked pytest
```

The default test suite excludes slow and network tests. Use `pytest -m slow`
for heavy tests or `pytest -m ""` for the complete suite.
For cluster installation, follow the [OzSTAR instructions](scripts/ozstar/README.md).

## Package commands

- `run_cosmic_integration --help`: generate rate grids from COMPAS data.
- `run_surrogate_workflow --help`: run the packaged surrogate workflow.
- `run_1d_lnl_check --help`: inspect a likelihood slice.

The packaged workflow and the experimental four-input study are separate
entry points; see the study guide before launching the research campaign.

## Historical work

Superseded standalone studies have been archived outside this checkout,
with source hashes and restoration instructions. See the
[study index](docs/studies/README.md#archived-studies).
Datasets, existing environments and the manuscript were left in place.
