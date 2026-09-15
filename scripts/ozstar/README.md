# OzSTAR environment and jobs

Use a dedicated Python 3.12.12 CPU environment. `uv.lock` fixes dependency
versions; `setup.sh` runs `uv sync --locked` on a login node and refuses to
modify an existing environment. Batch jobs never install packages.

```bash
export COMPAS_REPO=/absolute/path/to/checkout
export COMPAS_ENV=/absolute/path/to/new/environment
bash "$COMPAS_REPO/scripts/ozstar/setup.sh"
```

For a source bundle without `.git`, also set `COMPAS_SOURCE_VERSION` to the
recorded installed package version. Keep a source hash manifest alongside the
bundle: the version alone does not identify uncommitted scientific changes.

Run the small CPU/x64 environment and acquisition check on a compute node:

```bash
export COMPAS_RUN_DIR=/absolute/path/to/new/smoke-results
cd "$COMPAS_REPO"
sbatch --export=ALL scripts/ozstar/smoke.slurm
```

`environment.json`, `tests.log`, and `tests.xml` record results. This check
requires no COMPAS HDF5 inputs and is not a population-inference campaign.
Check Slurm's exit state as well as the test report before declaring success.

The historical data-generation scripts now share `runtime.sh`. Set
`COMPAS_INPUT` to an existing HDF5 file and `COMPAS_OUTPUT` to a new output
path before submitting. The random array treats `COMPAS_OUTPUT` as a prefix,
appending `_TASK_ID.csv`; its underlying CLI runs until the wall-time limit.
The regular grid CLI also treats the output as a prefix and appends `.csv`;
the wrapper checks for that file before starting. Its `-n 1` is a sample
count, not a CPU allocation.
The 32M/512M filenames are historical labels: the supplied input determines
the population. Array logs are written in the submission directory.

Jobs use explicit paths, CPU JAX, 64-bit arithmetic, and Slurm's allocated
thread count. The one-CPU, 8-GB smoke allocation is a test allocation, not a
measured recommendation for full GP training. Existing environments and
population files are left in place.
