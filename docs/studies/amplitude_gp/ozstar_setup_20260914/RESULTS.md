# OzSTAR setup and four-input acquisition sanity check

Completed 2026-09-15 (NZ time). No full inference campaign submitted.

## Environment and Slurm

Isolated deployment: `/fred/oz303/avajpeyi/codes/cosmic_integration_gp4_20260914`.
Its `.venv` uses Python 3.12.12, JAX/JAXlib 0.10.2, GPJax 0.18.0,
Equinox 0.13.8, Optax 0.2.8 and NumPyro 0.21.0. CPU/x64 checks passed.

The prior lockfile did not match pyproject.toml. Regeneration removed 85
obsolete package entries, without changing any retained package versions.
OzSTAR `uv sync --locked` now succeeds. The original remote lock is retained
as `uv.lock.original`. Existing environments and data were not changed.

The three historical Slurm scripts use explicit repository, environment,
input and output paths and a shared CPU runtime. They check existing outputs,
including the regular CLI's appended `.csv`; random-array output is CSV.
The regular `-n 1` remains a sample count. Installation happens on the login
node, not inside batch jobs. The broken `run_cosmic_integration` console
entry point was corrected to the callable in `ratesSampler.main` and its
installed `--help` entry point was verified locally and remotely.

Slurm smoke job **16546831** completed on **dave1**, exit **0:0**, elapsed
**46 seconds**. All **17 focused GP/acquisition tests passed**. See
`smoke_01/environment.json`, `tests.log`, `tests.xml` and
`deployment_verification.json`. Only the build-generated `_version.py`
differed from the uploaded source manifest during that verification.
The final CLI correction followed the smoke job and was separately checked
by locked reinstall and the actual console `--help` invocation.

Local full fast suite: **188 passed**, 3 heavy/network tests deselected,
one pre-existing Matplotlib warning. Added tests cover all four acquisition
gradients, fantasy variance, shell argument quoting, output guards, refusal
to overwrite environments, and the console entry point. Bash syntax and
`git diff --check` passed. No commits or pushes.

## Four-input BO

`../audit_continued_bo.py` checks all 12 continuation trajectories and all
9,000 added labels. Checkpoint hashes, exact 744-label prefixes, four-coordinate
bounds, uniqueness, batch-to-checkpoint label alignment and acquisition
novelty all pass. Results are saved in
`../continued_bo_20260914/acquisition_audit.json`.

Across the twelve runs, errors for the final 90 acquisition-selected points
near the peak are lower than for the first 90. Final near-peak RMSE ranges
are 0.111–0.139 lnL at expected count 100 and 0.308–0.431 at count 1000.
These sets differ and are selected by acquisition; they are not independent
holdouts or calibration evidence. Local optimizer success fractions range
from 0.626 to 0.978; unsuccessful starts do not automatically imply an
invalid batch because the acquisition retains the best finite novel candidate.

Together with the existing independent posterior scoring (all twelve joint
KL estimates improve from 744 to 1494 labels), this supports a functioning
acquisition loop. No four-input run yet passes the joint posterior accuracy
gate. This setup smoke test does not validate the full COMPAS data pipeline,
posterior sampling, missing modes, or production stopping on OzSTAR.
