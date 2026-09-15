#!/usr/bin/env bash
# Source from a job after setting COMPAS_REPO and COMPAS_ENV.
set -euo pipefail
: "${COMPAS_REPO:?Set COMPAS_REPO to the deployed checkout}"
: "${COMPAS_ENV:?Set COMPAS_ENV to the dedicated locked environment}"
test -x "$COMPAS_ENV/bin/python"
cd "$COMPAS_REPO"
export PATH="$COMPAS_ENV/bin:$PATH"
export PYTHONPATH="$COMPAS_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/compas-mpl-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
