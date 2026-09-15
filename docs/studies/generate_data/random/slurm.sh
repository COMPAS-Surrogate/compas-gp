#!/bin/bash
#SBATCH --job-name=generate_data_%A_%a
#SBATCH --output=generate_data_%A_%a.out
#SBATCH --error=generate_data_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-5

set -euo pipefail
: "${COMPAS_REPO:?Set COMPAS_REPO}"
: "${COMPAS_INPUT:?Set COMPAS_INPUT to the input HDF5 file}"
: "${COMPAS_OUTPUT:?Set COMPAS_OUTPUT to a new output path}"
: "${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array}"
COMPAS_OUTPUT="${COMPAS_OUTPUT}_${SLURM_ARRAY_TASK_ID}.csv"
source "$COMPAS_REPO/scripts/ozstar/runtime.sh"
test -f "$COMPAS_INPUT"
if [[ -e "$COMPAS_OUTPUT" ]]; then
    echo "Output already exists: $COMPAS_OUTPUT" >&2
    exit 1
fi
mkdir -p "$(dirname "$COMPAS_OUTPUT")"
exec generate_random_samples "$COMPAS_INPUT" -s "$SLURM_ARRAY_TASK_ID" -o "$COMPAS_OUTPUT"
