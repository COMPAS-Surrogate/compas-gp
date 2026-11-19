#!/usr/bin/env bash
#SBATCH --job-name=compas-gp4-paper
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail
: "${COMPAS_CAMPAIGN:?Set COMPAS_CAMPAIGN to the frozen bundle}"
: "${COMPAS_REPO:?Set COMPAS_REPO to the frozen original source}"
: "${COMPAS_ENV:?Set COMPAS_ENV to the existing locked environment}"
source "$COMPAS_REPO/scripts/ozstar/runtime.sh"
if [[ "${COMPAS_STAGE:-run}" == pilot ]]; then
    exec python "$COMPAS_CAMPAIGN/campaign.py" pilot --root "$COMPAS_CAMPAIGN" --index 0
fi
: "${SLURM_ARRAY_TASK_ID:?Run as an array}"
if [[ "${COMPAS_STAGE:-run}" == diagnostics ]]; then
    exec python "$COMPAS_CAMPAIGN/diagnostics.py" --root "$COMPAS_CAMPAIGN" --index "$SLURM_ARRAY_TASK_ID"
fi
exec python "$COMPAS_CAMPAIGN/campaign.py" run --root "$COMPAS_CAMPAIGN" --index "$SLURM_ARRAY_TASK_ID"
