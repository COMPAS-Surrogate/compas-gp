#!/usr/bin/env bash
#SBATCH --job-name=compas-gp4-paper
#SBATCH --output=compas-gp4-paper-%A_%a.out
#SBATCH --error=compas-gp4-paper-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
set -euo pipefail
: "${COMPAS_REPO:?Set COMPAS_REPO}"
: "${COMPAS_INPUTS:?Set COMPAS_INPUTS to the frozen campaign inputs}"
: "${COMPAS_RUN_DIR:?Set COMPAS_RUN_DIR to a new campaign output directory}"
source "$COMPAS_REPO/scripts/ozstar/runtime.sh"
if [[ "${COMPAS_STAGE:-run}" == pilot ]]; then
    python "$COMPAS_REPO/scripts/ozstar/environment_check.py"
    exec python docs/studies/amplitude_gp/paper_campaign.py pilot --inputs "$COMPAS_INPUTS" \
        --output "$COMPAS_RUN_DIR" --index 24
fi
: "${SLURM_ARRAY_TASK_ID:?Submit the campaign as a Slurm array}"
exec python docs/studies/amplitude_gp/paper_campaign.py run --inputs "$COMPAS_INPUTS" \
    --output "$COMPAS_RUN_DIR" --index "$SLURM_ARRAY_TASK_ID" --maximum 2994
