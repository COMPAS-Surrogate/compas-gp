#!/usr/bin/env bash
#SBATCH --job-name=gp4-warm-validation
#SBATCH --output=/fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916/validation-%A_%a.out
#SBATCH --error=/fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916/validation-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
export COMPAS_REPO=/fred/oz303/avajpeyi/studies/compas_gp4_paper_20260915/code
export COMPAS_ENV=/fred/oz303/avajpeyi/codes/cosmic_integration_gp4_20260914/.venv
source "$COMPAS_REPO/scripts/ozstar/runtime.sh"
python /fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916/warm_validation.py run --task-index "$SLURM_ARRAY_TASK_ID"
