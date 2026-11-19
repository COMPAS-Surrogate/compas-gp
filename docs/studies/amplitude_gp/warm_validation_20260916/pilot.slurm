#!/usr/bin/env bash
#SBATCH --job-name=gp4-warm-smoke
#SBATCH --output=/fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916/pilot-%j.out
#SBATCH --error=/fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916/pilot-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
export COMPAS_REPO=/fred/oz303/avajpeyi/studies/compas_gp4_paper_20260915/code
export COMPAS_ENV=/fred/oz303/avajpeyi/codes/cosmic_integration_gp4_20260914/.venv
source "$COMPAS_REPO/scripts/ozstar/runtime.sh"
python /fred/oz303/avajpeyi/studies/compas_gp4_warm_validation_20260916/warm_validation.py pilot
