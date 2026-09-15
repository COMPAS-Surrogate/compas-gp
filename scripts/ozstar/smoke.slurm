#!/usr/bin/env bash
#SBATCH --job-name=compas-gp4-smoke
#SBATCH --output=compas-gp4-smoke-%j.out
#SBATCH --error=compas-gp4-smoke-%j.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail
: "${COMPAS_REPO:?Set COMPAS_REPO}"
: "${COMPAS_RUN_DIR:?Set a new output directory}"
source "$COMPAS_REPO/scripts/ozstar/runtime.sh"
mkdir "$COMPAS_RUN_DIR"
python "$COMPAS_REPO/scripts/ozstar/environment_check.py" > "$COMPAS_RUN_DIR/environment.json"
python -m pytest -q tests/test_gp_noise_freeze.py tests/test_gpry_rules.py tests/test_four_input_followup.py \
    --junitxml="$COMPAS_RUN_DIR/tests.xml" > "$COMPAS_RUN_DIR/tests.log" 2>&1
