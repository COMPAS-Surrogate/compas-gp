#!/bin/bash
#SBATCH --job-name=generate_jeff_data
#SBATCH --output=generate_jeff_data%j.out
#SBATCH --error=generate_jeff_data%j.err
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail
: "${COMPAS_REPO:?Set COMPAS_REPO}"
: "${COMPAS_INPUT:?Set COMPAS_INPUT to the input HDF5 file}"
: "${COMPAS_OUTPUT:?Set COMPAS_OUTPUT to a new output path}"
source "$COMPAS_REPO/scripts/ozstar/runtime.sh"
test -f "$COMPAS_INPUT"
if [[ -e "$COMPAS_OUTPUT" || -e "$COMPAS_OUTPUT.csv" ]]; then
    echo "Output already exists: $COMPAS_OUTPUT" >&2
    exit 1
fi
mkdir -p "$(dirname "$COMPAS_OUTPUT")"
exec run_cosmic_integration "$COMPAS_OUTPUT" -i "$(basename "$COMPAS_INPUT")" -p "$(dirname "$COMPAS_INPUT")" -n 1 -v
