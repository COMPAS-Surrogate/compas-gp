#!/usr/bin/env bash
# Run from the repository root. The argument must be a new output directory.
set -euo pipefail
stopping_output=${1:?Pass a new output directory}
study_dir=docs/studies/operational_stopping
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-mpl
mkdir "$stopping_output"

.venv/bin/python "$study_dir/run_study.py" \
  --outdir "$stopping_output/fresh" --seeds 3000 3001 3002 3003 3004
.venv/bin/python "$study_dir/compare_rules.py" replay \
  --source "$stopping_output/fresh" --outdir "$stopping_output/replay"
.venv/bin/python "$study_dir/compare_rules.py" first-audit \
  --source "$stopping_output/fresh" --outdir "$stopping_output/first_audit"
.venv/bin/python "$study_dir/compare_rules.py" impact-replay \
  --source "$stopping_output/fresh" --audit-source "$stopping_output/first_audit" \
  --outdir "$stopping_output/impact"
.venv/bin/python "$study_dir/compare_rules.py" audit-stress \
  --source "$study_dir/confirmation_v1" --outdir "$stopping_output/audit_stress"
.venv/bin/python "$study_dir/compare_rules.py" replay \
  --source "$study_dir/compas_v1" --outdir "$stopping_output/replay_compas"
.venv/bin/python "$study_dir/compare_rules.py" first-audit-compas \
  --source "$study_dir/compas_v1" --outdir "$stopping_output/first_audit_compas"
.venv/bin/python "$study_dir/compare_rules.py" impact-replay \
  --source "$study_dir/compas_v1" --audit-source "$stopping_output/first_audit_compas" \
  --outdir "$stopping_output/impact_compas"

# Regenerate the shipped figures/report from their archived inputs:
# .venv/bin/python "$study_dir/publish_rule_comparison.py"
