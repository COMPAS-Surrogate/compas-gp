#!/usr/bin/env bash
# Run on a login node with network access; never install in a batch job.
set -euo pipefail
: "${COMPAS_REPO:?Set COMPAS_REPO}"
: "${COMPAS_ENV:?Set COMPAS_ENV to a NEW environment path}"
if [[ -e "$COMPAS_ENV" ]]; then
    echo "Refusing to modify existing environment: $COMPAS_ENV" >&2
    exit 1
fi
cd "$COMPAS_REPO"
export UV_PROJECT_ENVIRONMENT="$COMPAS_ENV"
# Source bundles without .git must carry their recorded package version.
if [[ ! -d .git ]]; then
    : "${COMPAS_SOURCE_VERSION:?Set the package version recorded in the source bundle}"
    export SETUPTOOLS_SCM_PRETEND_VERSION="$COMPAS_SOURCE_VERSION"
fi
uv sync --locked --python 3.12.12
