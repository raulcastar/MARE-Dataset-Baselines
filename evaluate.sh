#!/usr/bin/env bash
# Final evaluation of every trained baseline at native resolution -> results/<run>/eval.json,
# recording this repository's commit and the dataset revision.
#   MARE_DATA=/path/to/dataset [MARE_DATASET_REVISION=<Hub commit>] ./evaluate.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${MARE_DATA:?set MARE_DATA to the dataset folder (with samples/)}"
export MARE_DATA
MARE_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain -- mare_baselines Dockerfile requirements.txt 2>/dev/null)" ]; then
  MARE_GIT_COMMIT="$MARE_GIT_COMMIT+uncommitted-changes"   # commit first for numbers you publish
fi
export MARE_GIT_COMMIT MARE_DATASET_REVISION="${MARE_DATASET_REVISION:-}"
docker compose build
docker compose run --rm train mare_baselines.evaluate "$@"
