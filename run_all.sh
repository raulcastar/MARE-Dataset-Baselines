#!/usr/bin/env bash
# Trains and evaluates every baseline, one after another (each with a time limit).
#   MARE_DATA=/path/to/dataset ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${MARE_DATA:?set MARE_DATA to the dataset folder (with samples/)}"
export MARE_DATA
docker compose build
run() { docker compose run --rm train "$@"; }
run mare_baselines.seg --task object --epochs 30 --max-minutes 90
run mare_baselines.seg --task shadow --epochs 30 --max-minutes 90
run mare_baselines.seg --task spad_object --epochs 40 --max-minutes 60
run mare_baselines.seg --task rear_object --epochs 30 --max-minutes 90
run mare_baselines.depth --epochs 10 --max-minutes 90
