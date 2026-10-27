#!/usr/bin/env bash
# The model comparison: DeepLabV3+ and SegFormer-B2 on object and shadow segmentation
# (U-Net is in run_all.sh). Resumable: stop it any time (Ctrl+C, or `docker stop`) and run it
# again; finished runs are skipped and an interrupted one continues from its last epoch.
#   MARE_DATA=/path/to/dataset ./run_models.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${MARE_DATA:?set MARE_DATA to the dataset folder (with samples/)}"
export MARE_DATA
docker compose build
run() { docker compose run --rm train "$@"; }
for arch in deeplabv3plus segformer; do
  for task in object shadow; do
    run mare_baselines.seg --task "$task" --arch "$arch" --epochs 30 --max-minutes 120
  done
done
run mare_baselines.plots
run mare_baselines.figures
