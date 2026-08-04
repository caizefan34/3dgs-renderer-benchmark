#!/bin/bash
# Aggregate the round-57 sparse-pixel-rasterization matrix into r57-summary.json.
# Arms: px035 (renderer-level sparse pixels @ 35%) vs px100 (its own dense
# baseline, identical composed pipeline). Reference = dense -> delta/speedup of
# sparse vs dense.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
$PY scripts/higs/aggregate_run_summary.py \
  --out results/higs-round57/r57-summary.json \
  --group train_px100   "results/higs-round57/r57_train_px100_s*.json" \
  --group train_px035   "results/higs-round57/r57_train_px035_s*.json" \
  --group garden_px100  "results/higs-round57/r57_garden_px100_s*.json" \
  --group garden_px035  "results/higs-round57/r57_garden_px035_s*.json" \
  --group bicycle_px100 "results/higs-round57/r57_bicycle_px100_s*.json" \
  --group bicycle_px035 "results/higs-round57/r57_bicycle_px035_s*.json" \
  --reference train_px035 train_px100 \
  --reference garden_px035 garden_px100 \
  --reference bicycle_px035 bicycle_px100
