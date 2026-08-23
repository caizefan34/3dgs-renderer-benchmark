#!/bin/bash
# Aggregate the round-59 cull-mask refresh-interval sweep into r59-summary.json.
# 36 runs: 3 scenes x {fixed cull-interval 1,4,16} x 3 seeds + the phase-gated
# arm (g16 = cull every step until the densify window ends, every 16 after),
# all on the 720p eg recipe. k1 is the in-round baseline (== pre-patch
# per-step culling); k4/k16/g16 deltas and speedups are vs k1. The k1-vs-720p
# full reference reuses R56 same-resolution runs for the historical speedup
# convention; the R56 720p eg group is kept as a cross-round consistency check.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
$PY scripts/higs/aggregate_run_summary.py \
  --out results/higs-round59/r59-summary.json \
  --group train_k1   "results/higs-round59/r59_train_720p_k1_s*.json" \
  --group train_k4   "results/higs-round59/r59_train_720p_k4_s*.json" \
  --group train_k16  "results/higs-round59/r59_train_720p_k16_s*.json" \
  --group train_g16  "results/higs-round59/r59_train_720p_g16_s*.json" \
  --group garden_k1  "results/higs-round59/r59_garden_720p_k1_s*.json" \
  --group garden_k4  "results/higs-round59/r59_garden_720p_k4_s*.json" \
  --group garden_k16 "results/higs-round59/r59_garden_720p_k16_s*.json" \
  --group garden_g16 "results/higs-round59/r59_garden_720p_g16_s*.json" \
  --group bicycle_k1  "results/higs-round59/r59_bicycle_720p_k1_s*.json" \
  --group bicycle_k4  "results/higs-round59/r59_bicycle_720p_k4_s*.json" \
  --group bicycle_k16 "results/higs-round59/r59_bicycle_720p_k16_s*.json" \
  --group bicycle_g16 "results/higs-round59/r59_bicycle_720p_g16_s*.json" \
  --group train_720p_full   "results/higs-round56/r56_train_720p_full_s*.json" \
  --group train_720p_eg     "results/higs-round56/r56_train_720p_eg_s*.json" \
  --group garden_720p_full  "results/higs-round56/r56_garden_720p_full_s*.json" \
  --group garden_720p_eg    "results/higs-round56/r56_garden_720p_eg_s*.json" \
  --group bicycle_720p_full "results/higs-round56/r56_bicycle_720p_full_s*.json" \
  --group bicycle_720p_eg   "results/higs-round56/r56_bicycle_720p_eg_s*.json" \
  --reference train_k1   train_720p_full \
  --reference train_k4   train_k1 \
  --reference train_k16  train_k1 \
  --reference train_g16  train_k1 \
  --reference garden_k1  garden_720p_full \
  --reference garden_k4  garden_k1 \
  --reference garden_k16 garden_k1 \
  --reference garden_g16 garden_k1 \
  --reference bicycle_k1  bicycle_720p_full \
  --reference bicycle_k4  bicycle_k1 \
  --reference bicycle_k16 bicycle_k1 \
  --reference bicycle_g16 bicycle_k1