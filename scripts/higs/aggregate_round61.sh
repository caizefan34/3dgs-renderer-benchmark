#!/bin/bash
# Aggregate the round-61 LPIPS work-size sweep into r61-summary.json.
# 9 runs: 3 scenes x 3 seeds, --lpips-work-size 256 on the round-60 k1 masked-Adam
# 720p eg recipe. Reference = round-60 masked-Adam runs (identical config, ws=0),
# so speedup_train/deltas isolate the work-size lever only.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
$PY scripts/higs/aggregate_run_summary.py \
  --out results/higs-round61/r61-summary.json \
  --group train_ws256   "results/higs-round61/r61_train_720p_ws256_s*.json" \
  --group garden_ws256  "results/higs-round61/r61_garden_720p_ws256_s*.json" \
  --group bicycle_ws256 "results/higs-round61/r61_bicycle_720p_ws256_s*.json" \
  --group train_ma      "results/higs-round60/r60_train_720p_ma_s*.json" \
  --group garden_ma     "results/higs-round60/r60_garden_720p_ma_s*.json" \
  --group bicycle_ma    "results/higs-round60/r60_bicycle_720p_ma_s*.json" \
  --reference train_ws256   train_ma \
  --reference garden_ws256  garden_ma \
  --reference bicycle_ws256 bicycle_ma