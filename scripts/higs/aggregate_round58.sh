#!/bin/bash
# Aggregate the round-58 speed-max matrix into r58-summary.json.
# Arms: *_prog720 (720p target + progressive-res + eg) vs the round-56
# same-resolution reference (720p full). deltas/speedup are vs 720p full
# (matching the R56 table convention); the vs-plain-eg comparison is
# computed in the report (0.89-1.09x, i.e. progressive adds no speed at
# 720p -- the R58 negative finding).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
$PY scripts/higs/aggregate_run_summary.py \
  --out results/higs-round58/r58-summary.json \
  --group train_prog720   "results/higs-round58/r58_train_prog720_s*.json" \
  --group garden_prog720  "results/higs-round58/r58_garden_prog720_s*.json" \
  --group bicycle_prog720 "results/higs-round58/r58_bicycle_prog720_s*.json" \
  --group train_720p_full   "results/higs-round56/r56_train_720p_full_s*.json" \
  --group train_720p_eg     "results/higs-round56/r56_train_720p_eg_s*.json" \
  --group garden_720p_full  "results/higs-round56/r56_garden_720p_full_s*.json" \
  --group garden_720p_eg    "results/higs-round56/r56_garden_720p_eg_s*.json" \
  --group bicycle_720p_full "results/higs-round56/r56_bicycle_720p_full_s*.json" \
  --group bicycle_720p_eg   "results/higs-round56/r56_bicycle_720p_eg_s*.json" \
  --reference train_prog720 train_720p_full \
  --reference garden_prog720 garden_720p_full \
  --reference bicycle_prog720 bicycle_720p_full