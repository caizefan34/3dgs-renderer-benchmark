#!/bin/bash
# Aggregate the round-60 cull-masked-Adam sweep into r60-summary.json.
# 9 runs: 3 scenes x 3 seeds, k1 + --masked-adam on the 720p eg recipe.
# The k1 baseline reuses round-59 k1 runs (identical config; the
# masked_adam=False path is bit-identical to the round-59 path), and the
# historical k1-vs-720p-full reference reuses R56 same-resolution runs for
# the cross-round speedup convention.  The masked-Adam lever changes the
# optimizer step only, so the per-step wall comparison uses train_ms (which
# includes the optimizer) and speedup_train in addition to total_ms.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
$PY scripts/higs/aggregate_run_summary.py   --out results/higs-round60/r60-summary.json   --group train_k1   "results/higs-round59/r59_train_720p_k1_s*.json"   --group train_ma   "results/higs-round60/r60_train_720p_ma_s*.json"   --group garden_k1  "results/higs-round59/r59_garden_720p_k1_s*.json"   --group garden_ma  "results/higs-round60/r60_garden_720p_ma_s*.json"   --group bicycle_k1  "results/higs-round59/r59_bicycle_720p_k1_s*.json"   --group bicycle_ma  "results/higs-round60/r60_bicycle_720p_ma_s*.json"   --group train_720p_full   "results/higs-round56/r56_train_720p_full_s*.json"   --group garden_720p_full  "results/higs-round56/r56_garden_720p_full_s*.json"   --group bicycle_720p_full "results/higs-round56/r56_bicycle_720p_full_s*.json"   --reference train_ma   train_k1   --reference garden_ma  garden_k1   --reference bicycle_ma bicycle_k1   --reference train_k1   train_720p_full   --reference garden_k1  garden_720p_full   --reference bicycle_k1 bicycle_720p_full
