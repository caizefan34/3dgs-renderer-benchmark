#!/bin/bash
# Round 59 (step-breakdown profiling): run torch.profiler over 40 training
# steps of the HiGS backend at 4 operating points and dump kernel-level
# CUDA self-time tables. Purpose: locate the dominant per-step kernels so the
# next acceleration lever is evidence-based (after R57/R58 closed the
# sampling / resolution / renderer-pixel levers).
#
# Operating points (one A100 each, GPUs 0-3):
#   prof_garden_720p_r035  garden 1280x720  r=0.35 (speed operating point)
#   prof_garden_720p_r100  garden 1280x720  r=1.00 (full-tile reference)
#   prof_garden_1080p_r100 garden 1920x1080 r=1.00 (full-res reference)
#   prof_train_720p_r035   train  1280x720  r=0.35 (low-N scene)
set -u
export PATH=/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
OUT=/tmp/qsR59prof
mkdir -p $OUT

run_prof () {
  local gpu=$1 scene=$2 w=$3 h=$4 ratio=$5 tag=$6
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/higs/profile_step_breakdown.py \
    --base-dir /root/epic05-data/processed --scene "$scene" \
    --width $w --height $h --ratio $ratio --steps 40 \
    > $OUT/prof_${tag}.txt 2>&1
  echo "DONE $tag rc=$?"
}

run_prof 0 mipnerf360/garden 1280 720 0.35 garden_720p_r035 &
run_prof 1 mipnerf360/garden 1280 720 1.0  garden_720p_r100 &
run_prof 2 mipnerf360/garden 1920 1080 1.0 garden_1080p_r100 &
run_prof 3 tanks_and_temples/train 1280 720 0.35 train_720p_r035 &
wait
echo ALL_DONE_PROF