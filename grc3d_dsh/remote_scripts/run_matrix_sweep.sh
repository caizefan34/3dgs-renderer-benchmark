#!/bin/bash
# 32-block matrix sweep: one scene per invocation, sequential bundles.
SCOPE=$1; GPU=$2
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/epic05_grc3d
for IT in 5000 14000; do
  for CAM in 163 28; do
    for LAG in 0 16 64; do
      STEM=${SCOPE}_it${IT}_cam${CAM}_lag${LAG}
      B=inputs/bundles/${STEM}.pt
      O=outputs/probe_${STEM}_b32.json
      if [ -f $O ]; then echo "[skip] $O exists"; continue; fi
      echo "[sweep] $(date +%H:%M:%S) $STEM"
      PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_probe.py --gpu $GPU --backend gsplat         --bundle $B --output $O --coarse-div 4 --block-size 64 --max-blocks 32         --timing-repeats 20 --seed 20260927 > logs/probe_${STEM}_b32.log 2>&1
      echo "[sweep] $STEM rc=$? done $(date +%H:%M:%S)"
    done
  done
done
echo "[sweep] $SCOPE ALL DONE"
