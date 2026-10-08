#!/bin/bash
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/epic05_grc3d
for S in bicycle garden; do
  O=outputs/remedy_q2_${S}_it5000_cam163_lag16_b8.json
  echo "[remedy] $(date +%H:%M:%S) $S q=1/2"
  PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_probe.py --gpu 4 --backend gsplat     --bundle inputs/bundles/${S}_it5000_cam163_lag16.pt --output $O     --coarse-div 2 --block-size 64 --max-blocks 8 --timing-repeats 10 --seed 20260927     > logs/remedy_q2_${S}.log 2>&1
  echo "[remedy] $S rc=$? done $(date +%H:%M:%S)"
done
echo '[remedy] ALL DONE'
