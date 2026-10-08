#!/bin/bash
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/epic05_grc3d
for SCENE in bicycle garden; do
  for IT in 5000 14000; do
    SNAP=$(printf 'inputs/snapshots/snap_%s_%06d.pt' $SCENE $IT)
    for LAG in 0 16 64; do
      STEM=${SCENE}_it${IT}_cam163_lag${LAG}
      O=outputs/precond_${STEM}_curv_b32.json
      if [ -f $O ]; then echo "[skip] $O"; continue; fi
      echo "[precond] $(date +%H:%M:%S) $STEM"
      PYTHONDONTWRITEBYTECODE=1 $PY scripts/probe_precond.py --gpu 5         --bundle inputs/bundles/${STEM}.pt --snapshot $SNAP         --coarse-div 4 --block-size 64 --max-blocks 32 --seed 20260927         --out $O > logs/precond_${STEM}.log 2>&1
      echo "[precond] $STEM rc=$? done $(date +%H:%M:%S)"
    done
  done
done
echo '[precond] ALL DONE'
