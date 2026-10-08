#!/bin/bash
PY=/root/miniforge3/envs/gsplat/bin/python
GPU=$1; shift
cd /root/epic05_grc3d
for PAIR in "$@"; do
  S=$(echo $PAIR | cut -d: -f1); IT=$(echo $PAIR | cut -d: -f2)
  SNAP=$(printf 'inputs/snapshots/snap_%s_%06d.pt' $S $IT)
  O=outputs/cost_profile_${S}_it${IT}_cam163.json
  C=outputs/cost_profile_${S}_it${IT}_cam163.csv
  echo "[cost] $(date +%H:%M:%S) $S it$IT"
  PYTHONDONTWRITEBYTECODE=1 $PY scripts/cost_profile.py --gpu $GPU --scene $S     --bundle inputs/bundles/${S}_it${IT}_cam163_lag16.pt --snapshot $SNAP     --camera-index 163 --camera-seq outputs/replay_${S}_14000/camera_sequence.npy     --repeats 30 --out $O --csv $C > logs/cost_profile_${S}_it${IT}.log 2>&1
  echo "[cost] $S it$IT rc=$? done $(date +%H:%M:%S)"
done
echo '[cost] ALL DONE'
