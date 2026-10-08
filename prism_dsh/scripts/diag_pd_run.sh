#!/bin/bash
# Non-PD covariance diagnostic for both garden snapshots (read-only).
set -u
cd /root/epic05_prism/prism_gs_v0.1.1 || exit 3
PY=/root/miniforge3/envs/gsplat/bin/python
for st in 005000 014000; do
  echo "=== garden ${st} $(date '+%T') ==="
  CUDA_VISIBLE_DEVICES=2 "$PY" -u ../scripts/diag_pd_v1.py \
    --scene garden --state "$st" \
    --snapshot "/root/epic05_grc3d/inputs/snapshots/snap_garden_${st}.pt" \
    --out "../outputs/diag_pd_garden_${st}.json"
  echo "exit=$? $(date '+%T')"
done
echo DIAG_PD_DONE
