#!/bin/bash
# Order-B re-pairs for bicycle and garden on GPU 3 — the only GPU the MR0
# co-tenant leaves free (its shards hold GPUs 0,1,2,4,5,6,7). MR0's CPU load
# is a steady sweep, so it affects both arms of each pair symmetrically
# (pair-internal fairness). VOS arm first, then C0, scenes sequential.
set -u
export PY=/root/miniforge3/envs/gsplat/bin/python
export D=/root/vos_traces
export R=$D/vos_runs
cd "$D"
(
  for SC in bicycle garden; do
    echo "[$(date -u +%FT%TZ)] $SC re-pair: VOS first (GPU 3, MR0 co-tenant CPU load)"
    CUDA_VISIBLE_DEVICES=3 $PY vos_train_driver.py --scene "$SC" \
      --iterations 30000 --gpu 3 --vos-start 15000 --final-eval all \
      --outdir "$R"/vos_"$SC"_30k_b > "$D"/logs/vos_vos_"$SC"_30k_b.log 2>&1
    echo "[$(date -u +%FT%TZ)] $SC VOS_b done (exit $?)"
    echo "[$(date -u +%FT%TZ)] $SC re-pair: C0 second (GPU 3)"
    CUDA_VISIBLE_DEVICES=3 $PY vos_train_driver.py --scene "$SC" \
      --iterations 30000 --gpu 3 --vos-start 99999 --final-eval all \
      --outdir "$R"/c0_"$SC"_30k_b > "$D"/logs/vos_c0_"$SC"_30k_b.log 2>&1
    echo "[$(date -u +%FT%TZ)] $SC C0_b done (exit $?)"
  done
) > "$D"/logs/scene_clean_repairs.log 2>&1 &
echo "REPAIRS-GPU3 pid=$! LAUNCHED $(date -u +%FT%TZ)"
