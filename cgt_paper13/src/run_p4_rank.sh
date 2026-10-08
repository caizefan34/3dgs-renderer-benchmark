#!/bin/bash
set -u
DEV=${1:-1}
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data
OUT=/mnt/storage_pool/liaoyuanjun/cgt_paper13/results/p4_rank
mkdir -p "$OUT"
B1=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
declare -A CKPT
CKPT[room]=$B1/B1_room_multiphase_v2/checkpoints/iter_15000.pt
CKPT[garden]=$B1/B1_garden_30k/checkpoints/iter_30000.pt
CKPT[bicycle]=$B1/B1_bicycle_30k/checkpoints/iter_30000.pt
echo "[P4-rank] device=$DEV start $(date)"
for s in room garden bicycle; do
  for r in 4 8 16; do
    echo "=== $s R=$r $(date) ==="
    $PY /mnt/storage_pool/liaoyuanjun/cgt_paper13/src/paper13_cgt.py --scene "$s" --ckpt "${CKPT[$s]}" \
       --data-root "$DATA" --device "$DEV" --rank "$r" --views 5 --tiles-per-view 10 \
       --repeats 80 --warmup 15 --out "$OUT" 2>&1 | grep -vE 'Warning|wlo = np|overflow|divide|invalid value' | tail -1
  done
done
echo "[P4-rank] DONE $(date)"