#!/bin/bash
# CGT-PAPER13 P1 driver: 13-scene canonical gsplat full-frame baseline (tile16 + tile32).
set -u
DEV=${1:-1}
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360
OUT=/mnt/storage_pool/liaoyuanjun/cgt_paper13/results/p1
mkdir -p "$OUT"
SRCDIR=/mnt/storage_pool/liaoyuanjun/cgt_paper13/src
B1=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
C42=/mnt/storage_pool/liaoyuanjun/c42_13scene_results
declare -A CKPT
CKPT[bicycle]=$B1/B1_bicycle_30k/checkpoints/iter_30000.pt
CKPT[garden]=$B1/B1_garden_30k/checkpoints/iter_30000.pt
CKPT[room]=$B1/B1_room_multiphase_v2/checkpoints/iter_15000.pt
CKPT[bonsai]=$C42/mipnerf360/bonsai/reference/checkpoints/iter_30000.pt
CKPT[counter]=$C42/mipnerf360/counter/reference/checkpoints/iter_30000.pt
CKPT[flowers]=$C42/mipnerf360/flowers/reference/checkpoints/iter_30000.pt
CKPT[kitchen]=$C42/mipnerf360/kitchen/reference/checkpoints/iter_30000.pt
CKPT[stump]=$C42/mipnerf360/stump/reference/checkpoints/iter_30000.pt
CKPT[treehill]=$C42/mipnerf360/treehill/reference/checkpoints/iter_30000.pt
CKPT[train]=$C42/tanksandtemples/train/reference/checkpoints/iter_30000.pt
CKPT[truck]=$C42/tanksandtemples/truck/reference/checkpoints/iter_30000.pt
CKPT[drjohnson]=$C42/deepblending/drjohnson/reference/checkpoints/iter_30000.pt
CKPT[playroom]=$C42/deepblending/playroom/reference/checkpoints/iter_30000.pt
ORDER="room garden bicycle bonsai counter flowers kitchen stump treehill train truck drjohnson playroom"
echo "[P1] device=$DEV start $(date)"
for s in $ORDER; do
  echo "=== [P1] $s $(date) ==="
  $PY $SRCDIR/paper13_canonical.py --scene "$s" --ckpt "${CKPT[$s]}" --data-root "$DATA" \
     --device "$DEV" --tile-sizes 16,32 --repeats 30 --warmup 5 --n-test-views 12 --split-every 8 --out "$OUT" 2>&1 | grep -vE "Warning|non-writable" | tail -3
done
echo "[P1] DONE $(date)"