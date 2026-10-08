#!/bin/bash
# CGT-PAPER13 P3 driver: E2E build-amortization oracle, 13 scenes, per-frame (views=1).
set -u
DEV=${1:-1}
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data
OUT=/mnt/storage_pool/liaoyuanjun/cgt_paper13/results/p3
mkdir -p "$OUT"
B1=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
C42=/mnt/storage_pool/liaoyuanjun/c42_13scene_results
declare -A CKPT CANON
CKPT[bicycle]=$B1/B1_bicycle_30k/checkpoints/iter_30000.pt; CANON[bicycle]=17.797
CKPT[garden]=$B1/B1_garden_30k/checkpoints/iter_30000.pt; CANON[garden]=11.190
CKPT[room]=$B1/B1_room_multiphase_v2/checkpoints/iter_15000.pt; CANON[room]=11.507
CKPT[bonsai]=$C42/mipnerf360/bonsai/reference/checkpoints/iter_30000.pt; CANON[bonsai]=1.219
CKPT[counter]=$C42/mipnerf360/counter/reference/checkpoints/iter_30000.pt; CANON[counter]=4.619
CKPT[flowers]=$C42/mipnerf360/flowers/reference/checkpoints/iter_30000.pt; CANON[flowers]=23.495
CKPT[kitchen]=$C42/mipnerf360/kitchen/reference/checkpoints/iter_30000.pt; CANON[kitchen]=1.091
CKPT[stump]=$C42/mipnerf360/stump/reference/checkpoints/iter_30000.pt; CANON[stump]=10.144
CKPT[treehill]=$C42/mipnerf360/treehill/reference/checkpoints/iter_30000.pt; CANON[treehill]=30.513
CKPT[train]=$C42/tanksandtemples/train/reference/checkpoints/iter_30000.pt; CANON[train]=4.237
CKPT[truck]=$C42/tanksandtemples/truck/reference/checkpoints/iter_30000.pt; CANON[truck]=3.900
CKPT[drjohnson]=$C42/deepblending/drjohnson/reference/checkpoints/iter_30000.pt; CANON[drjohnson]=2.095
CKPT[playroom]=$C42/deepblending/playroom/reference/checkpoints/iter_30000.pt; CANON[playroom]=2.916
ORDER="room garden bicycle bonsai counter flowers kitchen stump treehill train truck drjohnson playroom"
echo "[P3] device=$DEV start $(date)"
for s in $ORDER; do
  echo "=== [P3] $s canon=${CANON[$s]}ms $(date) ==="
  $PY /mnt/storage_pool/liaoyuanjun/cgt_paper13/src/paper13_e2e.py --scene "$s" --ckpt "${CKPT[$s]}" \
     --data-root "$DATA" --device "$DEV" --canonical-ms "${CANON[$s]}" --views 1 --n-sample 50 --out "$OUT" 2>&1 | grep -vE 'Warning|wlo = np|overflow|divide|invalid value' | tail -2
done
echo "[P3] DONE $(date)"