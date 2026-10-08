#!/bin/bash
# run_g1.sh <GPU> <comma-list-of-combos>
# combo = scene:ckpt_iter:tag  e.g. room:5000:r5k
set -u
GPU="$1"; COMBOS="$2"
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
ROOT=/home/liaoyuanjun/cgt_realscene
SRC=$ROOT/src
OUT=$ROOT/logs/g1
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
DATA=/tmp/3dgs-renderer-benchmark/data
RUNROOT=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
IFS=',' read -ra ARR <<< "$COMBOS"
for combo in "${ARR[@]}"; do
  IFS=':' read -r scene iter tag <<< "$combo"
  if [ "$scene" = "room" ]; then
    if [ "$iter" = "30000" ]; then CKPT="$RUNROOT/B1_room_30k/checkpoints/iter_30000.pt";
    else CKPT="$RUNROOT/B1_room_multiphase_v2/checkpoints/iter_${iter}.pt"; fi
  else
    CKPT="$RUNROOT/B1_${scene}_30k/checkpoints/iter_30000.pt"
  fi
  OUTJSON="$OUT/g1_${scene}_${tag}.json"
  echo "[GPU$GPU] $combo -> $OUTJSON (ckpt=$CKPT)"
  timeout 1800 $PY "$SRC/g1_compressibility.py" --scene "$scene" --ckpt "$CKPT" \
     --data "$DATA" --views 6 --iterations 200 --max-tiles-per-view 250 \
     --seed 20261004 --out "$OUTJSON" \
     > "$OUT/g1_${scene}_${tag}.log" 2>&1
  echo "[GPU$GPU] $combo done rc=$? ; records=$(grep -o '\"n_records\": [0-9]*' "$OUTJSON" 2>/dev/null | head -1)"
done
echo "[GPU$GPU] ALL DONE"