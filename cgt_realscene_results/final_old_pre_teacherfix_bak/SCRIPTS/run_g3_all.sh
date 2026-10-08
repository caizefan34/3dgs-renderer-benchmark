#!/bin/bash
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data
RUNROOT=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
OUT=/home/liaoyuanjun/cgt_realscene/final
REC=$OUT/G1_RAW_RECORDS
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=4
cd "$OUT/SCRIPTS"
run() {  # scene ckpt outjson
  echo "=== G3 $1 ==="
  $PY g3_gpu_microbench.py --scene "$1" --ckpt "$2" --data "$DATA" \
     --views 6 --max-tiles-per-view 80 --ranks 4,8,16 --out "$3" 2>&1
}
run garden   "$RUNROOT/B1_garden_30k/checkpoints/iter_30000.pt"            "$REC/g3_garden_r30k.json"
run bicycle  "$RUNROOT/B1_bicycle_30k/checkpoints/iter_30000.pt"           "$REC/g3_bicycle_r30k.json"
run room     "$RUNROOT/B1_room_multiphase_v2/checkpoints/iter_5000.pt"     "$REC/g3_room_r5k.json"
run room     "$RUNROOT/B1_room_30k/checkpoints/iter_30000.pt"              "$REC/g3_room_r30k.json"
echo "=== ALL G3 DONE ==="
