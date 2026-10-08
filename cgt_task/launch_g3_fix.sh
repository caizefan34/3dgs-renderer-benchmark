#!/bin/bash
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data
RR=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
OUT=/home/liaoyuanjun/cgt_realscene/final
SCR=$OUT/SCRIPTS
REC=$OUT/G1_RAW_RECORDS
cd "$OUT" || exit 1
g3() { # gpu scene ckrel tag
  local gpu=$1 scene=$2 ckrel=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 setsid nohup "$PY" "$SCR/g3_gpu_microbench.py" \
    --scene "$scene" --ckpt "$RR/$ckrel" --data "$DATA" --views 6 --max-tiles-per-view 80 \
    --ranks 4,8,16 --out "$REC/g3_${tag}.json" \
    > "$OUT/logs/g3r_${tag}.log" 2>&1 < /dev/null &
  echo "launched g3 $tag gpu$gpu pid=$!"
}
# GPU4 (33GB free, heaviest): garden
g3 4 garden B1_garden_30k/checkpoints/iter_30000.pt garden_r30k
# GPU0 (15GB free): bicycle then room_r30k sequential
{ g3 0 bicycle B1_bicycle_30k/checkpoints/iter_30000.pt bicycle_r30k; sleep 3; g3 0 room B1_room_30k/checkpoints/iter_30000.pt room_r30k; } &
# GPU3 (18GB free): room_r5k
g3 3 room B1_room_multiphase_v2/checkpoints/iter_5000.pt room_r5k
wait
echo "ALL_G3_RELAUNCHED"
