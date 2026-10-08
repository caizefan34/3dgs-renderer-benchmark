#!/bin/bash
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data
RR=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
OUT=/home/liaoyuanjun/cgt_realscene/final
SCR=$OUT/SCRIPTS
REC=$OUT/G1_RAW_RECORDS
CERT=$OUT/G2_CERT
cd "$OUT" || exit 1
dense() { # gpu scene ckrel tag
  local gpu=$1 scene=$2 ckrel=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 setsid nohup "$PY" "$SCR/g2_certificate.py" \
    --scene "$scene" --ckpt "$RR/$ckrel" --data "$DATA" --views 2 --max-tiles-per-view 12 \
    --ranks 4,8,16 --sub 4 --iterations 200 --out "$CERT/G2_CONTINUOUS_DOMAIN_CERTIFICATE_${tag}.json" \
    > "$OUT/logs/dense_${tag}.log" 2>&1 < /dev/null &
  echo "launched dense $tag gpu$gpu pid=$!"
}
g3() { # gpu scene ckrel tag
  local gpu=$1 scene=$2 ckrel=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 setsid nohup "$PY" "$SCR/g3_gpu_microbench.py" \
    --scene "$scene" --ckpt "$RR/$ckrel" --data "$DATA" --views 6 --max-tiles-per-view 80 \
    --ranks 4,8,16 --out "$REC/g3_${tag}.json" \
    > "$OUT/logs/g3r_${tag}.log" 2>&1 < /dev/null &
  echo "launched g3 $tag gpu$gpu pid=$!"
}
# GPU3: dense room -> g3 room_r5k
{ dense 3 room B1_room_multiphase_v2/checkpoints/iter_15000.pt room_r15k; sleep 2; g3 3 room B1_room_multiphase_v2/checkpoints/iter_5000.pt room_r5k; } &
# GPU5: dense garden -> g3 room_r30k
{ dense 5 garden B1_garden_30k/checkpoints/iter_30000.pt garden_r30k; sleep 2; g3 5 room B1_room_30k/checkpoints/iter_30000.pt room_r30k; } &
# GPU6: dense bicycle -> g3 garden
{ dense 6 bicycle B1_bicycle_30k/checkpoints/iter_30000.pt bicycle_r30k; sleep 2; g3 6 garden B1_garden_30k/checkpoints/iter_30000.pt garden_r30k; } &
# GPU7: g3 bicycle
g3 7 bicycle B1_bicycle_30k/checkpoints/iter_30000.pt bicycle_r30k
wait
echo "ALL_DENSE_G3_LAUNCHED"
