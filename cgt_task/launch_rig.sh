#!/bin/bash
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data
RR=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
OUT=/home/liaoyuanjun/cgt_realscene/final
SCR=$OUT/SCRIPTS
CERT=$OUT/G2_CERT
cd "$OUT" || exit 1
rig() {
  local gpu=$1 scene=$2 ckrel=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 setsid nohup "$PY" "$SCR/g2_rigorous_certificate.py" \
    --scene "$scene" --ckpt "$RR/$ckrel" --data "$DATA" --views 2 --max-tiles-per-view 12 \
    --ranks 4,8,16 --iterations 200 --min-size 1.0 --max-depth 10 \
    --out "$CERT/G2_RIGOROUS_${tag}.json" \
    > "$OUT/logs/rigorous_${tag}.log" 2>&1 < /dev/null &
  echo "launched rigorous $tag gpu$gpu pid=$!"
}
rig 5 room B1_room_multiphase_v2/checkpoints/iter_15000.pt room_r15k
rig 6 garden B1_garden_30k/checkpoints/iter_30000.pt garden_r30k
rig 7 bicycle B1_bicycle_30k/checkpoints/iter_30000.pt bicycle_r30k
echo "ALL_RIG_LAUNCHED"
