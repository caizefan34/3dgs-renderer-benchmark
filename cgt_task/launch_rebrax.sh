#!/bin/bash
# Launch all G1 + G2-rigorous re-run jobs (fixed teacher). Idempotent-ish: backgrounds each with setsid nohup.
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
DATA=/tmp/3dgs-renderer-benchmark/data
RR=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
OUT=/home/liaoyuanjun/cgt_realscene/final
SCR=$OUT/SCRIPTS
REC=$OUT/G1_RAW_RECORDS
CERT=$OUT/G2_CERT
mkdir -p "$OUT/logs"
cd "$OUT" || exit 1

g1() { # gpu scene ckrel tag
  local gpu=$1 scene=$2 ckrel=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 setsid nohup "$PY" "$SCR/g1_compressibility.py" \
    --scene "$scene" --ckpt "$RR/$ckrel" --data "$DATA" --views 6 --iterations 200 \
    --max-tiles-per-view 250 --seed 20261004 --out "$REC/g1_${scene}_${tag}.json" \
    > "$OUT/logs/g1_${tag}.log" 2>&1 < /dev/null &
  echo "launched g1 $tag on gpu$gpu pid=$!"
}

rig() { # scene ckrel tag
  local scene=$1 ckrel=$2 tag=$3
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 setsid nohup "$PY" "$SCR/g2_rigorous_certificate.py" \
    --scene "$scene" --ckpt "$RR/$ckrel" --data "$DATA" --views 2 --max-tiles-per-view 12 \
    --ranks 4,8,16 --iterations 200 --min-size 1.0 --max-depth 10 \
    --out "$CERT/G2_RIGOROUS_${tag}.json" \
    > "$OUT/logs/rigorous_${tag}.log" 2>&1 < /dev/null &
  echo "launched rigorous $tag on gpu3 pid=$!"
}

# G1: garden already running on gpu0 (g1test). Add the rest.
g1 0 bicycle B1_bicycle_30k/checkpoints/iter_30000.pt bicycle
g1 1 room B1_room_multiphase_v2/checkpoints/iter_5000.pt room_r5k
g1 1 room B1_room_multiphase_v2/checkpoints/iter_10000.pt room_r10k
g1 2 room B1_room_multiphase_v2/checkpoints/iter_15000.pt room_r15k
g1 2 room B1_room_30k/checkpoints/iter_30000.pt room_r30k
# rigorous chain on gpu3
rig room B1_room_multiphase_v2/checkpoints/iter_15000.pt room_r15k
rig garden B1_garden_30k/checkpoints/iter_30000.pt garden_r30k
rig bicycle B1_bicycle_30k/checkpoints/iter_30000.pt bicycle_r30k
echo "ALL_LAUNCHED"
