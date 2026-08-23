#!/bin/bash
# Round-61 exp7: full LPIPS work-size sweep - 3 scenes x 3 seeds, --lpips-work-size 256,
# identical to round-60 config (r60 = ws0 control, already in results/higs-round60).
# Waves: 8 GPUs (wave1) + 1 GPU (wave2).
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR61exp7
mkdir -p $OUT

run_dyn () {
  local gpu=$1 scene=$2 seed=$3 tag=$4
  local extra=""
  if [ "$scene" != "tanks_and_temples/train" ]; then extra="--anchor-densify --anchor-densify-every 2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene "$scene" --backends higs_dynamic_ts     --n-train 4 --n-eval 3 --steps 3000 --width 1280 --height 720 --seed $seed     --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0     --error-refresh-every 25 --error-lambda 0.7 --eval-every 300     --lr-decay 0.1 --densify-window 1500     --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res --lpips-work-size 256     --cull-interval 1 --masked-adam     $extra     --out $OUT/${tag}.json > $OUT/${tag}.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}

# Wave 1: garden s0..2, bicycle s0..2, train s0..1 -> 8 jobs
i=0
for scene in mipnerf360/garden mipnerf360/bicycle; do
  for seed in 0 1 2; do
    sname=$(basename "$scene")
    run_dyn $i "$scene" $seed "r61_${sname}_720p_ws256_s${seed}" &
    i=$((i + 1))
  done
done
for seed in 0 1; do
  run_dyn $i tanks_and_temples/train $seed "r61_train_720p_ws256_s${seed}" &
  i=$((i + 1))
done
wait
echo WAVE1_DONE
# Wave 2: train s2
run_dyn 0 tanks_and_temples/train 2 "r61_train_720p_ws256_s2"
wait
echo EXP7_DONE
cat $OUT/rc.txt