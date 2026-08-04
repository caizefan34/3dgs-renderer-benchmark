#!/bin/bash
# Round-65 exp3: garden 1080p schedule knobs for the prog x decay cell.
#   ctrl s0 (full-res+MA, wave anchor) + pd05 s0 (0.5:0,1.0:1500, wave anchor)
#   pd075 s0/s1/s2 (0.75:0,1.0:1500) + pdramp s0/s1/s2 (0.5:0,0.75:1000,1.0:1500)
set -u
export PATH=/usr/local/cuda/bin:/root/miniforge3/envs/gsplat/bin:/usr/bin:/bin
export GSPLAT_SKIP_FROM_WORLD=1
export PYTHONPATH=/root/3dgs-roadmap-matrix/artifacts/renderer-sources/gsplat
cd /root/3dgs-roadmap-matrix
PY=/root/miniforge3/envs/gsplat/bin/python
BENCH=benchmark/run_higs_train_benchmark.py
BASE=/root/epic05-data/processed
OUT=/tmp/qsR65exp3
mkdir -p $OUT

run () {
  local gpu=$1 seed=$2 sched=$3 decay=$4 tag=$5
  local extra="--anchor-densify --anchor-densify-every 2"
  local sflag=""; local dflag=""
  if [ "$sched" != "none" ]; then sflag="--res-schedule $sched"; fi
  if [ "$decay" = "1" ]; then dflag="--masked-adam-union-decay 0.99 --masked-adam-union-decay-eval-proj"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY $BENCH --base-dir $BASE --scene mipnerf360/garden --backends higs_dynamic_ts \
    --n-train 4 --n-eval 3 --steps 3000 --width 1920 --height 1080 --seed $seed \
    --tile-sampling-ratio 0.35 --sampling-mode error_guided --error-alpha 1.0 \
    --error-refresh-every 25 --error-lambda 0.7 --eval-every 300 \
    --lr-decay 0.1 --densify-window 1500 \
    --lpips-loss-weight 0.1 --lpips-loss-every 25 --lpips-full-res \
    --cull-interval 1 --masked-adam $dflag $sflag \
    $extra --out $OUT/$tag.json > $OUT/$tag.log 2>&1
  echo "DONE $tag rc=$?" >> $OUT/rc.txt
}
run 0 0 none 0 garden_ctrl_s0 &
run 1 0 "0.5:0,1.0:1500" 1 garden_pd05_s0 &
run 2 0 "0.75:0,1.0:1500" 1 garden_pd075_s0 &
run 3 1 "0.75:0,1.0:1500" 1 garden_pd075_s1 &
run 4 2 "0.75:0,1.0:1500" 1 garden_pd075_s2 &
run 5 0 "0.5:0,0.75:1000,1.0:1500" 1 garden_pdramp_s0 &
run 6 1 "0.5:0,0.75:1000,1.0:1500" 1 garden_pdramp_s1 &
run 7 2 "0.5:0,0.75:1000,1.0:1500" 1 garden_pdramp_s2 &
wait
echo R65EXP3_DONE
cat $OUT/rc.txt