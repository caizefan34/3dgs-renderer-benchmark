#!/bin/bash
# VOS Phase-9: 30K protocol runs. Per scene: instrumented C0 baseline
# (--vos-start 99999: proxy + bitmap capture active, zero VOS steps) THEN the
# VOS arm (15K->30K), SEQUENTIALLY on the same GPU for honest timing.
# Quality gates (Phase-11) compare PSNR/SSIM/LPIPS within the twin envelope.
set -u
export PY=/root/miniforge3/envs/gsplat/bin/python
export D=/root/vos_traces
export R=$D/vos_runs
cd "$D"

run_scene () {
  local SCENE=$1 GPU=$2
  echo "[$(date -u +%FT%TZ)] $SCENE C0 baseline (GPU $GPU)"
  # Exec-time CVD is REQUIRED here: runtime os.environ setenv inside the
  # driver does not pin in this container (empirically confirmed 2026-09-30;
  # exec-time env does pin — vos_cvd_probe2.sh).
  CUDA_VISIBLE_DEVICES=$GPU $PY vos_train_driver.py --scene "$SCENE" \
    --iterations 30000 --gpu "$GPU" \
    --vos-start 99999 --final-eval all \
    --outdir "$R"/c0_"$SCENE"_30k > "$D"/logs/vos_c0_"$SCENE"_30k.log 2>&1
  echo "[$(date -u +%FT%TZ)] $SCENE C0 done (exit $?)"
  echo "[$(date -u +%FT%TZ)] $SCENE VOS arm (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU $PY vos_train_driver.py --scene "$SCENE" \
    --iterations 30000 --gpu "$GPU" \
    --vos-start 15000 --final-eval all \
    --outdir "$R"/vos_"$SCENE"_30k > "$D"/logs/vos_vos_"$SCENE"_30k.log 2>&1
  echo "[$(date -u +%FT%TZ)] $SCENE VOS done (exit $?)"
}

# three scenes in parallel on three free GPUs (each scene sequential within)
setsid nohup bash -c "$(declare -f run_scene); run_scene room 3" \
  < /dev/null > "$D"/logs/scene_room_30k.log 2>&1 &
echo "ROOM pid=$!"
setsid nohup bash -c "$(declare -f run_scene); run_scene bicycle 4" \
  < /dev/null > "$D"/logs/scene_bicycle_30k.log 2>&1 &
echo "BICYCLE pid=$!"
setsid nohup bash -c "$(declare -f run_scene); run_scene garden 5" \
  < /dev/null > "$D"/logs/scene_garden_30k.log 2>&1 &
echo "GARDEN pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
