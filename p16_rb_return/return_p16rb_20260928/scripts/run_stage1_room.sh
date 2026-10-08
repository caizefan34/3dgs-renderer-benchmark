#!/bin/bash
# P16-RB Stage 1: clean FULL room pilot per method, SEQUENTIAL on physical A100 GPU 2.
#
# Rules honoured here:
#  - one physical A100 only per compared run; the same card as the P12 C0 baseline
#  - primary native run: minimal wrapper only (bundle/p16rb_wrap_run.py), no CUDA phase
#    timers inserted into the method
#  - contamination record: GPU UUID + compute processes before/after + external VRAM
#    sampler (monitoring only; it never touches the measured process)
#  - training wall excludes final rendering/eval and checkpoint serialization: for
#    nerficg that is timings.txt `Time:`, for graphdeco the native `Training time:`
#  - graphdeco test rendering runs as a SEPARATE, untimed command afterwards
#  - resolution: common-resolution rerun (-r 1 / IMAGE_SCALE_FACTOR 1.0) so the test
#    renders are 1920x1080 like P12's, per prereg section 5
set -u
D=/root/epic05_p16rb
GPU=2
PY=/root/miniforge3/envs/p16_fastergs/bin/python
GS=/root/miniforge3/envs/gsplat/bin/python
WRAP=$D/bundle/p16rb_wrap_run.py
DS=$D/common_data/mipnerf360/room
S1=$D/logs/stage1.log
: > $S1

log() { echo "[$(date -Is)] $*" | tee -a $S1; }

capture() {  # $1 label  -> $2 = prefix
  local L=$D/logs/stage1_$2
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
    --format=csv,noheader > ${L}.gpu_before.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_before.csv
  uptime > ${L}.uptime_before.txt
}

capture_after() {
  local L=$D/logs/stage1_$2
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total \
    --format=csv,noheader > ${L}.gpu_after.csv
  env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader > ${L}.procs_after.csv
  uptime > ${L}.uptime_after.txt
}

# ---- P12_C0 baseline is reused (triad already re-ran it); no retraining here.

# ---------------------------------------------------------------- 1. FasterGS
M=FasterGS; CWD=$D/src/nerficg; OUT=$D/runs/$M/room
mkdir -p $OUT
capture before $M
$PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/stage1_${M}.vram.csv \
  --duration 28800 > $D/logs/stage1_${M}.sampler.log 2>&1 &
SAMP=$!
log "$M train start"
$PY $WRAP --method $M --scene room --cwd "$CWD" \
  --command "PATH=/root/miniforge3/envs/p16_fastergs/bin:\$PATH python scripts/train.py -c $D/configs/p16_fastergs_room_1920.yaml" \
  --out $OUT/wrapper_train.json --log $D/logs/stage1_${M}_train.stdout.log
log "$M train rc=$?"
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
capture_after after $M
echo "STAGE1_${M}_TRAIN_DONE $(date -Is)" | tee -a $S1

# ---------------------------------------------------------------- 2. FasterGSFused
M=FasterGSFused; CWD=$D/src/nerficg; OUT=$D/runs/$M/room
mkdir -p $OUT
capture before $M
$PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/stage1_${M}.vram.csv \
  --duration 28800 > $D/logs/stage1_${M}.sampler.log 2>&1 &
SAMP=$!
log "$M train start"
$PY $WRAP --method $M --scene room --cwd "$CWD" \
  --command "PATH=/root/miniforge3/envs/p16_fastergs/bin:\$PATH python scripts/train.py -c $D/configs/p16_fastergsfused_room_1920.yaml" \
  --out $OUT/wrapper_train.json --log $D/logs/stage1_${M}_train.stdout.log
log "$M train rc=$?"
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
capture_after after $M
echo "STAGE1_${M}_TRAIN_DONE $(date -Is)" | tee -a $S1

# ---------------------------------------------------------------- 3. FastGS
M=FastGS; CWD=$D/src/FastGS; OUT=$D/runs/$M/room; JOB=p16_${M}_room
mkdir -p $OUT
capture before $M
$PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/stage1_${M}.vram.csv \
  --duration 28800 > $D/logs/stage1_${M}.sampler.log 2>&1 &
SAMP=$!
log "$M train start"
$PY $WRAP --method $M --scene room --cwd "$CWD" \
  --command "CUDA_VISIBLE_DEVICES=$GPU OAR_JOB_ID=$JOB PATH=/root/miniforge3/envs/p16_fastgs/bin:\$PATH python train.py -s $DS -i images --eval -r 1 --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0008" \
  --out $OUT/wrapper_train.json --log $D/logs/stage1_${M}_train.stdout.log
log "$M train rc=$?"
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
capture_after after $M
log "$M render start (untimed)"
$PY $WRAP --method $M --scene room --cwd "$CWD" \
  --command "CUDA_VISIBLE_DEVICES=$GPU PATH=/root/miniforge3/envs/p16_fastgs/bin:\$PATH python render.py -m output/$JOB --skip_train" \
  --out $OUT/wrapper_render.json --log $D/logs/stage1_${M}_render.stdout.log
log "$M render rc=$?"
echo "STAGE1_${M}_DONE $(date -Is)" | tee -a $S1

# ---------------------------------------------------------------- 4. SADGS
M=SADGS; CWD=$D/src/SADGS; OUT=$D/runs/$M/room; JOB=p16_${M}_room
mkdir -p $OUT
capture before $M
$PY $D/scripts/p16_vram_sampler.py --gpu $GPU --out $D/logs/stage1_${M}.vram.csv \
  --duration 28800 > $D/logs/stage1_${M}.sampler.log 2>&1 &
SAMP=$!
log "$M train start"
$PY $WRAP --method $M --scene room --cwd "$CWD" \
  --command "CUDA_VISIBLE_DEVICES=$GPU OAR_JOB_ID=$JOB PATH=/root/miniforge3/envs/p16_sadgs/bin:\$PATH python train.py -s $DS -i images --eval -r 1 --scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 3000 --prune_iterations 1900" \
  --out $OUT/wrapper_train.json --log $D/logs/stage1_${M}_train.stdout.log
log "$M train rc=$?"
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
capture_after after $M
log "$M render start (untimed)"
$PY $WRAP --method $M --scene room --cwd "$CWD" \
  --command "CUDA_VISIBLE_DEVICES=$GPU PATH=/root/miniforge3/envs/p16_sadgs/bin:\$PATH python render.py -m output/$JOB --skip_train --iteration 3000" \
  --out $OUT/wrapper_render.json --log $D/logs/stage1_${M}_render.stdout.log
log "$M render rc=$?"
echo "STAGE1_${M}_DONE $(date -Is)" | tee -a $S1

echo "STAGE1_ALL_DONE $(date -Is)" | tee -a $S1
