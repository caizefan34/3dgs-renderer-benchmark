#!/bin/bash
# H0-R5K deterministic regeneration of the frozen P12/C0 seed-42 trajectory.
# Emits ckpt_5000/10000/15000/20000/25000.pt for room and bicycle into an
# ISOLATED dir. Canonical tree /root/epic05_n1/runs is NEVER touched.
# Training semantics are byte-identical to the canonical n1snap run; the ONLY
# delta is the added 5000 snapshot, which is save-only (post-step, no RNG/param
# mutation) per the trainer header and the N1 restore-bit-exact audit.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
TRAINER=/root/epic05_n1/publication_trainer_p12_n1snap.py
ROOT=/root/epic05_h0r5k
OUT=$ROOT/runs
LOG=$ROOT/logs
mkdir -p "$OUT" "$LOG"

# args copied verbatim from the canonical P12/C0 launch (see N1_SOURCE_IDENTITY
# + /root/vos_launch_phase4.sh TRAINER_ARGS); only --snapshot-iterations adds 5000.
ARGS="--arm c0 --loss fused --eps2d 0.3 --final-eval all --timing-grade FUNCTIONAL_ONLY --lpips"

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$LOG/gpu_before.csv"
date -u +%FT%TZ >> "$LOG/gpu_before.csv"

echo "[h0r5k] launching room@GPU0 bicycle@GPU1 $(date -u +%FT%TZ)"

setsid nohup $PY "$TRAINER" $ARGS --scene room --iterations 25000 \
  --snapshot-iterations 5000,10000,15000,20000,25000 --gpu 0 \
  --outdir "$OUT/r5k_c0f_s42_room" \
  < /dev/null > "$LOG/r5k_room.log" 2>&1 &
echo "[h0r5k] room pid=$!"

setsid nohup $PY "$TRAINER" $ARGS --scene bicycle --iterations 25000 \
  --snapshot-iterations 5000,10000,15000,20000,25000 --gpu 1 \
  --outdir "$OUT/r5k_c0f_s42_bicycle" \
  < /dev/null > "$LOG/r5k_bicycle.log" 2>&1 &
echo "[h0r5k] bicycle pid=$!"

echo "[h0r5k] both launched $(date -u +%FT%TZ)"
