#!/bin/bash
# VOS Phase-5 round 3b: post-opacity-reset oracle test with a VALID in-process
# pair — room_T15100's own ckpt_15100 + its own trace segment 15001..15100.
# No topology events run in (15000, 15100], so segment N == ckpt N.
# The ckpt state includes the REAL opacity reset at 15000 (opacity m/v zeroed,
# opacity params reset to the init value) — a true reset edge case, not synthetic.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
O=$D/oracle
cd "$D"
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T15100 --it-lo 15001 --it-hi 15100 \
  --init-mode ckpt --ckpt "$D"/room_T15100/ckpt_15100.pt --n-rows 100000 --seed 7 \
  --obs-points 10,37,99 --fence-points 50 \
  --outdir "$O" --tag room_T15100_postreset < /dev/null > "$D"/logs/oracle_O5b2.log 2>&1 &
echo "O5b2 pid=$!"
echo "LAUNCHED $(date -u +%FT%TZ)"
