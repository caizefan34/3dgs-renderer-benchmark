#!/bin/bash
# VOS Phase-5 round 3: densify-region + post-reset oracle tests using the
# T5/T15 traced runs' own checkpoints (in-process trace+ckpt pairing — the
# R-run ckpts diverged from the T30 trajectory via atomic nondeterminism).
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
O=$D/oracle
ls -la "$D"/room_T5/ckpt_5000.pt "$D"/room_T15/ckpt_15000.pt
cd "$D"

# O4b: densify-region stretch 5001..5100, T5's own ckpt_5000 (N must match)
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T5 --it-lo 5001 --it-hi 5100 \
  --init-mode ckpt --ckpt "$D"/room_T5/ckpt_5000.pt --n-rows 100000 --seed 7 \
  --obs-points 10,37,99 --fence-points 50 \
  --outdir "$O" --tag room_T5_densify < /dev/null > "$D"/logs/oracle_O4b.log 2>&1 &
echo "O4b pid=$!"

# O5b: post-reset stretch 15001..15100, T15's own ckpt_15000 (opacity m/v zeroed
# by the real reset at 15000 — a REAL reset edge case, not synthetic)
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T15 --it-lo 15001 --it-hi 15100 \
  --init-mode ckpt --ckpt "$D"/room_T15/ckpt_15000.pt --n-rows 100000 --seed 7 \
  --obs-points 10,37,99 --fence-points 50 \
  --outdir "$O" --tag room_T15_postreset < /dev/null > "$D"/logs/oracle_O5b.log 2>&1 &
echo "O5b pid=$!"

echo "ORACLE_R3_LAUNCHED $(date -u +%FT%TZ)"
