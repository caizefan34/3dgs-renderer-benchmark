#!/bin/bash
# VOS Phase-5 oracle round 2: fixed random-init + bicycle/garden ckpt runs +
# synthetic opacity-reset edge case + traced T5/T15 room runs for densify-region
# ckpt pairing (T30 ckpts pair only with their own in-process traces; the R runs
# diverged via atomic nondeterminism, N mismatch ~470 at iter 5000).
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
O=$D/oracle
mkdir -p "$O" "$D"/logs
cd "$D"

# O1b: random init (fixed subset logic), vos-region segment
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T30 --it-lo 15001 --it-hi 16000 \
  --init-mode random --n-rows 100000 --seed 7 \
  --obs-points 50,137,299,500,777,999 --fence-points 400 \
  --outdir "$O" --tag room_rand_15k < /dev/null > "$D"/logs/oracle_O1b.log 2>&1 &
echo "O1b pid=$!"

# O6: bicycle ckpt_30000 (realistic state), 100K rows
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/bicycle_T30 --it-lo 29001 --it-hi 30000 \
  --init-mode ckpt --ckpt "$D"/bicycle_T30/ckpt_30000.pt --n-rows 100000 --seed 7 \
  --obs-points 50,137,299,500,700,777,999 --fence-points 700 \
  --outdir "$O" --tag bicycle_ckpt30k < /dev/null > "$D"/logs/oracle_O6.log 2>&1 &
echo "O6 pid=$!"

# O7: garden ckpt_30000 (realistic state), 100K rows
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/garden_T30 --it-lo 29001 --it-hi 30000 \
  --init-mode ckpt --ckpt "$D"/garden_T30/ckpt_30000.pt --n-rows 100000 --seed 7 \
  --obs-points 50,137,299,500,700,777,999 --fence-points 700 \
  --outdir "$O" --tag garden_ckpt30k < /dev/null > "$D"/logs/oracle_O7.log 2>&1 &
echo "O7 pid=$!"

# O9: synthetic opacity-reset edge case (zero m/v), random init
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T30 --it-lo 15001 --it-hi 16000 \
  --init-mode random --n-rows 100000 --seed 9 --zero-opacity-mv \
  --obs-points 50,137,299,500,777,999 --fence-points 400 \
  --outdir "$O" --tag room_rand_zeroMV < /dev/null > "$D"/logs/oracle_O9.log 2>&1 &
echo "O9 pid=$!"

# T5/T15 traced room runs WITH ckpts (in-process trace+ckpt pairing for the
# densify-region and post-reset oracle tests)
setsid nohup $PY vos_trace_driver.py --scene room --iterations 5000 --gpu 4 \
  --final-eval subset --outdir "$D"/room_T5 < /dev/null > "$D"/logs/room_T5.log 2>&1 &
echo "T5 pid=$!"
setsid nohup $PY vos_trace_driver.py --scene room --iterations 15000 --gpu 5 \
  --final-eval subset --outdir "$D"/room_T15 < /dev/null > "$D"/logs/room_T15.log 2>&1 &
echo "T15 pid=$!"

echo "ORACLE_R2_LAUNCHED $(date -u +%FT%TZ)"
