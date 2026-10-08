#!/bin/bash
# VOS Phase-5 replay-oracle launch matrix (room validation set).
# All runs: exact-replay equivalence vs torch Adam (A_f default / A_s single-tensor).
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
O=$D/oracle
mkdir -p "$O" "$D"/logs
cd "$D"

# O1: random init, 100K rows, vos-region segment 15001..16000, fence at 400
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T30 --it-lo 15001 --it-hi 16000 \
  --init-mode random --n-rows 100000 --seed 7 \
  --obs-points 50,137,299,500,777,999 --fence-points 400 \
  --outdir "$O" --tag room_rand_15k < /dev/null > "$D"/logs/oracle_O1.log 2>&1 &
echo "O1 pid=$!"

# O2: ckpt_30000 init (realistic state), 100K rows, segment 29001..30000, fence 700
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T30 --it-lo 29001 --it-hi 30000 \
  --init-mode ckpt --ckpt "$D"/room_T30/ckpt_30000.pt --n-rows 100000 --seed 7 \
  --obs-points 50,137,299,500,700,777,999 --fence-points 700 \
  --outdir "$O" --tag room_ckpt30k < /dev/null > "$D"/logs/oracle_O2.log 2>&1 &
echo "O2 pid=$!"

# O3: FULL-N control, ckpt init, 500 iters (29501..30000)
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T30 --it-lo 29501 --it-hi 30000 \
  --init-mode ckpt --ckpt "$D"/room_T30/ckpt_30000.pt --n-rows 0 --seed 7 \
  --obs-points 50,137,299,450 --fence-points 300 \
  --outdir "$O" --tag room_ckpt30k_fullN < /dev/null > "$D"/logs/oracle_O3.log 2>&1 &
echo "O3 pid=$!"

# O4: densify-region stretch 5001..5100 with ckpt_5000 init (N matches post-event)
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T30 --it-lo 5001 --it-hi 5100 \
  --init-mode ckpt --ckpt "$D"/room_R5/ckpt_5000.pt --n-rows 100000 --seed 7 \
  --obs-points 10,37,99 --fence-points 50 \
  --outdir "$O" --tag room_ckpt5k_densify < /dev/null > "$D"/logs/oracle_O4.log 2>&1 &
echo "O4 pid=$!"

# O5: post-reset stretch 15001..15100 with ckpt_15000 init (opacity m/v all-zero edge case)
setsid nohup $PY vos_replay_oracle.py \
  --trace-dir "$D"/room_T30 --it-lo 15001 --it-hi 15100 \
  --init-mode ckpt --ckpt "$D"/room_R15/ckpt_15000.pt --n-rows 100000 --seed 7 \
  --obs-points 10,37,99 --fence-points 50 \
  --outdir "$O" --tag room_ckpt15k_reset < /dev/null > "$D"/logs/oracle_O5.log 2>&1 &
echo "O5 pid=$!"

echo "ORACLE_LAUNCHED $(date -u +%FT%TZ)"
