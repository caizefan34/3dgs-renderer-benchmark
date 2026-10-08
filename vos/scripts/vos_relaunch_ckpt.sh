#!/bin/bash
# Relaunch the 6 stage-checkpoint runs WITH --checkpoints (idle GPUs 1-6).
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
D=/root/vos_traces
T=/root/epic05_p13x/publication_trainer_p13x.py
mkdir -p "$D"/logs
cd "$D"
TRAINER_ARGS="--arm c0 --loss fused --eps2d 0.3 --final-eval all --timing-grade FUNCTIONAL_ONLY --lpips --checkpoints"
setsid nohup $PY "$T" $TRAINER_ARGS --scene room     --iterations 15000 --gpu 1 --outdir "$D"/room_R15     < /dev/null > "$D"/logs/room_R15c.log 2>&1 &
echo "room_R15c pid=$!"
setsid nohup $PY "$T" $TRAINER_ARGS --scene bicycle  --iterations 15000 --gpu 2 --outdir "$D"/bicycle_R15  < /dev/null > "$D"/logs/bicycle_R15c.log 2>&1 &
echo "bicycle_R15c pid=$!"
setsid nohup $PY "$T" $TRAINER_ARGS --scene garden   --iterations 15000 --gpu 3 --outdir "$D"/garden_R15   < /dev/null > "$D"/logs/garden_R15c.log 2>&1 &
echo "garden_R15c pid=$!"
setsid nohup $PY "$T" $TRAINER_ARGS --scene room     --iterations 5000  --gpu 4 --outdir "$D"/room_R5      < /dev/null > "$D"/logs/room_R5c.log 2>&1 &
echo "room_R5c pid=$!"
setsid nohup $PY "$T" $TRAINER_ARGS --scene bicycle  --iterations 5000  --gpu 5 --outdir "$D"/bicycle_R5   < /dev/null > "$D"/logs/bicycle_R5c.log 2>&1 &
echo "bicycle_R5c pid=$!"
setsid nohup $PY "$T" $TRAINER_ARGS --scene garden   --iterations 5000  --gpu 6 --outdir "$D"/garden_R5    < /dev/null > "$D"/logs/garden_R5c.log 2>&1 &
echo "garden_R5c pid=$!"
echo "RELAYNCHED $(date -u +%FT%TZ)"
