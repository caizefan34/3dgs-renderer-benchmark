#!/bin/bash
# P8 fixed-state replay: 3 scenes x 4 arms (dump + timing), one scene per
# free GPU (1/2/3). Correctness-relevant: co-residence with nothing (GPUs
# 1-3 idle); the bicycle GPU-0 rerun runs concurrently elsewhere.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
run_scene() {  # $1 scene, $2 gpu
  for arm in b1a a0 c0 gmain; do
    $PY /root/58_p8_replay.py --arm $arm --scene $1 --gpu $2 --phase dump \
      > /root/3dgs-publication-epic05/logs/p8_replay_${1}_${arm}_dump.log 2>&1
    $PY /root/58_p8_replay.py --arm $arm --scene $1 --gpu $2 --phase timing \
      > /root/3dgs-publication-epic05/logs/p8_replay_${1}_${arm}_timing.log 2>&1
    echo "scene $1 arm $arm done rc=$?"
  done
}
run_scene room 1 > /root/3dgs-publication-epic05/logs/p8_replay_room.log 2>&1 &
P1=$!
run_scene bicycle 2 > /root/3dgs-publication-epic05/logs/p8_replay_bicycle.log 2>&1 &
P2=$!
run_scene garden 3 > /root/3dgs-publication-epic05/logs/p8_replay_garden.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
echo ALL_REPLAY_DONE
