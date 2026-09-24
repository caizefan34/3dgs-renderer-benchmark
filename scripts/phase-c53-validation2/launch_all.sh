#!/bin/bash
# Launch all 5 workload pair collections in parallel
cd ~/3dgs-renderer-benchmark

nohup python3 scripts/phase-c53-validation2/collect_workload_pairs.py --gpu 0 --scene room --seed 42 > logs/phase-c53-validation2/room.log 2>&1 &
echo "Room (GPU 0) PID: $!"

nohup python3 scripts/phase-c53-validation2/collect_workload_pairs.py --gpu 1 --scene garden --seed 42 > logs/phase-c53-validation2/garden.log 2>&1 &
echo "Garden (GPU 1) PID: $!"

nohup python3 scripts/phase-c53-validation2/collect_workload_pairs.py --gpu 2 --scene bicycle --seed 42 > logs/phase-c53-validation2/bicycle.log 2>&1 &
echo "Bicycle (GPU 2) PID: $!"

nohup python3 scripts/phase-c53-validation2/collect_workload_pairs.py --gpu 3 --scene room --seed 123 > logs/phase-c53-validation2/room_seed123.log 2>&1 &
echo "Room seed123 (GPU 3) PID: $!"

nohup python3 scripts/phase-c53-validation2/collect_workload_pairs.py --gpu 4 --scene garden --seed 123 > logs/phase-c53-validation2/garden_seed123.log 2>&1 &
echo "Garden seed123 (GPU 4) PID: $!"

echo "All 5 collections launched. Check logs in ~10 min for first checkpoint."
