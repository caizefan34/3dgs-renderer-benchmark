#!/bin/bash
# ============================================================================
# Reference V1 Baseline — Room 30K Canonical Trajectory
# 
# THE ONLY allowed baseline for all future paper experiments.
# Runs full 30K iterations with C49/C50/C53 instrumentation.
# 
# Usage: nohup bash scripts/phase-c0/run_room_30k.sh > room_30k.log 2>&1 &
# ============================================================================
cd /home/liaoyuanjun/3dgs-renderer-benchmark

echo "=============================================="
echo "Reference V1 Baseline — Room 30K"
echo "Started: $(date)"
echo "=============================================="

python3 -u -c "
import sys
sys.path.insert(0, 'baseline/reference_v1')
sys.path.insert(0, 'src')
from config import ReferenceV1Config
from trainer import run_training

config = ReferenceV1Config(scene='room', iterations=30000)
# gsplat adaptation: absgrad requires 4x higher threshold (0.0008 vs official 0.0002)
config.densify_grad_threshold = 0.0008
# All other defaults match official Graphdeco 3DGS
run_training(
    config,
    'results/reference_v1/room_30k',
    allow_dirty=True
)
" 2>&1

echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
