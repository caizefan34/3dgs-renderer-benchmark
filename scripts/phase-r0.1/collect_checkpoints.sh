#!/bin/bash
# Collect checkpoints at 2000, 5000, 10000, 14000 for R0.1 continuation experiments
# Uses frozen REFERENCE_V1_ABSGRAD code. NOT a 30K training.
cd /home/liaoyuanjun/3dgs-renderer-benchmark

echo "=== Checkpoint collection: 14K iterations ==="
echo "Started: $(date)"

# Copy frozen camera sequence from 30K run to ensure identical training
cp results/reference_v1/room_30k/camera_sequence.npy results/reference_v1/ckpt_14k_camera_sequence.npy 2>/dev/null || true

python3 -u -c "
import sys, os, shutil
sys.path.insert(0, 'baseline/reference_v1')
sys.path.insert(0, 'src')
from config import ReferenceV1Config
from trainer import run_training

config = ReferenceV1Config(scene='room', iterations=14000)
# All defaults match frozen REFERENCE_V1_ABSGRAD
run_training(
    config,
    'results/reference_v1/ckpt_14k',
    allow_dirty=True  # infrastructure run, not paper experiment
)
" 2>&1

echo "=== Finished: $(date) ==="
