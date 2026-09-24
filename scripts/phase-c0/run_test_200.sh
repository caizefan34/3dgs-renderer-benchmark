#!/bin/bash
# Quick 200-iteration test run
cd /home/liaoyuanjun/3dgs-renderer-benchmark
python3 baseline/reference_v1/trainer.py \
    --scene room \
    --iterations 200 \
    --output_dir results/reference_v1/test_200 \
    --allow_dirty \
    2>&1
