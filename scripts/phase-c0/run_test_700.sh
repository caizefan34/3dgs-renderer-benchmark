#!/bin/bash
# Quick 700-iteration test run (triggers first densification at 600)
cd /home/liaoyuanjun/3dgs-renderer-benchmark
python3 baseline/reference_v1/trainer.py \
    --scene room \
    --iterations 700 \
    --output_dir results/reference_v1/test_700 \
    --allow_dirty \
    2>&1
