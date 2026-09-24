#!/bin/bash
cd ~/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 python3 -u experiments/r2_profile_only/r21a_benchmark.py > results/reference_v1/r2.1a/benchmark.log 2>&1
echo "EXIT_CODE=$?" >> results/reference_v1/r2.1a/benchmark.log
