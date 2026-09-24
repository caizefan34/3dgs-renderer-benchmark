#!/bin/bash
cd /home/liaoyuanjun/3dgs-renderer-benchmark
git add baseline/reference_v1/config.py baseline/reference_v1/trainer.py scripts/phase-r0.1/
git commit -m "add: checkpoint_iterations config and R0.1 scripts"
echo "=== SHA ==="
git rev-parse HEAD
echo "=== DIRTY ==="
git diff --quiet && echo false || echo true
