#!/bin/bash
# 2000-iteration test with absgrad threshold 0.0008 (gsplat recommendation)
cd /home/liaoyuanjun/3dgs-renderer-benchmark
python3 -u -c "
import sys
sys.path.insert(0, 'baseline/reference_v1')
sys.path.insert(0, 'src')
from config import ReferenceV1Config
config = ReferenceV1Config(scene='room', iterations=2000)
config.densify_grad_threshold = 0.0008  # gsplat absgrad recommendation
from trainer import run_training
run_training(config, 'results/reference_v1/test_2k_absgrad8', allow_dirty=True)
" 2>&1
