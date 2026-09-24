#!/bin/bash
# 700-iteration test with unbuffered output
cd /home/liaoyuanjun/3dgs-renderer-benchmark
python3 -u -c "
import sys
sys.path.insert(0, 'baseline/reference_v1')
sys.path.insert(0, 'src')
from config import ReferenceV1Config
config = ReferenceV1Config(scene='room', iterations=700)
config.densify_grad_threshold = 1e-5  # Calibrated for gsplat absgrad scale
from trainer import run_training
run_training(config, 'results/reference_v1/test_700_calib2', allow_dirty=True)
" 2>&1
