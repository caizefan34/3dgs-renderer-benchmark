#!/bin/bash
# 700-iteration test with screen-space gradient scaling + official threshold
cd /home/liaoyuanjun/3dgs-renderer-benchmark
python3 -u -c "
import sys
sys.path.insert(0, 'baseline/reference_v1')
sys.path.insert(0, 'src')
from config import ReferenceV1Config
config = ReferenceV1Config(scene='room', iterations=700)
# Official threshold 0.0002 — now should work with pixel-space gradient scaling
from trainer import run_training
run_training(config, 'results/reference_v1/test_700_pixel', allow_dirty=True)
" 2>&1
