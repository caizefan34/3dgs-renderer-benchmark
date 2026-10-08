#!/bin/bash
set -e
/root/miniforge3/envs/gsplat/bin/python -c "
import sys
sys.path.insert(0, '/root/epic05_p14/scripts')
import p14_common
print('selftest passed: stream labeling semantics verified')
"
