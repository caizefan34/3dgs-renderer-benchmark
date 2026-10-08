#!/bin/bash
set -e
/root/miniforge3/envs/gsplat/bin/pip install --no-deps /tmp/pipcheck/pyarrow-25.0.1-cp310-cp310-manylinux_2_28_x86_64.whl 2>&1 | tail -2
/root/miniforge3/envs/gsplat/bin/python -c "import pyarrow, numpy, torch; print('pyarrow', pyarrow.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"
echo "=== training progress ==="
tail -3 /root/epic05_n1/runs/n1snap_c0f_s42_room/stdout.log || true
tail -3 /root/epic05_n1/runs/n1snap_c0f_s42_bicycle/stdout.log || true
