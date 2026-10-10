#!/bin/bash
set +e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
echo "=== py_compile ==="
$PY -c "import py_compile; py_compile.compile('/tmp/C51J_harness.py', doraise=True); print('PYCOMPILE_OK')" 2>&1 | tail -2
echo "=== emit-plan (no GPU needed) ==="
$PY /tmp/C51J_harness.py --emit-plan --repo-root /home/liaoyuanjun/3dgs-renderer-benchmark --runtime-flag "--sparse_backward_topk 0.5" 2>&1 | head -40
echo "RC=$?"
echo "DONE"
