#!/bin/bash
set +e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
echo "=== py_compile ==="
$PY -c "import py_compile; py_compile.compile('/tmp/C51J_harness.py', doraise=True); print('PYCOMPILE_OK')" 2>&1 | tail -3
echo "=== live gate run (gpu 0, expect refuse: <30GiB free) ==="
$PY /tmp/C51J_harness.py --arm native_baseline --scene room --iters 1000 --repo-root $REPO --gpu 0 2>&1 | head -30
echo "RC=$?"
echo "=== live gate with home fallback storage override (still gpu-blocked) ==="
C51J_FALLBACK_OUT=$HOME/C51J_runs $PY /tmp/C51J_harness.py --arm native_baseline --scene room --iters 1000 --repo-root $REPO --gpu 1 2>&1 | grep -E "RESOURCE GATE|FREE mem|output_dir|fallback|RC=|Refusing" | head -10
echo "DONE"
