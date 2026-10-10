#!/bin/bash
set +e
for PY in /root/miniforge3/envs/gsplat/bin/python /root/miniforge3/envs/higs-13scene-env/bin/python /root/miniforge3/bin/python python3; do
  echo "=== try $PY ==="
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | head -3
done
echo "=== py_compile harness with gsplat env ==="
/root/miniforge3/envs/gsplat/bin/python -c "import py_compile; py_compile.compile('/tmp/C51J_harness.py', doraise=True); print('PYCOMPILE_OK')" 2>&1 | head -8
echo "=== resource gate dry-run (native_baseline room 1k) ==="
/root/miniforge3/envs/gsplat/bin/python /tmp/C51J_harness.py --arm native_baseline --scene room --iters 1000 --dry-run 2>&1 | head -40
echo "EXIT_GATE=$?"
