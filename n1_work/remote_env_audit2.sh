#!/bin/bash
set -u
echo "=== find fused_ssim ==="
find /root/miniforge3/envs/gsplat -name "fused_ssim_cuda.so" 2>/dev/null
find /root/miniforge3/envs/gsplat -name "fused_ssim" -maxdepth 6 -type d 2>/dev/null
echo "=== other envs ==="
ls /root/miniforge3/envs/
for e in /root/miniforge3/envs/*/bin/python; do
  echo "--- $e"
  $e -c "import sys; print(sys.version.split()[0])" 2>/dev/null
  $e -c "import pyarrow; print('pyarrow', pyarrow.__version__)" 2>/dev/null || echo "no pyarrow"
  $e -c "import sklearn; print('sklearn', sklearn.__version__)" 2>/dev/null || echo "no sklearn"
  $e -c "import pandas; print('pandas', pandas.__version__)" 2>/dev/null || echo "no pandas"
done
echo "=== base python ==="
/root/miniforge3/bin/python -c "import pyarrow; print('pyarrow', pyarrow.__version__)" 2>/dev/null || echo "no pyarrow"
/root/miniforge3/bin/python -c "import sklearn; print('sklearn', sklearn.__version__)" 2>/dev/null || echo "no sklearn"
/root/miniforge3/bin/python -c "import pandas; print('pandas', pandas.__version__)" 2>/dev/null || echo "no pandas"
echo "=== pip offline check ==="
/root/miniforge3/envs/gsplat/bin/pip --version 2>/dev/null
timeout 15 /root/miniforge3/envs/gsplat/bin/pip download pyarrow --no-deps -d /tmp/pipcheck 2>&1 | tail -3
echo "=== config.py ==="
cat /root/3dgs-renderer-benchmark/baseline/reference_v1/config.py
