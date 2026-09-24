#!/bin/bash
# R6 recon: environment, checkpoints, baseline code hashes, profilers
# Run via:  Get-Content -Raw experiments/r6/recon.sh | ssh mx 'bash -s'
set -uo pipefail

PY="$HOME/miniforge3/envs/anysplat/bin/python"

echo "=ENV="
cat > /tmp/r6_envcheck.py << 'PYEOF'
import sys, gsplat, torch
print("py", sys.version.split()[0])
print("gsplat", gsplat.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gfile", gsplat.__file__)
try:
    import gsplat.cuda as _c
    print("cuda_pkg", getattr(_c, "__file__", "n/a"))
except Exception as e:
    print("cuda_pkg_err", repr(e))
PYEOF
timeout 120 "$PY" /tmp/r6_envcheck.py </dev/null 2>&1
echo "RC_env=$?"

echo "=NCU/NSYS="
which ncu nsys 2>&1
ls /opt/nvidia/nsight-compute*/ncu /opt/nvidia/nsight-systems*/nsys /usr/local/cuda/bin/ncu /usr/local/cuda/bin/nsys 2>/dev/null
ncu --version 2>&1 | head -3
nsys --version 2>&1 | head -3

echo "=CKPT room/bicycle/garden (r4_13scene_v2)="
find /mnt/storage_pool/liaoyuanjun/r4_13scene_v2 -maxdepth 4 -name 'iter_*.pt' 2>/dev/null | grep -Ei 'room|bicycle|garden' | sort
echo "=r4_13scene_v2 toplevel="
ls /mnt/storage_pool/liaoyuanjun/r4_13scene_v2/ 2>/dev/null

echo "=ANY OTHER ckpt dirs="
find /mnt/storage_pool/liaoyuanjun -maxdepth 3 -type d -name checkpoints 2>/dev/null | head -30

echo "=HASHES+GIT="
cd "$HOME/3dgs-renderer-benchmark" || exit 1
sha256sum baseline/reference_v1/trainer.py baseline/reference_v1/config.py experiments/r4/r4_train_wrapper.py baseline/reference_v1/gaussian_model.py 2>/dev/null
git rev-parse HEAD
git status --short | head -5

echo "=GPU NOW="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "=RECON DONE="
