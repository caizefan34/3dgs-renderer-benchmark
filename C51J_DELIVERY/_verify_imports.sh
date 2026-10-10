#!/bin/bash
set +e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
echo "=== repo present on MX? ==="
ls -d $REPO 2>/dev/null && echo "REPO_OK" || echo "REPO_ABSENT"
echo "=== stage5 benchmark.py present? ==="
ls -l $REPO/scripts/phase-c51-stage5/benchmark.py 2>/dev/null || echo "benchmark.py ABSENT"
echo "=== src infra (gaussian_model/dataset/loss) present? ==="
ls $REPO/src/gaussian_model.py $REPO/src/dataset.py $REPO/src/loss.py 2>/dev/null || echo "some src ABSENT"
echo "=== can we import gsplat + rasterization signature in nerficg? ==="
$PY - <<'PYEOF' 2>&1 | head -20
try:
    from gsplat import rasterization
    import inspect
    print("rasterization import OK; params:", list(inspect.signature(rasterization).parameters)[:14])
except Exception as e:
    print("rasterization import FAIL:", repr(e)[:200])
try:
    import torch
    print("cuda:", torch.cuda.is_available(), "dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except Exception as e:
    print("torch FAIL:", repr(e)[:120])
PYEOF
echo "=== MipNeRF360 room data present + camera count ==="
find /mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/room -maxdepth 1 -type d 2>/dev/null | head
ls /mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/room 2>/dev/null | head
echo "DONE"
