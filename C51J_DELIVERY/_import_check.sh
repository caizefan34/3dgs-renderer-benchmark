#!/bin/bash
set +e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
REPO=/home/liaoyuanjun/3dgs-renderer-benchmark
echo "=== can benchmark.py + its infra import under nerficg (no run, import only)? ==="
cd $REPO/scripts/phase-c51-stage5
$PY - <<'PYEOF' 2>&1 | head -25
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
try:
    from gaussian_model import GaussianModel
    print("GaussianModel OK")
except Exception as e:
    print("GaussianModel FAIL:", repr(e)[:160])
try:
    from dataset import GTDataset, load_initial_checkpoint
    print("dataset OK")
except Exception as e:
    print("dataset FAIL:", repr(e)[:160])
try:
    from loss import d_ssim_loss
    print("loss OK")
except Exception as e:
    print("loss FAIL:", repr(e)[:160])
try:
    import benchmark
    print("benchmark.run_experiment present:", hasattr(benchmark, "run_experiment"))
except Exception as e:
    print("benchmark FAIL:", repr(e)[:200])
PYEOF
echo "DONE"
