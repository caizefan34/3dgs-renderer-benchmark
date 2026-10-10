#!/bin/bash
set +e
PY=/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python
echo "=== does importing benchmark.py allocate GPU memory? ==="
$PY - <<'PYEOF' 2>&1 | head -20
import torch
torch.cuda.init() if False else None
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    before = torch.cuda.memory_allocated(0)
print("before import allocated:", torch.cuda.memory_allocated(0) if torch.cuda.is_available() else "n/a")
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/phase-c51-stage5")))
import benchmark
after = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
print("after import allocated:", after, "delta:", after-before)
# Is cuda visible? is device 0 the A100?
print("cuda available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
PYEOF
echo "DONE"
