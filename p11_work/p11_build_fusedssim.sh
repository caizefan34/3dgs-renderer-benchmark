#!/bin/bash
# P11: build fused-ssim into the gsplat conda env (measurement-only; existing public impl)
set -euo pipefail
export CUDA_ARCHITECTURES=80
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/fused-ssim
$PY -m pip install --no-build-isolation --quiet . 2>&1 | tail -5 || true
$PY - <<'EOF'
import fused_ssim, torch
a = torch.rand(1, 3, 64, 64, device="cuda")
b = torch.rand(1, 3, 64, 64, device="cuda")
s = fused_ssim.fused_ssim(a, b, padding="same")
torch.cuda.synchronize()
print("fused_ssim OK:", float(s))
import fused_ssim_cuda
print("module file:", fused_ssim_cuda.__file__)
EOF
