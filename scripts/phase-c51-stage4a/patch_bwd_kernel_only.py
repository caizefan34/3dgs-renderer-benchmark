#!/usr/bin/env python3
"""Patch only RasterizeToPixels3DGSBwd.cu — the CUDA kernel (fixed launch call order)."""
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/phase-c51-stage4a")
from patch_cuda import patch_bwd_kernel
patch_bwd_kernel()
print("Done patching RasterizeToPixels3DGSBwd.cu")
