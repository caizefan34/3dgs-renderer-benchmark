#!/usr/bin/env python3
"""
R3-VEC — direct unit test of _accumulate_tile_bounds.
Calls the actual production function with a small scene and verifies
that it runs clean and returns valid results.
"""
import sys, os, math
import numpy as np
import torch

SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT)

import r3_certificate_runner as new

print("=" * 72)
print("R3-VEC — Direct _accumulate_tile_bounds test")
print("=" * 72)

torch.manual_seed(123)

# Build a minimal but realistic scene
N = 64
H, W = 64, 64
tile_size = 16
tile_h = (H + tile_size - 1) // tile_size
tile_w = (W + tile_size - 1) // tile_size

device = torch.device("cuda")
means2d = torch.randn(1, N, 2, device=device) * 30 + 32
conics = torch.rand(1, N, 3, device=device) * 0.5 + 0.5
opacities = torch.rand(1, N, device=device) * 0.9 + 0.05

print(f"  N={N} HxW={H}x{W} tile_size={tile_size} tiles={tile_h}x{tile_w}")

# Inspect the function signature
import inspect
sig = inspect.signature(new._accumulate_tile_bounds)
print(f"  signature: {sig}")

try:
    result = new._accumulate_tile_bounds(
        opacities, conics, means2d,
        None, None,
        0.5, tile_h, tile_w, tile_size,
        H=H, W=W
    )
    print("\nResult OK")
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)} finite={torch.isfinite(v).all().item()}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"  type={type(result)} len={len(result)}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nFAILED: {type(e).__name__}: {e}")
    sys.exit(1)
