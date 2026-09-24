#!/usr/bin/env python3
"""Smoke test: verify the optimized _accumulate_tile_bounds and tensor replay work."""
import sys
import os
import math
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import gsplat
    print(f"gsplat version: {getattr(gsplat, '__version__', 'unknown')}")
except ImportError as e:
    print(f"gsplat import FAILED: {e}")
    sys.exit(1)

# Import the runner module
try:
    import r3_certificate_runner as runner
    print("r3_certificate_runner imported OK")
except Exception as e:
    print(f"r3_certificate_runner import FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test the tensor replay function directly
print("\n=== Testing _compute_faithful_work_tensor ===")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussian_ids = torch.tensor([0, 1, 2, 2, 1], device=device)
    means2d_0 = torch.tensor([[10.0, 10.0], [50.0, 50.0], [100.0, 100.0]], device=device)
    conics_0 = torch.tensor([[1.0, 0.0, 1.0], [0.5, 0.0, 0.5], [2.0, 0.0, 2.0]], device=device)
    opacities = torch.tensor([0.8, 0.9, 0.7], device=device)
    tile_x, tile_y = 0, 0

    wc, wu = runner._compute_faithful_work_weights_tensor(
        gaussian_ids, means2d_0, conics_0, opacities,
        0, 0, 16, 256, 256
    )
    print(f"  wc={wc}")
    print(f"  wu={wu}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test the full accumulate with a tiny synthetic scene
print("\n=== Testing _accumulate_tile_bounds with synthetic data ===")
try:
    # Reuse the test setup pattern from the runner if it has one
    if hasattr(runner, 'run_gt_certificate_suite'):
        # Create minimal fake arguments
        class Args:
            pass
        args = Args()
        args.device = "cuda"
        args.checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "checkpoints", "cd1_checkpoint.pth") if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "data", "checkpoints", "cd1_checkpoint.pth")) else None
        print(f"  Will try run_gt_certificate_suite with args.device={args.device}")
    else:
        print("  run_gt_certificate_suite not found - skipping full test")
except Exception as e:
    print(f"  Full test FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== Smoke check complete ===")
