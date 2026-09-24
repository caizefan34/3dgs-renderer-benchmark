#!/usr/bin/env python3
"""Smoke test: can we import the R3 runner and run 1 sample end-to-end?"""
import sys, os
sys.path.insert(0, "experiments/r3")
sys.path.insert(0, "baseline/reference_v1")
sys.path.insert(0, "scripts/epic05/phase7")

import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda}")

try:
    import r3_certificate_runner as R
    print("import r3_certificate_runner: OK")
except Exception as e:
    import traceback
    print(f"import r3_certificate_runner FAILED:\n{traceback.format_exc()}")
    sys.exit(1)

# Show key module members
members = [m for m in dir(R) if not m.startswith('_')]
print(f"\nModule members ({len(members)}):")
for m in members:
    print(f"  {m}")

# Show what R.run_r3_measurement expects
import inspect
try:
    sig = inspect.signature(R.run_r3_measurement)
    print(f"\nrun_r3_measurement signature: {sig}")
except Exception as e:
    print(f"sig error: {e}")

# Try to access config paths
try:
    from config import ReferenceV1Config
    cfg = ReferenceV1Config(scene="room", iterations=30000)
    print(f"\nReferenceV1Config: scene={cfg.scene}, iterations={cfg.iterations}")
    for attr in dir(cfg):
        if not attr.startswith('_'):
            val = getattr(cfg, attr)
            if isinstance(val, (str, int, float, bool, tuple, list)):
                print(f"  {attr} = {val}")
except Exception as e:
    import traceback
    print(f"config import FAILED:\n{traceback.format_exc()}")
