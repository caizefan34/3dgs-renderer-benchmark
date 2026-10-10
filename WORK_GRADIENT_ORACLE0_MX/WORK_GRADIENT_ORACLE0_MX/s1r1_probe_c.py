#!/usr/bin/env python3
"""Probe: import torch first, then load _C via ctypes/importlib."""
import sys, os
import torch  # loads libc10.so etc into the process
_C_so_dir = "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend"
sys.path.insert(0, _C_so_dir)
try:
    import _C
    print("OK imported _C directly (torch preloaded)")
    print("has forward:", hasattr(_C,"forward"), "has backward:", hasattr(_C,"backward"))
    print("forward doc:", repr(_C.forward.__doc__)[:800])
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAILED:", e)
