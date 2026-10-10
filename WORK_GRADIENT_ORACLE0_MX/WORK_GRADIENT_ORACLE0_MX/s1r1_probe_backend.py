#!/usr/bin/env python3
"""Probe: can we import the native FasterGS backend in isolation (sys.path only)?"""
import sys, os
FGS = "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting"
sys.path.insert(0, FGS + "/FasterGSFusedCudaBackend")
sys.path.insert(0, FGS)
try:
    from FasterGSFusedCudaBackend import _C
    print("OK imported FasterGSFusedCudaBackend._C")
    print("forward:", hasattr(_C,"forward"), "backward:", hasattr(_C,"backward"))
    print("forward doc:", (_C.forward.__doc__ or "")[:300])
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAILED:", e)
