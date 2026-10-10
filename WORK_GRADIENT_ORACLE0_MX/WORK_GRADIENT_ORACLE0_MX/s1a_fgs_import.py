#!/usr/bin/env python3
"""Retry FasterGS _C import with torch libs preloaded; check ABI compat."""
import os, sys, ctypes, glob
import torch
# Preload torch libs so _C can resolve libc10.so etc.
torch_lib = os.path.dirname(torch.__file__)
for lib in ["libc10.so","libtorch.so","libtorch_cpu.so","libtorch_python.so","libtorch_cuda.so","libc10_cuda.so"]:
    p = os.path.join(torch_lib, lib)
    if os.path.exists(p):
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL); print("preloaded", lib)
        except Exception as e: print("preload fail", lib, e)

FGB = "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting/FasterGSFusedCudaBackend"
sys.path.insert(0, FGB)
try:
    from FasterGSFusedCudaBackend import _C
    print("FasterGS _C imported OK; ops:", [a for a in dir(_C) if not a.startswith('__')])
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAILED:", e)
