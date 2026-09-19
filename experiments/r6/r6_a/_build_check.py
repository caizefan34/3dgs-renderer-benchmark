#!/usr/bin/env python3
"""Build and verify the R6-A instrumented gsplat.
This forces a fresh build by ensuring the build directory does not exist,
so gsplat's JIT loader will build from scratch.
"""
import os, sys, shutil
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["PYTHONPATH"] = "/tmp/r6a_patched"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["FAST_COMPILE"] = "1"
os.environ["PATH"] = "/home/liaoyuanjun/miniforge3/envs/anysplat/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")

# Ensure clean build state
build_dir = os.path.expanduser("~/.cache/torch_extensions/py310_cu124/gsplat_cuda")
if os.path.exists(build_dir):
    print(f"Removing stale build dir: {build_dir}")
    shutil.rmtree(build_dir)

sys.path.insert(0, "/tmp/r6a_patched")

try:
    import gsplat
    print("gsplat imported OK")
    from gsplat.cuda._backend import _C
    print("_C loaded OK")
    # Check the debug counter functions exist
    assert hasattr(_C, "r6a_reset_debug_counter"), "r6a_reset_debug_counter missing"
    assert hasattr(_C, "r6a_get_debug_counter"), "r6a_get_debug_counter missing"
    print("R6-A debug counter bindings OK")
    print("BUILD SUCCESS")
except Exception as e:
    print(f"BUILD FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
