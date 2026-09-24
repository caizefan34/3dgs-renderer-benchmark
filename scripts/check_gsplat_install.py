#!/usr/bin/env python3
"""Check gsplat installation state."""
import sys, os

print("Python:", sys.executable)
print("sys.path (first 5):", sys.path[:5])

# Check if gsplat is importable
try:
    import gsplat
    print(f"gsplat version: {gsplat.__version__}")
    print(f"gsplat file: {gsplat.__file__}")
    gsplat_dir = os.path.dirname(gsplat.__file__)
    cuda_dir = os.path.join(gsplat_dir, "cuda")
    csrc_so = os.path.join(gsplat_dir, "csrc.so")
    print(f"csrc.so exists: {os.path.exists(csrc_so)}")
    if os.path.exists(cuda_dir):
        print(f"cuda dir exists: {cuda_dir}")
        # Check _backend.py
        backend = os.path.join(cuda_dir, "_backend.py")
        print(f"_backend.py exists: {os.path.exists(backend)}")
    # Try import csrc
    try:
        from gsplat import csrc as _C
        print("csrc import: OK")
    except ImportError as e:
        print(f"csrc import FAILED: {e}")
except ImportError as e:
    print(f"gsplat import FAILED: {e}")

# Check pip show
import subprocess
r = subprocess.run(["pip", "show", "gsplat"], capture_output=True, text=True)
print(f"\npip show gsplat:\n{r.stdout}")
