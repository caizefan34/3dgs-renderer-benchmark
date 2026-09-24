#!/usr/bin/env python3
"""Check both gsplat installations."""
import os, sys, importlib

paths = [
    "/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages",
    "/home/liaoyuanjun/.local/lib/python3.10/site-packages",
]

for p in paths:
    gsplat_dir = os.path.join(p, "gsplat")
    csrc_so = os.path.join(gsplat_dir, "csrc.so")
    print(f"\n{gsplat_dir}:")
    print(f"  exists: {os.path.exists(gsplat_dir)}")
    print(f"  csrc.so exists: {os.path.exists(csrc_so)}")
    if os.path.exists(csrc_so):
        import subprocess
        r = subprocess.run(["file", csrc_so], capture_output=True, text=True)
        print(f"  file: {r.stdout.strip()}")
        # Check size
        print(f"  size: {os.path.getsize(csrc_so)} bytes")

# Try importing from conda env path only
print("\n--- Trying conda env gsplat only ---")
# Remove user-local from path
sys.path = [p for p in sys.path if '.local' not in p]
sys.path.insert(0, "/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages")

# Clear cached modules
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("gsplat"):
        del sys.modules[mod_name]

try:
    import gsplat
    print(f"gsplat version: {gsplat.__version__}")
    print(f"gsplat file: {gsplat.__file__}")
    from gsplat import csrc as _C
    print("csrc import: OK")
except ImportError as e:
    print(f"FAILED: {e}")
