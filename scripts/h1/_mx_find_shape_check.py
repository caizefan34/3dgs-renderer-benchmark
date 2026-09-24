"""Locate 'must have shape' in gsplat source and show benchmark call convention."""
import os, re

# 1) Find gsplat package path from the conda env
env_py = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
print("=== gsplat location ===")
try:
    import subprocess
    out = subprocess.check_output([
        env_py, "-c",
        "import gsplat, os; print(os.path.dirname(gsplat.__file__))"
    ], text=True)
    print(out.strip())
    GSPLAT_ROOT = out.strip()
except Exception as e:
    print("env python failed:", e)
    GSPLAT_ROOT = "/mnt/storage_pool/liaoyuanjun/higs-13scene-mx/gsplat"

# 2) grep for 'must have shape' and 'viewmats' error messages
print("\n=== 'must have shape' occurrences ===")
for dirpath, dirnames, filenames in os.walk(GSPLAT_ROOT):
    if "build" in dirpath or ".git" in dirpath:
        continue
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            with open(p, errors="replace") as fh:
                lines = fh.readlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if "must have shape" in line or "must be" in line and "shape" in line:
                print(f"{p}:{i+1}: {line.strip()}")

# 3) Show how the benchmark defines and calls _std_ll_forward
bench = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark/run_higs_train_benchmark.py"
print("\n=== benchmark call sites (context) ===")
with open(bench) as fh:
    src = fh.readlines()

for i, line in enumerate(src):
    if any(k in line for k in ["fully_projected(", "fully_projection(", "rasterize_gaussian_higs", "create_higs_renderer(", "fully_fused_projection", "isect_tiles("]):
        print(f"\n--- L{i+1}: {line.rstrip()}")
        for j in range(max(0, i-3), min(len(src), i+14)):
            print(f"  {j+1}: {src[j].rstrip()}")
