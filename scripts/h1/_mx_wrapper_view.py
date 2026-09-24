"""Show _wrapper.py sections around 'viewmats' with validation, and benchmark _std_ll_forward."""
import os, re

ss = lambda p: open(p, errors="replace").read()

wrapper = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-himsg-mx/gsplat/cuda/_wrapper.py"
bench = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/benchmark/run_himsg_train_benchmark.py"

src = ss(wrapper)
lines = src.splitlines()

# 1) print context around def fully_fused_projection
idx = None
for i, line in enumerate(lines):
    if "def fully_fused_projection" in line:
        idx = i
        break
print(f"=== fully_fused_projection at L{idx+1} ===")
for j in range(idx, min(idx + 90, len(lines))):
    print(f"{j+1}: {lines[j]}")

# 2) grep for 'viewmats' everywhere in wrapper, print count
hits = [(i, l) for i, l in enumerate(lines) if "viewmats" in l and ("shape" in l or "=" in l or ":" in l)]
print("\n=== viewmats lines mentioning shape or param ===")
for i, l in hits[:40]:
    print(f"{i+1}: {l.strip()}")
