"""Extract key gsplat-higs-mx rasterization function signatures & validation logic."""
import os, sys, re, glob

# Find the right gsplat home: the resolved import path from the env
code = r'''
import gsplat, os
print("GS_PATH", os.path.dirname(gsplat.__file__))
'''
# We cannot easily capture env python output here without running; instead search
# candidate roots
roots = [
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat",
]
found = []
for r in roots:
    if os.path.isdir(r):
        found.append(r)
print("roots_found", found)

for H in found:
    w = os.path.join(H, "cuda", "_wrapper.py")
    if not os.path.isfile(w):
        print("NO WRAPPER", w)
        continue
    print(f"\n========== {w} ==========")
    with open(w, errors="replace") as fh:
        lines = fh.readlines()
    # find function defs related to proj / rasterize
    for i, line in enumerate(lines):
        if line.startswith("def ") and any(k in line for k in ["proj", "raster", "isect", "fused", "persp"]):
            # print signature + body until next def (max 30 lines)
            print(f"\n--- L{i+1}: {line.rstrip()}")
            for j in range(i + 1, min(i + 30, len(lines))):
                nxt = lines[j]
                if nxt.startswith("def ") and j > i:
                    break
                print(f"    L{j+1}: {nxt.rstrip()}")
    # also find assert / raise messages about viewmats
    print("\n--- viewmats error messages ---")
    for i, line in enumerate(lines):
        if "viewmats" in line and ("raise" in line or "assert" in line or "must" in line):
            print(f"  L{i+1}: {line.rstrip()}")
            print(f"  ctx: {lines[max(0,i-2):i+3]}")
