"""Inspect gsplat-higs-mx API for viewmats handling (finding the shape check)."""
import os, re, glob

H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat"

# Find files containing the viewmats shape check
for root, dirs, files in os.walk(H):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            with open(p, errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if "viewmats must have shape" in line:
                        print(f"{p}:{i}: {line.rstrip()}")

# Show the wrapper functions with 'fully_proj' or 'rasterize' defs
for root, dirs, files in os.walk(H):
    for f in files:
        if f == "_wrapper.py":
            p = os.path.join(root, f)
            print(f"\n=== {p} ===")
            with open(p) as fh:
                lines = fh.readlines()
            for i, line in enumerate(lines):
                if ("def " in line and ("rasterize" in line or "proj" in line or "isect" in line)):
                    # print function signature and next few lines
                    print(f"  L{i+1}: {line.rstrip()}")
                    for j in range(i + 1, min(i + 12, len(lines))):
                        if lines[j].strip() == "":
                            break
                        print(f"    L{j+1}: {lines[j].rstrip()}")
                    print()
