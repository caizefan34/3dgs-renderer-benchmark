#!/usr/bin/env python3
"""Check gsplat versions and source availability across mx conda envs."""
import sys, pathlib, subprocess

# Check anyplat env's gsplat csrc
p_any = pathlib.Path("/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat")
csrc_any = p_any / "cuda" / "csrc"
print("=== anyplat user-site gsplat ===")
print("root:", p_any)
print("csrc files:", len(list(csrc_any.iterdir())) if csrc_any.is_dir() else "NO CSRCDIR")
if csrc_any.is_dir():
    for f in sorted(csrc_any.iterdir()):
        if f.name.startswith("raster"):
            print("  ", f.name)
    print("  has rasterize_to_pixels_bwd.cu:", (csrc_any / "rasterize_to_pixels_bwd.cu").is_file())

# Check chorus env's gsplat
p_ch = pathlib.Path("/home/liaoyuanjun/miniforge3/envs/chorus/lib/python3.10/site-packages/gsplat")
csrc_ch = p_ch / "cuda" / "csrc"
print("\n=== chorus env gsplat ===")
print("root:", p_ch)
print("has rasterize_to_pixels_bwd.cu:", (csrc_ch / "rasterize_to_pixels_bwd.cu").is_file())
print("has ext.cpp:", (csrc_ch / "ext.cpp").is_file())
print("has types.cuh:", (csrc_ch / "types.cuh").is_file())

# Check version files
for name, p in [("anyplat", p_any), ("chorus", p_ch)]:
    ver_file = p / "_version.py"
    if ver_file.is_file():
        print(f"\n{name} version:", ver_file.read_text().strip())
    init_file = p / "__init__.py"
    if init_file.is_file():
        txt = init_file.read_text()
        for line in txt.split("\n"):
            if "version" in line.lower() and "=" in line:
                print(f"{name}:", line.strip())
                break

# Check anyplat env's gsplat via import
print("\n=== anyplat env gsplat import ===")
r = subprocess.run([sys.executable, "-c", "import gsplat; print(gsplat.__version__)"],
                   capture_output=True, text=True)
print("stdout:", r.stdout.strip())
print("stderr:", r.stderr.strip()[:200] if r.stderr else "")
