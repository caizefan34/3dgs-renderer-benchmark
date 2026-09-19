#!/usr/bin/env python3
"""Probe gsplat source location on mx for R6-B patching."""
import gsplat, pathlib
p = pathlib.Path(gsplat.__file__).parent
print("gsplat_root:", p)
csrc = p / "cuda" / "csrc"
print("csrc_exists:", csrc.is_dir())
print("rasterize_bwd:", (csrc / "rasterize_to_pixels_bwd.cu").is_file())
print("ext_cpp:", (csrc / "ext.cpp").is_file())
print("types_cuh:", (csrc / "types.cuh").is_file())
