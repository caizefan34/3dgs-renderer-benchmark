#!/usr/bin/env python3
"""Patch only Rasterization.cpp — fix for the 2DGS scope issue."""
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/phase-c51-stage4a")
from patch_cuda import patch_rasterization_cpp
patch_rasterization_cpp()
print("Done patching Rasterization.cpp")
