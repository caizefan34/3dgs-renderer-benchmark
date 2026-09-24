#!/usr/bin/env python3
"""Verify all imports work in gsplat env."""
import sys
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark")
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\src")
sys.path.insert(0, r"C:\Users\36570\3dgs-renderer-benchmark\scripts\epic05\phase7")

import gsplat
print(f"gsplat version: {gsplat.__version__}")

from gsplat import rasterization
print("gsplat.rasterization: OK")

from gsplat.cuda._backend import _C
print(f"CUDA backend: OK ({type(_C).__name__})")

from gaussian_model import GaussianModel
print("GaussianModel: OK")

from loss import combined_loss
print("combined_loss: OK")

from dataset import GTDataset
print("GTDataset: OK")

print("\nAll imports successful!")
