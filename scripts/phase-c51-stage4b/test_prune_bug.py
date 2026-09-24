#!/usr/bin/env python3
"""Test prune_and_reset shape bug."""
import torch
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from gaussian_model import GaussianModel

# Create a model with 100 Gaussians
model = GaussianModel(num_points=100, sh_degree=0, max_sh_degree=3, device="cuda")
import numpy as np
xyz = torch.randn(100, 3, device="cuda")
model.init_from_sfm(xyz=xyz, shs=torch.randn(100, 1, 3, device="cuda"))

print(f"opacity shape: {model.opacity.shape}")
print(f"opacity dim: {model.opacity.dim()}")

# Simulate what prune_and_reset does
opacity_threshold = 0.005
opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
print(f"opacities shape: {opacities.shape}")
near_threshold = (opacities < opacity_threshold * 10) & (opacities >= opacity_threshold)
print(f"near_threshold shape: {near_threshold.shape}")
print(f"near_threshold sum: {near_threshold.sum().item()}")

if near_threshold.sum() > 0:
    reset_count = near_threshold.sum().item()
    reset_val = torch.logit(torch.full((reset_count, 1), opacity_threshold * 2, device="cuda"))
    print(f"reset_val shape: {reset_val.shape}")
    try:
        model.opacity.data[near_threshold] = reset_val
        print("Assignment succeeded!")
    except RuntimeError as e:
        print(f"Error: {e}")
        # Try fix: unsqueeze near_threshold
        try:
            model.opacity.data[near_threshold.unsqueeze(1)] = reset_val
            print("Fix with unsqueeze succeeded!")
        except RuntimeError as e2:
            print(f"Fix also failed: {e2}")
            # Try fix 2: use 1D reset_val
            try:
                model.opacity.data[near_threshold] = reset_val.squeeze(-1)
                print("Fix with squeeze succeeded!")
            except RuntimeError as e3:
                print(f"Fix also failed: {e3}")
