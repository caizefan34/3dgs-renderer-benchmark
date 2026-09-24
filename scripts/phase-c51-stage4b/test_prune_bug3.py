#!/usr/bin/env python3
"""Test prune_and_reset with 1D opacity (from load_ply)."""
import torch
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from gaussian_model import GaussianModel

model = GaussianModel(num_points=100, sh_degree=0, max_sh_degree=3, device="cuda")
xyz = torch.randn(100, 3, device="cuda")
# 1D opacity (as returned by load_ply)
opacity_1d = torch.logit(torch.full((100,), 0.03, device="cuda"))  # 1D!
model.init_from_sfm(xyz=xyz, opacity_logit=opacity_1d, shs=torch.randn(100, 1, 3, device="cuda"))

print(f"opacity shape: {model.opacity.shape}")

# Simulate prune_and_reset
opacity_threshold = 0.005
opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
print(f"opacities shape: {opacities.shape}")
near_threshold = (opacities < opacity_threshold * 10) & (opacities >= opacity_threshold)
print(f"near_threshold sum: {near_threshold.sum().item()}")

reset_count = near_threshold.sum().item()
reset_val = torch.logit(torch.full((reset_count, 1), opacity_threshold * 2, device="cuda"))
print(f"reset_val shape: {reset_val.shape}")
print(f"model.opacity.data[near_threshold] shape: {model.opacity.data[near_threshold].shape}")

try:
    model.opacity.data[near_threshold] = reset_val
    print("Assignment succeeded")
except RuntimeError as e:
    print(f"FAILED: {e}")
    # Fix: match dimensionality
    if model.opacity.dim() == 1:
        reset_val_fixed = reset_val.squeeze(-1)
    else:
        reset_val_fixed = reset_val
    try:
        model.opacity.data[near_threshold] = reset_val_fixed
        print(f"Fix (squeeze) succeeded, shape: {reset_val_fixed.shape}")
    except RuntimeError as e2:
        print(f"Fix FAILED: {e2}")
