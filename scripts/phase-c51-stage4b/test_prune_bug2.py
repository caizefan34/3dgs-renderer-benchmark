#!/usr/bin/env python3
"""Test prune_and_reset shape bug with controlled opacity values."""
import torch
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from gaussian_model import GaussianModel

model = GaussianModel(num_points=100, sh_degree=0, max_sh_degree=3, device="cuda")
xyz = torch.randn(100, 3, device="cuda")
model.init_from_sfm(xyz=xyz, shs=torch.randn(100, 1, 3, device="cuda"))

# Set opacities so some are near threshold (0.005 * 10 = 0.05)
# sigmoid(opacity_logit) = 0.03 for some, 0.9 for others
model.opacity.data[:50] = torch.logit(torch.tensor(0.03))  # near threshold
model.opacity.data[50:] = torch.logit(torch.tensor(0.9))   # well above

print(f"opacity shape: {model.opacity.shape}")

# Simulate prune_and_reset
opacity_threshold = 0.005
opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
print(f"opacities shape: {opacities.shape}")
near_threshold = (opacities < opacity_threshold * 10) & (opacities >= opacity_threshold)
print(f"near_threshold shape: {near_threshold.shape}, sum: {near_threshold.sum().item()}")

reset_count = near_threshold.sum().item()
reset_val = torch.logit(torch.full((reset_count, 1), opacity_threshold * 2, device="cuda"))
print(f"reset_val shape: {reset_val.shape}")

# Test the assignment
indexed = model.opacity.data[near_threshold]
print(f"model.opacity.data[near_threshold] shape: {indexed.shape}")

try:
    model.opacity.data[near_threshold] = reset_val
    print("Original assignment succeeded")
except RuntimeError as e:
    print(f"Original FAILED: {e}")
    # Fix 1: squeeze reset_val
    try:
        model.opacity.data[near_threshold] = reset_val.squeeze(-1)
        print("Fix 1 (squeeze reset_val) succeeded")
    except RuntimeError as e2:
        print(f"Fix 1 FAILED: {e2}")
    # Fix 2: unsqueeze near_threshold
    try:
        model.opacity.data[near_threshold.unsqueeze(-1)] = reset_val
        print("Fix 2 (unsqueeze mask) succeeded")
    except RuntimeError as e3:
        print(f"Fix 2 FAILED: {e3}")
