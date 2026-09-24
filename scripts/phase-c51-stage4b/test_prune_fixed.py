#!/usr/bin/env python3
"""Test that prune_and_reset works after the fix."""
import torch
import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from gaussian_model import GaussianModel

model = GaussianModel(num_points=100, sh_degree=0, max_sh_degree=3, device="cuda")
xyz = torch.randn(100, 3, device="cuda")
# 1D opacity (as returned by load_ply)
opacity_1d = torch.logit(torch.full((100,), 0.03, device="cuda"))
model.init_from_sfm(xyz=xyz, opacity_logit=opacity_1d, shs=torch.randn(100, 1, 3, device="cuda"))

print(f"Before: opacity shape={model.opacity.shape}, GS={model.xyz.shape[0]}")

# Call prune_and_reset at step 3000
removed = model.prune_and_reset(opacity_threshold=0.005, reset_interval=3000, current_step=3000)
print(f"prune_and_reset returned: {removed}")
print(f"After: opacity shape={model.opacity.shape}, GS={model.xyz.shape[0]}")
print("SUCCESS: prune_and_reset works with 1D opacity!")
