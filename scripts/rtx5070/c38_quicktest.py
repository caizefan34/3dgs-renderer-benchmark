#!/usr/bin/env python3
"""Quick test of C38 infrastructure: load model, run 5 iters, check state capture."""
import sys, time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from loss import combined_loss
from dataset import GTDataset

print("=== C38 Quick Test ===")
device = "cuda"
torch.manual_seed(42)

repo_root = Path(__file__).resolve().parent.parent.parent
print("Loading dataset...")
dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=device)
print(f"  {len(dataset)} cameras")

cam0 = dataset.get_camera(0)
gt0 = dataset.get_gt_image(0)
print(f"  Camera 0: {cam0.image_width}x{cam0.image_height}")
print(f"  GT image: {gt0.shape}")

ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
print(f"\nLoading checkpoint: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=device)
print(f"  Model: {model.xyz.shape[0]} Gaussians, SH degree: {model.sh_degree}")
print(f"  xyz range: [{model.xyz.min().item():.2f}, {model.xyz.max().item():.2f}]")

spatial_lr_scale = 46.64
optimizer = torch.optim.Adam([
    {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale, "eps": 1e-15, "betas": (0.9, 0.999)},
    {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
    {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
])

# Run 5 iterations and capture state
print("\nRunning 5 iterations with state capture...")
states = []
for i in range(5):
    old_xyz = model.xyz.detach().clone()

    data = model.forward()
    rendered, _, info = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
        width=cam0.image_width, height=cam0.image_height,
        tile_size=16, packed=True, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    )
    rendered_img = rendered[0].clamp(0, 1)
    loss = combined_loss(rendered_img, gt0, lambda_dssim=0.2)["loss"]

    # Capture state
    snap = {
        "means2d": info["means2d"].detach().cpu().clone(),
        "conics": info["conics"].detach().cpu().clone(),
        "depths": info["depths"].detach().cpu().clone(),
        "radii": info["radii"].detach().cpu().clone(),
        "flatten_ids": info["flatten_ids"].detach().cpu().clone(),
        "isect_offsets": info["isect_offsets"].detach().cpu().clone(),
        "n_isects": info["flatten_ids"].shape[0],
    }
    states.append(snap)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    d_xyz = (model.xyz - old_xyz).norm(dim=1).mean().item()
    print(f"  iter {i}: N={model.xyz.shape[0]} n_isect={snap['n_isects']} "
          f"loss={loss.item():.4f} |dxyz|={d_xyz:.3e}")

# Compare states
print("\n=== State comparison (iter 0 vs iter 1) ===")
s0, s1 = states[0], states[1]
print(f"  means2d shape: {s0['means2d'].shape} vs {s1['means2d'].shape}")
m2d_diff = (s0["means2d"] - s1["means2d"]).abs()
print(f"  means2d max delta: {m2d_diff.max().item():.6e}")
print(f"  means2d mean delta: {m2d_diff.mean().item():.6e}")

print(f"  radii exact eq: {s0['radii'].equal(s1['radii'])}")
print(f"  flatten_ids exact eq: {s0['flatten_ids'].equal(s1['flatten_ids'])}")
print(f"  isect_offsets exact eq: {s0['isect_offsets'].equal(s1['isect_offsets'])}")
print(f"  n_isects: {s0['n_isects']} vs {s1['n_isects']}")

# Check conics
conics_diff = (s0["conics"] - s1["conics"]).abs()
print(f"  conics max delta: {conics_diff.max().item():.6e}")

# Check depths
depths_diff = (s0["depths"] - s1["depths"]).abs()
print(f"  depths max delta: {depths_diff.max().item():.6e}")

print("\n=== Quick test PASSED ===")
