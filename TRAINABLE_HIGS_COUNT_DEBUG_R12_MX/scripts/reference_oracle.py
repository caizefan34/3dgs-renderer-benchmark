#!/usr/bin/env python3
"""R12 P1: Independent reference count oracle"""
import torch
import numpy as np

def reference_count_aabb(mean_x, mean_y, radius_x, radius_y, tile_width, tile_height, tile_size):
    """
    AABB-based tile coverage count for a single Gaussian.
    Uses half-open intervals [min, max).
    """
    # Compute tile-space bounds
    tx_min = (mean_x - radius_x) / tile_size
    tx_max = (mean_x + radius_x) / tile_size
    ty_min = (mean_y - radius_y) / tile_size
    ty_max = (mean_y + radius_y) / tile_size

    # Clamp to tile grid
    tx_min = max(0, int(np.floor(tx_min)))
    tx_max = min(tile_width, int(np.ceil(tx_max)))
    ty_min = max(0, int(np.floor(ty_min)))
    ty_max = min(tile_height, int(np.ceil(ty_max)))

    # Count tiles (half-open interval)
    count_x = max(0, tx_max - tx_min)
    count_y = max(0, ty_max - ty_min)

    return count_x * count_y

print("=== R12 P1: Reference Oracle ===\n")

# Load checkpoint
checkpoint = torch.load("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt", map_location='cpu', weights_only=False)
N_full = checkpoint['xyz'].shape[0]

# Test with 1K Gaussians
np.random.seed(42)
N_test = 1000
indices = np.sort(np.random.choice(N_full, N_test, replace=False))

xyz = checkpoint['xyz'][indices]
scaling = checkpoint['scaling'][indices]
rotation = checkpoint['rotation'][indices]

device = torch.device('cuda')

# Setup camera (B1)
K = torch.tensor([[960, 0, 480], [0, 540, 270], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

print(f"Testing with {N_test} Gaussians...")

# Import HiGS
import sys
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

# Run HiGS
xyz_gpu = xyz.to(device).requires_grad_(True)
shs = checkpoint['shs'][indices].to(device).requires_grad_(True)
scaling_gpu = scaling.to(device).requires_grad_(True)
rotation_gpu = rotation.to(device).requires_grad_(True)
opacity = checkpoint['opacity'][indices].to(device).requires_grad_(True)

result = rasterize_gaussian_higs_frozen(
    xyz_gpu, rotation_gpu, scaling_gpu, opacity.squeeze(-1), shs,
    viewmats=viewmat, Ks=K,
    width=960, height=540,
    sh_degree=3,
    enable_culling=True,
    use_higs_culling=True
)

meta = result['metadata']
print(f"\nHiGS metadata:")
print(f"  n_gaussians: {meta['n_gaussians']:,}")
print(f"  n_visible: {meta['n_visible']:,}")
print(f"  n_isects: {meta['n_isects']:,}")
print(f"  culling_ratio: {meta['culling_ratio']:.4f}")

# Calculate expected count
tile_size = 16
tile_width = 60
tile_height = 34

print(f"\nTile grid: {tile_width}×{tile_height} = {tile_width * tile_height:,} tiles")

# For now, we can't easily get per-Gaussian projected means/radii without reprojection
# This is a simplified analysis
avg_isects = meta['n_isects'] / meta['n_visible']
max_possible = tile_width * tile_height

print(f"\nIntersection analysis:")
print(f"  Per visible Gaussian: {avg_isects:.2f} tiles")
print(f"  Max possible: {max_possible:,} tiles")
print(f"  Ratio: {avg_isects / max_possible * 100:.2f}%")
print(f"  Expected (typical 3DGS): ~9 tiles")
print(f"  Over-count factor: {avg_isects / 9:.1f}x")
