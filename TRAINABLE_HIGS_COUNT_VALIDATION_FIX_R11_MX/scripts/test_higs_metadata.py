#!/usr/bin/env python3
"""R11 P1: Safe count-only diagnostic"""
import sys
import torch
import numpy as np

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

print("=== R11 P1: Safe Count-Only Diagnostic ===\n")

# Load checkpoint
checkpoint = torch.load("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt", map_location='cpu', weights_only=False)
N_full = checkpoint['xyz'].shape[0]

# Test with 580K first (known to work)
np.random.seed(42)
N_test = 580_000
indices = np.sort(np.random.choice(N_full, N_test, replace=False))

device = torch.device('cuda')
xyz = checkpoint['xyz'][indices].to(device)
shs = checkpoint['shs'][indices].to(device)
scaling = checkpoint['scaling'][indices].to(device)
rotation = checkpoint['rotation'][indices].to(device)
opacity = checkpoint['opacity'][indices].to(device)

print(f"Testing with {N_test:,} Gaussians...")

# Setup camera
K = torch.tensor([[960, 0, 480], [0, 540, 270], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

result = rasterize_gaussian_higs_frozen(
    xyz, rotation, scaling, opacity.squeeze(-1), shs,
    viewmats=viewmat, Ks=K,
    width=960, height=540,
    sh_degree=3,
    enable_culling=True,
    use_higs_culling=True
)

print(f"\nResult keys: {result.keys()}")
print(f"\nMetadata: {result.get('metadata', {})}")

# Check if n_isects is in metadata
if 'metadata' in result:
    meta = result['metadata']
    print(f"\nMetadata contents:")
    for k, v in meta.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
