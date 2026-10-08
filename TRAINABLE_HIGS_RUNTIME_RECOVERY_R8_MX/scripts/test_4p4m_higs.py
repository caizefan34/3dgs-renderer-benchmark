#!/usr/bin/env python3
"""R8 P4: 4.4M OOM reproduction attempt"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

print("=== R8 P4: 4.4M OOM Reproduction ===\n")

# Load checkpoint
checkpoint_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
N_full = checkpoint['xyz'].shape[0]
print(f"Checkpoint: {N_full:,} Gaussians")

# Create 4.4M test state
np.random.seed(42)
N_test = 4_413_928
indices = np.sort(np.random.choice(N_full, N_test, replace=False))

device = torch.device('cuda')
xyz = checkpoint['xyz'][indices].to(device).requires_grad_(True)
shs = checkpoint['shs'][indices].to(device).requires_grad_(True)
scaling = checkpoint['scaling'][indices].to(device).requires_grad_(True)
rotation = checkpoint['rotation'][indices].to(device).requires_grad_(True)
opacity = checkpoint['opacity'][indices].to(device).requires_grad_(True)

print(f"State moved to GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Setup camera (B4 from R3)
height, width = 540, 960
fx, fy = width, height
cx, cy = width / 2, height / 2

# Batch of 4 cameras
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).repeat(4, 1, 1)
viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(4, 1, 1)

print(f"\nRunning HiGS rasterization (4.4M, B4)...")
torch.cuda.reset_peak_memory_stats()

try:
    result = rasterize_gaussian_higs_frozen(
        xyz, rotation, scaling, opacity.squeeze(-1), shs,
        viewmats=viewmat, Ks=K,
        width=width, height=height,
        sh_degree=3,
        enable_culling=True,
        use_higs_culling=True
    )

    render_colors = result['frame'].squeeze(0)
    render_alphas = result['alpha'].squeeze(0)

    print(f"   ✓ Forward SUCCESS")
    print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"   Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    # Test backward
    print(f"\nRunning backward...")
    loss = render_colors.sum() + render_alphas.sum()
    loss.backward()

    print(f"   ✓ Backward SUCCESS")
    print(f"   Final memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"\n✓✓✓ 4.4M HiGS test PASSED (NO OOM)")

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"   ✗ OOM REPRODUCED")
        print(f"   Error: {e}")
        print(f"   Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    else:
        print(f"   ✗ Other error: {e}")
        import traceback
        traceback.print_exc()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
