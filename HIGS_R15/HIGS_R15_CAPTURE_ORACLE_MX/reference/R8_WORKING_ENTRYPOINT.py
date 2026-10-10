#!/usr/bin/env python3
"""R8 P2: 580K HiGS smoke test"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

print("=== R8 P2: 580K HiGS Smoke Test ===\n")

# Load checkpoint
checkpoint_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
N_full = checkpoint['xyz'].shape[0]
print(f"Checkpoint: {N_full:,} Gaussians")

# Create 580K test state
np.random.seed(42)
N_test = 580_000
indices = np.sort(np.random.choice(N_full, N_test, replace=False))

device = torch.device('cuda')
xyz = checkpoint['xyz'][indices].to(device).requires_grad_(True)
shs = checkpoint['shs'][indices].to(device).requires_grad_(True)
scaling = checkpoint['scaling'][indices].to(device).requires_grad_(True)
rotation = checkpoint['rotation'][indices].to(device).requires_grad_(True)
opacity = checkpoint['opacity'][indices].to(device).requires_grad_(True)

print(f"State moved to GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Setup camera (from R3)
height, width = 540, 960
fx, fy = width, height
cx, cy = width / 2, height / 2

K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

print(f"\nRunning HiGS rasterization (580K)...")
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
    
    print(f"   Result keys: {result.keys()}")
    print(f"   Result type: {type(result)}")

    # Try different key names
    if 'render_colors' in result:
        render_colors = result['render_colors'].squeeze(0)
        render_alphas = result['render_alphas'].squeeze(0)
    elif 'colors' in result:
        render_colors = result['colors'].squeeze(0)
        render_alphas = result['alphas'].squeeze(0)
    elif 'frame' in result:
        render_colors = result['frame'].squeeze(0)
        render_alphas = result['alpha'].squeeze(0)
    else:
        print(f"   ✗ Unknown result structure")
        sys.exit(1)
    
    print(f"   ✓ Forward SUCCESS")
    print(f"   Render colors shape: {render_colors.shape}")
    print(f"   Render alphas shape: {render_alphas.shape}")
    print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # Test backward
    print(f"\nRunning backward...")
    loss = render_colors.sum() + render_alphas.sum()
    loss.backward()
    
    print(f"   ✓ Backward SUCCESS")
    print(f"   xyz.grad shape: {xyz.grad.shape}")
    print(f"   Final memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print(f"\n✓✓✓ 580K HiGS test PASSED")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
