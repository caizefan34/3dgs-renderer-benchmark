#!/usr/bin/env python3
"""P1: Reproduce OOM with HiGS-specific function"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

print("=== P1: OOM Reproduction with HiGS ===\n")

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
xyz = checkpoint['xyz'][indices].to(device)
shs = checkpoint['shs'][indices].to(device)
scaling = checkpoint['scaling'][indices].to(device)
rotation = checkpoint['rotation'][indices].to(device)
opacity = checkpoint['opacity'][indices].to(device)

print(f"State moved to GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Try HiGS-specific function
print("\n1. Trying rasterize_gaussian_higs_frozen...")
try:
    from gsplat import rasterize_gaussian_higs_frozen
    
    height, width = 540, 960
    fx, fy = width, height
    cx, cy = width / 2, height / 2
    
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)
    viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    
    torch.cuda.reset_peak_memory_stats()
    
    render, meta = rasterize_gaussian_higs_frozen(
        means=xyz,
        quats=rotation,
        scales=scaling,
        opacities=opacity.squeeze(-1),
        colors=shs,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=3,
        render_mode="RGB+ED",
    )
    
    print(f"   ✓ SUCCESS")
    print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"   ✗ OOM as expected")
        print(f"   Error: {e}")
        print(f"   Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    else:
        print(f"   ✗ Other error: {e}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
