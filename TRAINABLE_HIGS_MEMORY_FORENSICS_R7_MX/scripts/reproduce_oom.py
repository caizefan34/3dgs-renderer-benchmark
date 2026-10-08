#!/usr/bin/env python3
"""P1: Reproduce the exact OOM at 4.4M"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

print("=== P1: OOM Reproduction ===\n")

# Check GPU
print("1. GPU Environment:")
if torch.cuda.is_available():
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU: {props.name}")
    print(f"   Total memory: {props.total_memory / 1e9:.2f} GB")
    
    # Check free memory
    torch.cuda.reset_peak_memory_stats()
    free_before = torch.cuda.mem_get_info()
    print(f"   Free before: {free_before[0] / 1e9:.2f} GB")
    print(f"   Total: {free_before[1] / 1e9:.2f} GB")
else:
    print("   ✗ CUDA not available")
    sys.exit(1)

# Load checkpoint
print("\n2. Loading checkpoint...")
checkpoint_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
print(f"   Path: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print(f"   ✗ Checkpoint not found")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
N_full = checkpoint['xyz'].shape[0]
print(f"   ✓ Loaded checkpoint")
print(f"   Total Gaussians: {N_full:,}")

# Create test state with 4.4M
print("\n3. Creating 4.4M test state...")
np.random.seed(42)
N_test = 4_413_928
indices = np.sort(np.random.choice(N_full, N_test, replace=False))
print(f"   Test N: {N_test:,}")

# Move to GPU
print("\n4. Moving to GPU...")
xyz = checkpoint['xyz'][indices].to(device)
shs = checkpoint['shs'][indices].to(device)
scaling = checkpoint['scaling'][indices].to(device)
rotation = checkpoint['rotation'][indices].to(device)
opacity = checkpoint['opacity'][indices].to(device)

print(f"   ✓ State moved to GPU")
print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Try to run HiGS
print("\n5. Attempting HiGS rasterization...")
try:
    from gsplat import rasterization
    
    # Create dummy camera
    height, width = 540, 960
    fx, fy = width, height
    cx, cy = width / 2, height / 2
    
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)
    viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    
    print(f"   Resolution: {width}x{height}")
    print(f"   Calling rasterization...")
    
    # This should trigger the OOM
    render_colors, render_alphas, meta = rasterization(
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
    
    print(f"   ✓ SUCCESS (unexpected!)")
    print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"   ✗ OOM as expected")
        print(f"   Error: {e}")
        print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"   Reserved memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    else:
        print(f"   ✗ Other error: {e}")
        
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

print("\n6. Memory state after OOM...")
free_after = torch.cuda.mem_get_info()
print(f"   Free after: {free_after[0] / 1e9:.2f} GB")
print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
