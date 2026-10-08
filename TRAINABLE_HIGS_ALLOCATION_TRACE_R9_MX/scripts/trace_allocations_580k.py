#!/usr/bin/env python3
"""R9 P1-P2: Allocation tracing and count-only diagnostic"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

# Patch torch to log allocations
allocation_log = []

original_empty = torch.empty
original_empty_strided = torch.empty_strided

def traced_empty(*args, **kwargs):
    result = original_empty(*args, **kwargs)
    if result.is_cuda and result.numel() > 1000000:  # Only log large allocations
        alloc_info = {
            'shape': list(result.shape),
            'dtype': str(result.dtype),
            'numel': result.numel(),
            'bytes': result.element_size() * result.numel(),
            'device': str(result.device)
        }
        allocation_log.append(alloc_info)
        print(f"ALLOCATION: {alloc_info['numel']:,} elements, {alloc_info['bytes'] / 1e9:.2f} GB, shape={alloc_info['shape']}")
    return result

torch.empty = traced_empty
torch.empty_strided = traced_empty

from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

print("=== R9 P1-P2: Allocation Tracing ===\n")

# Load checkpoint
checkpoint_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
N_full = checkpoint['xyz'].shape[0]

# Test with 580K first
np.random.seed(42)
N_test = 580_000
indices = np.sort(np.random.choice(N_full, N_test, replace=False))

device = torch.device('cuda')
xyz = checkpoint['xyz'][indices].to(device).requires_grad_(True)
shs = checkpoint['shs'][indices].to(device).requires_grad_(True)
scaling = checkpoint['scaling'][indices].to(device).requires_grad_(True)
rotation = checkpoint['rotation'][indices].to(device).requires_grad_(True)
opacity = checkpoint['opacity'][indices].to(device).requires_grad_(True)

print(f"Testing 580K with allocation tracing...")
torch.cuda.reset_peak_memory_stats()
allocation_log.clear()

try:
    result = rasterize_gaussian_higs_frozen(
        xyz, rotation, scaling, opacity.squeeze(-1), shs,
        viewmats=torch.eye(4, device=device).unsqueeze(0),
        Ks=torch.tensor([[960, 0, 480], [0, 540, 270], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0),
        width=960, height=540,
        sh_degree=3,
        enable_culling=True,
        use_higs_culling=True
    )
    
    print(f"\n✓ Forward SUCCESS")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\nLarge allocations (>1M elements): {len(allocation_log)}")
    
    # Sort by size
    allocation_log.sort(key=lambda x: x['bytes'], reverse=True)
    print("\nTop 10 allocations:")
    for i, alloc in enumerate(allocation_log[:10]):
        print(f"  {i+1}. {alloc['numel']:,} elements, {alloc['bytes'] / 1e9:.3f} GB, shape={alloc['shape']}")
    
    # Calculate total
    total_bytes = sum(a['bytes'] for a in allocation_log)
    print(f"\nTotal allocated (large tensors): {total_bytes / 1e9:.2f} GB")
    
except Exception as e:
    print(f"\n✗ FAILED: {e}")
