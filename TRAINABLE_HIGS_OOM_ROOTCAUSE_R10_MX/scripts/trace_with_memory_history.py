#!/usr/bin/env python3
"""R10 P1: PyTorch allocator memory history for OOM forensics"""
import sys
import os
import torch
import numpy as np
import pickle
import traceback

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

print("=== R10 P1: PyTorch Allocator Memory History ===\n")

# Enable memory history BEFORE any CUDA operations
print("1. Enabling PyTorch memory history...")
torch.cuda.memory._record_memory_history(
    enabled='all',
    max_entries=20000
)
print("   ✓ Memory history recording enabled")

# Load checkpoint
checkpoint_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
N_full = checkpoint['xyz'].shape[0]
print(f"\n2. Checkpoint loaded: {N_full:,} Gaussians")

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

print(f"3. State moved to GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Setup camera (B4 from R3)
height, width = 540, 960
fx, fy = width, height
cx, cy = width / 2, height / 2

K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).repeat(4, 1, 1)
viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(4, 1, 1)

print(f"\n4. Running HiGS rasterization (4.4M, B4)...")
print(f"   Memory history recording...")
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
    
    print(f"   ✓ Forward SUCCESS (unexpected!)")
    print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"   ✗ OOM REPRODUCED")
        print(f"   Error: {e}")
        
        # Capture memory state
        print(f"\n5. Capturing OOM evidence...")
        
        # Save memory snapshot
        snapshot_path = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OOM_ROOTCAUSE_R10_MX/raw/pytorch_oom_snapshot.pickle"
        try:
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"   ✓ Snapshot saved: {snapshot_path}")
        except Exception as snap_err:
            print(f"   ⚠ Snapshot failed: {snap_err}")
        
        # Get memory stats
        stats = torch.cuda.memory_stats()
        print(f"\n6. Memory stats at OOM:")
        print(f"   Allocated: {stats.get('allocated_bytes.all.current', 0) / 1e9:.2f} GB")
        print(f"   Reserved: {stats.get('reserved_bytes.all.current', 0) / 1e9:.2f} GB")
        print(f"   Active: {stats.get('active_bytes.all.current', 0) / 1e9:.2f} GB")
        
        # Get device memory info
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"\n7. Device memory:")
        print(f"   Total: {total_bytes / 1e9:.2f} GB")
        print(f"   Free: {free_bytes / 1e9:.2f} GB")
        
        # Save traceback
        tb_str = traceback.format_exc()
        tb_path = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OOM_ROOTCAUSE_R10_MX/logs/oom_traceback.log"
        with open(tb_path, 'w') as f:
            f.write(tb_str)
        print(f"   ✓ Traceback saved: {tb_path}")
        
        # Save memory summary
        summary = torch.cuda.memory_summary()
        summary_path = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_OOM_ROOTCAUSE_R10_MX/logs/memory_summary.log"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"   ✓ Memory summary saved: {summary_path}")
        
    else:
        print(f"   ✗ Other error: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

finally:
    # Stop memory history
    print(f"\n8. Stopping memory history...")
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"   ✓ Recording stopped")
