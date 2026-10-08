#!/usr/bin/env python3
"""
R4 OOM Stack Trace - Identify exact allocation point
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
import traceback

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
sys.path.insert(0, '/mnt/storage_pool/liaoyuanjun/higs-13scene-env/lib/python3.10/site-packages')

import torch.nn.functional as F
from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

def load_bicycle_state(ckpt_path):
    """Load bicycle checkpoint"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    means = ckpt['xyz']
    quats = ckpt['rotation']
    scales = ckpt['scaling']
    opacities = ckpt['opacity']
    shs = ckpt['shs']
    
    quats = F.normalize(quats, dim=-1)
    
    return {
        'means': means,
        'quats': quats,
        'scales': scales,
        'opacities': opacities,
        'shs': shs
    }

def main():
    ckpt_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
    
    print("=== R4 OOM Stack Trace ===")
    print(f"Loading state: {ckpt_path}")
    
    gaussians = load_bicycle_state(ckpt_path)
    N = gaussians['means'].shape[0]
    print(f"N = {N}")
    
    device = 'cuda'
    
    # Load to GPU
    means = gaussians['means'].to(device).requires_grad_(True)
    quats = gaussians['quats'].to(device).requires_grad_(True)
    scales = gaussians['scales'].to(device).requires_grad_(True)
    opacities = gaussians['opacities'].to(device).requires_grad_(True)
    shs = gaussians['shs'].to(device).requires_grad_(True)
    
    print(f"Gaussian state loaded: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # Camera
    width, height = 960, 540
    angle = 0.0
    R = torch.tensor([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ], dtype=torch.float32, device=device)
    
    t = torch.tensor([2.0, 0.5, 2.0], dtype=torch.float32, device=device)
    
    K = torch.tensor([
        [width * 1.0, 0, width / 2],
        [0, height * 1.0, height / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    
    print(f"Camera setup complete")
    print(f"Memory before HiGS: {torch.cuda.memory_allocated() / 1e9:.3f} GB allocated")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    
    try:
        result_dict = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, shs,
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=width, height=height,
            sh_degree=3,
            enable_culling=True,
            use_higs_culling=True
        )
        print("SUCCESS - No OOM")
        
    except RuntimeError as e:
        print(f"\n=== OOM ERROR ===")
        print(f"Error: {e}")
        print(f"\n=== Full Stack Trace ===")
        traceback.print_exc()
        
        print(f"\n=== Memory State at OOM ===")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        print(f"Max reserved: {torch.cuda.max_memory_reserved() / 1e9:.3f} GB")

if __name__ == '__main__':
    main()
