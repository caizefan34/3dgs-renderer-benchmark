#!/usr/bin/env python3
"""
R4 N-Scaling Test - Find HiGS OOM cliff
Tests multiple N values with both std_ll and HiGS
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
import gc

sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')
sys.path.insert(0, '/mnt/storage_pool/liaoyuanjun/higs-13scene-env/lib/python3.10/site-packages')

import torch.nn.functional as F
from torch.cuda import Event
from gsplat import rasterization
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

def sample_gaussians(gaussians, target_N, seed=42):
    """Deterministically sample subset of Gaussians"""
    np.random.seed(seed)
    
    N_full = gaussians['means'].shape[0]
    if target_N >= N_full:
        return gaussians
    
    indices = np.random.choice(N_full, target_N, replace=False)
    indices = torch.from_numpy(np.sort(indices))
    
    return {
        'means': gaussians['means'][indices],
        'quats': gaussians['quats'][indices],
        'scales': gaussians['scales'][indices],
        'opacities': gaussians['opacities'][indices],
        'shs': gaussians['shs'][indices]
    }

def get_bicycle_cameras(n_train=1, n_eval=0, width=960, height=540):
    """Generate bicycle camera poses"""
    cameras = []
    
    for i in range(n_train):
        angle = 2 * np.pi * i / n_train
        R = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=torch.float32)
        
        t = torch.tensor([2.0 * np.cos(angle), 0.5, 2.0 * np.sin(angle)], dtype=torch.float32)
        
        cameras.append({
            'R': R,
            't': t,
            'width': width,
            'height': height,
            'fx': width * 1.0,
            'fy': height * 1.0,
            'cx': width / 2,
            'cy': height / 2
        })
    
    return cameras

def test_backend(gaussians_subset, backend, cam):
    """Test a single backend, return peak memory or None if OOM"""
    device = 'cuda'
    torch.cuda.reset_peak_memory_stats()
    
    try:
        means = gaussians_subset['means'].to(device).requires_grad_(True)
        quats = gaussians_subset['quats'].to(device).requires_grad_(True)
        scales = gaussians_subset['scales'].to(device).requires_grad_(True)
        opacities = gaussians_subset['opacities'].to(device).requires_grad_(True)
        shs = gaussians_subset['shs'].to(device).requires_grad_(True)
        
        K = torch.tensor([
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        viewmat = torch.eye(4, device=device)
        viewmat[:3, :3] = cam['R'].to(device)
        viewmat[:3, 3] = cam['t'].to(device)
        
        bg = torch.zeros(1, 3, device=device)
        
        if backend == 'std_ll':
            render_colors, render_alphas, meta = rasterization(
                means, quats, scales, opacities, shs,
                viewmat.unsqueeze(0), K.unsqueeze(0),
                width=cam['width'], height=cam['height'],
                backgrounds=bg, sh_degree=3
            )
            render_colors = render_colors.squeeze(0)
            render_alphas = render_alphas.squeeze(0)
        else:
            result_dict = rasterize_gaussian_higs_frozen(
                means, quats, scales, opacities, shs,
                viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                width=cam['width'], height=cam['height'],
                sh_degree=3,
                enable_culling=True,
                use_higs_culling=True,
                freeze_topology=False
            )
            render_colors = result_dict['frame'].squeeze(0)
            render_alphas = result_dict['alpha'].squeeze(0)
        
        # Forward + backward
        loss = render_colors.sum() + render_alphas.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        
        del means, quats, scales, opacities, shs, render_colors, render_alphas
        gc.collect()
        torch.cuda.empty_cache()
        
        return peak_gb
        
    except RuntimeError as e:
        if 'OutOfMemory' in str(e):
            del means, quats, scales, opacities, shs
            gc.collect()
            torch.cuda.empty_cache()
            return None
        else:
            raise

def main():
    ckpt_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
    
    print("=== R4 N-Scaling Test ===")
    print(f"Loading full state: {ckpt_path}")
    
    gaussians_full = load_bicycle_state(ckpt_path)
    N_full = gaussians_full['means'].shape[0]
    print(f"Full N = {N_full}")
    
    cameras = get_bicycle_cameras(n_train=1, n_eval=0, width=960, height=540)
    cam = cameras[0]
    
    # Test N values
    N_values = [580_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 3_500_000, 4_000_000, 4_413_928]
    
    results = []
    
    for N in N_values:
        print(f"\n=== Testing N = {N:,} ===")
        
        # Sample subset
        gaussians_subset = sample_gaussians(gaussians_full, N, seed=42)
        
        # Test std_ll
        print(f"  std_ll...", end=' ', flush=True)
        std_peak = test_backend(gaussians_subset, 'std_ll', cam)
        if std_peak is not None:
            print(f"PASS ({std_peak:.2f} GB)")
        else:
            print(f"OOM")
        
        # Test HiGS
        print(f"  HiGS...", end=' ', flush=True)
        higs_peak = test_backend(gaussians_subset, 'higs', cam)
        if higs_peak is not None:
            print(f"PASS ({higs_peak:.2f} GB)")
        else:
            print(f"OOM")
        
        results.append({
            'N': N,
            'std_ll_peak_gb': std_peak,
            'higs_peak_gb': higs_peak,
            'std_ll_status': 'PASS' if std_peak is not None else 'OOM',
            'higs_status': 'PASS' if higs_peak is not None else 'OOM'
        })
        
        # If HiGS OOM, stop here
        if higs_peak is None:
            print(f"  HiGS OOM at N={N:,}, stopping N-scaling")
            break
    
    # Save results
    output = {
        'N_full': N_full,
        'resolution': '960x540',
        'batch_size': 1,
        'sampling_method': 'deterministic_random_subset_seed_42',
        'results': results
    }
    
    with open('/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_MEMSCALING_R4_MX/raw/n_scaling.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n=== Summary ===")
    for r in results:
        std_str = f"{r['std_ll_peak_gb']:.2f} GB" if r['std_ll_peak_gb'] else "OOM"
        higs_str = f"{r['higs_peak_gb']:.2f} GB" if r['higs_peak_gb'] else "OOM"
        print(f"N={r['N']:>9,}: std_ll={std_str:>10}, HiGS={higs_str:>10}")

if __name__ == '__main__':
    main()
