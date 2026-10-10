#!/usr/bin/env python3
"""
Trainable HiGS R3 Benchmark - Final Adjudication
Two cells: HN-540-B4, HN-1080-B4
"""
import sys
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
import hashlib

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
    
    # Normalize
    quats = F.normalize(quats, dim=-1)
    
    return {
        'means': means,
        'quats': quats,
        'scales': scales,
        'opacities': opacities,
        'shs': shs
    }

def get_bicycle_cameras(n_train=4, n_eval=3, width=960, height=540):
    """Generate bicycle camera poses"""
    # Simple circular camera path
    cameras = []
    
    # Training cameras
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
    
    # Eval cameras
    for i in range(n_eval):
        angle = 2 * np.pi * (i + 0.5) / n_eval
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

def benchmark_cell(gaussians, cameras, backend, cell_name, n_warmup=50, n_timed=100):
    """Benchmark a single cell"""
    print(f"\n=== {cell_name} - {backend} ===")
    
    device = 'cuda'
    
    # Move to GPU
    means = gaussians['means'].to(device).requires_grad_(True)
    quats = gaussians['quats'].to(device).requires_grad_(True)
    scales = gaussians['scales'].to(device).requires_grad_(True)
    opacities = gaussians['opacities'].to(device).requires_grad_(True)
    shs = gaussians['shs'].to(device).requires_grad_(True)
    
    # Background
    bg = torch.zeros(1, 3, device=device)
    
    # Warmup
    print(f"Warmup: {n_warmup} iterations")
    for i in range(n_warmup):
        for cam in cameras:
            K = torch.tensor([
                [cam['fx'], 0, cam['cx']],
                [0, cam['fy'], cam['cy']],
                [0, 0, 1]
            ], dtype=torch.float32, device=device)
            
            viewmat = torch.eye(4, device=device)
            viewmat[:3, :3] = cam['R'].to(device)
            viewmat[:3, 3] = cam['t'].to(device)
            
            if backend == 'std_ll':
                render_colors, render_alphas, meta = rasterization(
                    means, quats, scales, opacities, shs,
                    viewmat.unsqueeze(0), K.unsqueeze(0),
                    width=cam['width'], height=cam['height'],
                    backgrounds=bg, sh_degree=3
                )
                render_colors = render_colors.squeeze(0)
                render_alphas = render_alphas.squeeze(0)
            else:  # higs_native
                result = rasterize_gaussian_higs_frozen(
                    means, quats, scales, opacities, shs,
                    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                    width=cam['width'], height=cam['height'],
                    sh_degree=3,
                    enable_culling=True,
                    use_higs_culling=True
                )
                render_colors = result['render_colors'].squeeze(0)
                render_alphas = result['render_alphas'].squeeze(0)
            
            loss = render_colors.sum() + render_alphas.sum()
            loss.backward()
    
    torch.cuda.synchronize()
    
    # Timed iterations
    print(f"Timed: {n_timed} iterations")
    forward_times = []
    backward_times = []
    
    for i in range(n_timed):
        # Forward timing
        torch.cuda.synchronize()
        start_fwd = Event(enable_timing=True)
        end_fwd = Event(enable_timing=True)
        
        start_fwd.record()
        for cam in cameras:
            K = torch.tensor([
                [cam['fx'], 0, cam['cx']],
                [0, cam['fy'], cam['cy']],
                [0, 0, 1]
            ], dtype=torch.float32, device=device)
            
            viewmat = torch.eye(4, device=device)
            viewmat[:3, :3] = cam['R'].to(device)
            viewmat[:3, 3] = cam['t'].to(device)
            
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
                result = rasterize_gaussian_higs_frozen(
                    means, quats, scales, opacities, shs,
                    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                    width=cam['width'], height=cam['height'],
                    sh_degree=3,
                    enable_culling=True,
                    use_higs_culling=True
                )
                render_colors = result['render_colors'].squeeze(0)
                render_alphas = result['render_alphas'].squeeze(0)
        end_fwd.record()
        torch.cuda.synchronize()
        
        forward_time = start_fwd.elapsed_time(end_fwd)
        forward_times.append(forward_time)
        
        # Backward timing
        torch.cuda.synchronize()
        start_bwd = Event(enable_timing=True)
        end_bwd = Event(enable_timing=True)
        
        start_bwd.record()
        loss = render_colors.sum() + render_alphas.sum()
        loss.backward()
        end_bwd.record()
        torch.cuda.synchronize()
        
        backward_time = start_bwd.elapsed_time(end_bwd)
        backward_times.append(backward_time)
        
        # Zero gradients
        if means.grad is not None:
            means.grad = None
        if quats.grad is not None:
            quats.grad = None
        if scales.grad is not None:
            scales.grad = None
        if opacities.grad is not None:
            opacities.grad = None
        if shs.grad is not None:
            shs.grad = None
    
    # Statistics
    forward_times = np.array(forward_times)
    backward_times = np.array(backward_times)
    fb_times = forward_times + backward_times
    
    results = {
        'backend': backend,
        'cell': cell_name,
        'n_warmup': n_warmup,
        'n_timed': n_timed,
        'forward_ms': {
            'median': float(np.median(forward_times)),
            'mean': float(np.mean(forward_times)),
            'std': float(np.std(forward_times)),
            'min': float(np.min(forward_times)),
            'max': float(np.max(forward_times))
        },
        'backward_ms': {
            'median': float(np.median(backward_times)),
            'mean': float(np.mean(backward_times)),
            'std': float(np.std(backward_times)),
            'min': float(np.min(backward_times)),
            'max': float(np.max(backward_times))
        },
        'fb_cuda_ms': {
            'median': float(np.median(fb_times)),
            'mean': float(np.mean(fb_times)),
            'std': float(np.std(fb_times)),
            'min': float(np.min(fb_times)),
            'max': float(np.max(fb_times))
        }
    }
    
    print(f"Forward: {results['forward_ms']['median']:.3f} ms")
    print(f"Backward: {results['backward_ms']['median']:.3f} ms")
    print(f"FB_cuda: {results['fb_cuda_ms']['median']:.3f} ms")
    
    return results

def main():
    ckpt_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
    
    print("=== Trainable HiGS R3 Benchmark ===")
    print(f"Loading state: {ckpt_path}")
    
    gaussians = load_bicycle_state(ckpt_path)
    N = gaussians['means'].shape[0]
    print(f"N = {N}")
    
    assert 4_000_000 <= N <= 4_800_000, f"N={N} out of range"
    
    # Cell 1: HN-540-B4
    print("\n=== Cell 1: HN-540-B4 ===")
    cameras_540 = get_bicycle_cameras(n_train=1, n_eval=0, width=960, height=540)
    
    results_b_540 = benchmark_cell(gaussians, cameras_540, 'std_ll', 'HN-540-B4')
    results_h_540 = benchmark_cell(gaussians, cameras_540, 'higs_native', 'HN-540-B4')
    
    # Cell 2: HN-1080-B4
    print("\n=== Cell 2: HN-1080-B4 ===")
    cameras_1080 = get_bicycle_cameras(n_train=1, n_eval=0, width=1920, height=1080)
    
    results_b_1080 = benchmark_cell(gaussians, cameras_1080, 'std_ll', 'HN-1080-B4')
    results_h_1080 = benchmark_cell(gaussians, cameras_1080, 'higs_native', 'HN-1080-B4')
    
    # Calculate speedups
    def calc_speedup(b_res, h_res):
        return {
            'S_forward': b_res['forward_ms']['median'] / h_res['forward_ms']['median'],
            'S_backward': b_res['backward_ms']['median'] / h_res['backward_ms']['median'],
            'S_FB': b_res['fb_cuda_ms']['median'] / h_res['fb_cuda_ms']['median']
        }
    
    speedup_540 = calc_speedup(results_b_540, results_h_540)
    speedup_1080 = calc_speedup(results_b_1080, results_h_1080)
    
    print("\n=== Results ===")
    print(f"HN-540-B4: S_FB = {speedup_540['S_FB']:.3f}x")
    print(f"HN-1080-B4: S_FB = {speedup_1080['S_FB']:.3f}x")
    
    # Save results
    output = {
        'workload': {
            'N': N,
            'ckpt_path': ckpt_path
        },
        'HN-540-B4': {
            'B': results_b_540,
            'H': results_h_540,
            'speedup': speedup_540
        },
        'HN-1080-B4': {
            'B': results_b_1080,
            'H': results_h_1080,
            'speedup': speedup_1080
        }
    }
    
    with open('/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_SCALING0_R3_MX/raw/benchmark_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to raw/benchmark_results.json")

if __name__ == '__main__':
    main()
