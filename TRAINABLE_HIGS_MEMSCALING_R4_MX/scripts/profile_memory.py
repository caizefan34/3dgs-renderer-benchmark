#!/usr/bin/env python3
"""
R4 Memory Profiling - Investigate HiGS OOM at 4.4M Gaussians
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

def get_gpu_memory():
    """Get current GPU memory stats"""
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        'max_reserved_gb': torch.cuda.max_memory_reserved() / 1e9
    }

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

def profile_std_ll(gaussians, cameras, label):
    """Profile std_ll memory usage"""
    print(f"\n=== Profiling std_ll - {label} ===")
    
    device = 'cuda'
    torch.cuda.reset_peak_memory_stats()
    
    # Load to GPU
    mem_before = get_gpu_memory()
    print(f"Before GPU load: {mem_before}")
    
    means = gaussians['means'].to(device).requires_grad_(True)
    quats = gaussians['quats'].to(device).requires_grad_(True)
    scales = gaussians['scales'].to(device).requires_grad_(True)
    opacities = gaussians['opacities'].to(device).requires_grad_(True)
    shs = gaussians['shs'].to(device).requires_grad_(True)
    
    mem_after_load = get_gpu_memory()
    print(f"After GPU load: {mem_after_load}")
    
    gaussian_state_gb = mem_after_load['allocated_gb'] - mem_before['allocated_gb']
    
    bg = torch.zeros(1, 3, device=device)
    
    # Forward pass
    torch.cuda.synchronize()
    mem_before_fwd = get_gpu_memory()
    
    cam = cameras[0]
    K = torch.tensor([
        [cam['fx'], 0, cam['cx']],
        [0, cam['fy'], cam['cy']],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = cam['R'].to(device)
    viewmat[:3, 3] = cam['t'].to(device)
    
    try:
        render_colors, render_alphas, meta = rasterization(
            means, quats, scales, opacities, shs,
            viewmat.unsqueeze(0), K.unsqueeze(0),
            width=cam['width'], height=cam['height'],
            backgrounds=bg, sh_degree=3
        )
        render_colors = render_colors.squeeze(0)
        render_alphas = render_alphas.squeeze(0)
        
        torch.cuda.synchronize()
        mem_after_fwd = get_gpu_memory()
        fwd_peak_gb = mem_after_fwd['max_allocated_gb'] - mem_before['allocated_gb']
        
        print(f"Forward peak: {fwd_peak_gb:.3f} GB")
        print(f"Forward allocated: {mem_after_fwd['allocated_gb']:.3f} GB")
        
        # Backward pass
        torch.cuda.synchronize()
        loss = render_colors.sum() + render_alphas.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        mem_after_bwd = get_gpu_memory()
        bwd_peak_gb = mem_after_bwd['max_allocated_gb'] - mem_before['allocated_gb']
        
        print(f"Backward peak: {bwd_peak_gb:.3f} GB")
        print(f"Backward allocated: {mem_after_bwd['allocated_gb']:.3f} GB")
        
        result = {
            'status': 'SUCCESS',
            'gaussian_state_gb': gaussian_state_gb,
            'forward_peak_gb': fwd_peak_gb,
            'backward_peak_gb': bwd_peak_gb,
            'total_peak_gb': max(fwd_peak_gb, bwd_peak_gb),
            'meta_keys': list(meta.keys()) if isinstance(meta, dict) else []
        }
        
    except RuntimeError as e:
        print(f"ERROR: {e}")
        mem_error = get_gpu_memory()
        result = {
            'status': 'OOM',
            'error': str(e),
            'gaussian_state_gb': gaussian_state_gb,
            'memory_at_error': mem_error
        }
    
    # Cleanup
    del means, quats, scales, opacities, shs
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def profile_higs(gaussians, cameras, label):
    """Profile HiGS memory usage"""
    print(f"\n=== Profiling HiGS - {label} ===")
    
    device = 'cuda'
    torch.cuda.reset_peak_memory_stats()
    
    # Load to GPU
    mem_before = get_gpu_memory()
    print(f"Before GPU load: {mem_before}")
    
    means = gaussians['means'].to(device).requires_grad_(True)
    quats = gaussians['quats'].to(device).requires_grad_(True)
    scales = gaussians['scales'].to(device).requires_grad_(True)
    opacities = gaussians['opacities'].to(device).requires_grad_(True)
    shs = gaussians['shs'].to(device).requires_grad_(True)
    
    mem_after_load = get_gpu_memory()
    print(f"After GPU load: {mem_after_load}")
    
    gaussian_state_gb = mem_after_load['allocated_gb'] - mem_before['allocated_gb']
    
    # Forward pass
    torch.cuda.synchronize()
    mem_before_fwd = get_gpu_memory()
    
    cam = cameras[0]
    K = torch.tensor([
        [cam['fx'], 0, cam['cx']],
        [0, cam['fy'], cam['cy']],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = cam['R'].to(device)
    viewmat[:3, 3] = cam['t'].to(device)
    
    try:
        result_dict = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, shs,
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=cam['width'], height=cam['height'],
            sh_degree=3,
            enable_culling=True,
            use_higs_culling=True
        )
        
        torch.cuda.synchronize()
        mem_after_fwd = get_gpu_memory()
        fwd_peak_gb = mem_after_fwd['max_allocated_gb'] - mem_before['allocated_gb']
        
        print(f"Forward peak: {fwd_peak_gb:.3f} GB")
        print(f"Forward allocated: {mem_after_fwd['allocated_gb']:.3f} GB")
        
        render_colors = result_dict['render_colors'].squeeze(0)
        render_alphas = result_dict['render_alphas'].squeeze(0)
        
        # Backward pass
        torch.cuda.synchronize()
        loss = render_colors.sum() + render_alphas.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        mem_after_bwd = get_gpu_memory()
        bwd_peak_gb = mem_after_bwd['max_allocated_gb'] - mem_before['allocated_gb']
        
        print(f"Backward peak: {bwd_peak_gb:.3f} GB")
        print(f"Backward allocated: {mem_after_bwd['allocated_gb']:.3f} GB")
        
        result = {
            'status': 'SUCCESS',
            'gaussian_state_gb': gaussian_state_gb,
            'forward_peak_gb': fwd_peak_gb,
            'backward_peak_gb': bwd_peak_gb,
            'total_peak_gb': max(fwd_peak_gb, bwd_peak_gb),
            'result_keys': list(result_dict.keys())
        }
        
    except RuntimeError as e:
        print(f"ERROR: {e}")
        mem_error = get_gpu_memory()
        result = {
            'status': 'OOM',
            'error': str(e),
            'gaussian_state_gb': gaussian_state_gb,
            'memory_at_error': mem_error
        }
    
    # Cleanup
    del means, quats, scales, opacities, shs
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def main():
    ckpt_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
    
    print("=== R4 Memory Profiling ===")
    print(f"Loading state: {ckpt_path}")
    
    gaussians = load_bicycle_state(ckpt_path)
    N = gaussians['means'].shape[0]
    print(f"N = {N}")
    
    cameras = get_bicycle_cameras(n_train=1, n_eval=0, width=960, height=540)
    
    # Profile std_ll
    std_result = profile_std_ll(gaussians, cameras, "540p B1")
    
    # Profile HiGS
    higs_result = profile_higs(gaussians, cameras, "540p B1")
    
    # Save results
    output = {
        'N': N,
        'resolution': '960x540',
        'batch_size': 1,
        'std_ll': std_result,
        'higs_native': higs_result
    }
    
    with open('/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_MEMSCALING_R4_MX/raw/memory_profile_4.4M.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"std_ll: {std_result['status']}")
    print(f"HiGS: {higs_result['status']}")
    
    if std_result['status'] == 'SUCCESS':
        print(f"std_ll peak: {std_result['total_peak_gb']:.3f} GB")
    if higs_result['status'] == 'SUCCESS':
        print(f"HiGS peak: {higs_result['total_peak_gb']:.3f} GB")
    elif higs_result['status'] == 'OOM':
        print(f"HiGS OOM: {higs_result.get('error', 'unknown')}")

if __name__ == '__main__':
    main()
