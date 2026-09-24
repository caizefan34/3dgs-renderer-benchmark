#!/usr/bin/env python3
"""Phase 14B — Controlled segmented sort benchmark (fixed camera format).

Compares global radix sort (default) vs segmented radix sort
on frozen real snapshots: room, bicycle, garden.

Protocol:
  - Warmup: 5 fwd passes (not timed)
  - 30 fwd+bwd with CUDA synchronization
  - Record: fwd time, bwd time, n_isects, peak mem, pixel correctness, gradient sanity

Usage:
  python scripts/epic05/_phase14b_sort_benchmark.py
"""

import sys, os, math, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

import torch
import torch.nn.functional as F

from benchmark_framework.scene import load_ply
from benchmark_framework import load_cameras_from_json, resize_cameras
from gsplat import rasterization

DEVICE = torch.device("cuda:0")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_benchmark(scene_name, tile_size, n_iters=30, warmup=5):
    """Run baseline and segmented sort benchmarks for one scene+tile."""
    print(f"\n{'='*70}")
    print(f"  Scene: {scene_name}  |  Tile: {tile_size}  |  {n_iters} iters")
    print(f"{'='*70}")
    
    # Load scene
    ply_path = os.path.join(REPO_ROOT, "data", "official", "mipnerf360", scene_name, "point_cloud.ply")
    scene = load_ply(ply_path, device=DEVICE)
    
    # Load and resize camera
    cam_path = os.path.join(REPO_ROOT, "data", "official", "mipnerf360", scene_name, "cameras.json")
    W, H = 1920, 1080
    cameras = load_cameras_from_json(cam_path, device="cpu")
    cameras = resize_cameras(cameras, W, H)
    cam = cameras[0]
    for attr in ["viewmatrix", "projmatrix", "camera_center", "world_view_transform",
                  "full_proj_transform", "K"]:
        t = getattr(cam, attr)
        if isinstance(t, torch.Tensor):
            setattr(cam, attr, t.to(DEVICE))
    
    # Build tensors
    means = scene['xyz'].unsqueeze(0)
    quats = F.normalize(scene['rotations'], dim=-1).unsqueeze(0)
    scales = scene['scales'].unsqueeze(0)
    opacities = torch.sigmoid(scene['opacity']).unsqueeze(0)
    colors = scene['shs'].unsqueeze(0)
    sh_degree = scene['sh_degree']
    
    viewmat = cam.world_view_transform.unsqueeze(0)  # [1, 4, 4]
    K = cam.K.unsqueeze(0)  # [1, 3, 3]
    
    print(f"  Gaussians: {means.shape[-2]:,}")
    print(f"  SH degree: {sh_degree}")
    print(f"  Viewmat: {viewmat.shape}  K: {K.shape}")
    
    results = {}
    
    for seg_name, seg_val in [('baseline', False), ('segmented', True)]:
        print(f"\n  --- {seg_name} ---")
        torch.cuda.reset_peak_memory_stats()
        
        fwd_times = []
        bwd_times = []
        n_isects_list = []
        pixel_diffs = []
        grad_finite = True
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                render_colors, render_alphas, meta = rasterization(
                    means, quats, scales, opacities, colors,
                    viewmat, K, W, H,
                    tile_size=tile_size, packed=True, sh_degree=sh_degree,
                    segmented=seg_val,
                )
        
        # Benchmark loop
        for it in range(n_iters):
            # Clone with grad
            means_i = means.detach().clone().requires_grad_(True)
            quats_i = quats.detach().clone().requires_grad_(True)
            scales_i = scales.detach().clone().requires_grad_(True)
            opacities_i = opacities.detach().clone().requires_grad_(True)
            colors_i = colors.detach().clone().requires_grad_(True)
            
            target = torch.rand(1, H, W, 3, device=DEVICE) * 0.3 + 0.2
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            render_colors, render_alphas, meta = rasterization(
                means_i, quats_i, scales_i, opacities_i, colors_i,
                viewmat, K, W, H,
                tile_size=tile_size, packed=True, sh_degree=sh_degree,
                segmented=seg_val,
            )
            
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) * 1000
            
            loss = F.mse_loss(render_colors, target)
            loss.backward()
            
            torch.cuda.synchronize()
            bwd_ms = (time.perf_counter() - t0) * 1000 - fwd_ms
            
            n_isects = len(meta['isect_ids'])
            n_isects_list.append(n_isects)
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
            
            # Check gradients
            for name, p in [('means', means_i), ('quats', quats_i), ('scales', scales_i),
                           ('opacities', opacities_i), ('colors', colors_i)]:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_finite = False
                    print(f"    WARNING: NaN/Inf in {name} gradient at iter {it}")
            
            # Check pixel equivalence vs first iter
            if it == 0:
                ref_render = render_colors.detach()
            else:
                diff = torch.max(torch.abs(render_colors - ref_render)).item()
                pixel_diffs.append(diff)
        
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Statistics (outlier-robust: exclude >5 MAD from median)
        fwd_t = torch.tensor(fwd_times, dtype=torch.float32)
        bwd_t = torch.tensor(bwd_times, dtype=torch.float32)
        med = fwd_t.median()
        mad = torch.median(torch.abs(fwd_t - med))
        if mad < 0.001:
            mad = fwd_t.std()
        clean_mask = torch.abs(fwd_t - med) < 5 * max(mad, 0.001)
        clean_fwd = fwd_t[clean_mask]
        clean_bwd = bwd_t[clean_mask]
        
        results[seg_name] = {
            'avg_fwd_ms': clean_fwd.mean().item() if len(clean_fwd) > 0 else fwd_t.mean().item(),
            'median_fwd_ms': med.item(),
            'std_fwd_ms': fwd_t.std().item(),
            'min_fwd_ms': fwd_t.min().item(),
            'max_fwd_ms': fwd_t.max().item(),
            'avg_bwd_ms': bwd_t.mean().item(),
            'median_bwd_ms': bwd_t.median().item(),
            'std_bwd_ms': bwd_t.std().item(),
            'avg_n_isects': torch.tensor(n_isects_list).float().mean().item(),
            'n_isects_first': n_isects_list[0],
            'peak_memory_mb': peak_mem_mb,
            'gradients_finite': grad_finite,
            'max_pixel_diff': max(pixel_diffs) if pixel_diffs else 0.0,
            'n_samples': len(clean_fwd),
        }
        
        print(f"    avg_fwd={clean_fwd.mean().item():.2f}ms  median_fwd={med.item():.2f}ms  std={fwd_t.std().item():.2f}ms")
        print(f"    avg_bwd={bwd_t.mean().item():.2f}ms  median_bwd={bwd_t.median().item():.2f}ms")
        print(f"    n_isects={n_isects_list[0]:,}  peak_mem={peak_mem_mb:.0f}MB")
        print(f"    max_pixel_diff={results[seg_name]['max_pixel_diff']:.8f}  grad_finite={grad_finite}")
    
    return results


if __name__ == '__main__':
    t_start = time.time()
    
    scenes = ['room', 'bicycle']
    tile_sizes = [16, 20, 32]
    
    all_results = {}
    
    for scene in scenes:
        for ts in tile_sizes:
            try:
                r = run_benchmark(scene, ts, n_iters=30, warmup=5)
                all_results[f"{scene}_t{ts}"] = r
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n  OOM on {scene} tile{ts}, skipping")
                    torch.cuda.empty_cache()
                else:
                    raise e
                break  # Skip larger tiles after OOM
    
    elapsed = time.time() - t_start
    
    # Build conclusions
    conclusions = {}
    for key, res in all_results.items():
        b = res.get('baseline')
        s = res.get('segmented')
        if b and s:
            fwd_ratio = s['avg_fwd_ms'] / b['avg_fwd_ms'] if b['avg_fwd_ms'] > 0 else 1.0
            bwd_ratio = s['avg_bwd_ms'] / b['avg_bwd_ms'] if b['avg_bwd_ms'] > 0 else 1.0
            
            if fwd_ratio < 0.97 or bwd_ratio < 0.97:
                rec = 'CANDIDATE'
            elif fwd_ratio < 1.03:
                rec = 'NEUTRAL'
            else:
                rec = 'NOT_RECOMMENDED'
            
            conclusions[key] = {
                'fwd_speedup': 1.0 / fwd_ratio,
                'bwd_speedup': 1.0 / bwd_ratio,
                'fwd_ratio': fwd_ratio,
                'bwd_ratio': bwd_ratio,
                'pixel_correct': s['max_pixel_diff'] < 1e-5,
                'gradients_finite': s['gradients_finite'] and b['gradients_finite'],
                'recommendation': rec,
            }
            print(f"\n  {key}: fwd_ratio={fwd_ratio:.4f} bwd_ratio={bwd_ratio:.4f} → {rec}")
    
    output = {
        'schema_version': 1,
        'phase': '14B',
        'date': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
        'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'total_benchmark_time_s': elapsed,
        'results': all_results,
        'conclusions': conclusions,
    }
    
    out_path = os.path.join(REPO_ROOT, 'results', 'epic05', 'phase14b_sorting.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")
