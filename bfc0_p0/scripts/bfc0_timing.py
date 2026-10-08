#!/usr/bin/env python3
"""BFC0 DSH-6: Timing and economics analysis."""
import os, sys, json, time
import torch
import numpy as np

def measure_baseline_timing(
    scene_data,
    device="cuda",
    tile_size=16,
    n_warmup=10,
    n_measure=100,
):
    """Measure baseline timing for forward/backward/full iteration."""
    import gsplat
    from gsplat import rasterization
    
    points_xyz = torch.from_numpy(scene_data["points_xyz"]).to(device)
    N = len(points_xyz)
    
    # Initialize parameters
    means = points_xyz.clone()
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.05 + 0.01
    opacities = torch.sigmoid(torch.randn(N, device=device) * 0.5)
    colors = torch.rand(N, 3, device=device)
    
    for p in [means, quats, scales, opacities, colors]:
        p.requires_grad_(True)
    
    cam_data = scene_data["cameras"][0]
    viewmats = torch.from_numpy(cam_data["viewmat"]).to(device).unsqueeze(0)
    Ks = torch.from_numpy(cam_data["K"]).to(device).unsqueeze(0)
    width = cam_data["width"]
    height = cam_data["height"]
    
    # Warmup
    for _ in range(n_warmup):
        render_colors, render_alphas, meta = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks,
            width=width, height=height,
            tile_size=tile_size, render_mode="RGB", packed=False,
        )
        loss = render_colors.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Measure forward
    fwd_times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        render_colors, render_alphas, meta = rasterization(
            means=means.detach(), quats=quats.detach(), scales=scales.detach(),
            opacities=opacities.detach(), colors=colors.detach(),
            viewmats=viewmats, Ks=Ks,
            width=width, height=height,
            tile_size=tile_size, render_mode="RGB", packed=False,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)
    
    # Measure backward
    bwd_times = []
    for _ in range(n_measure):
        render_colors, render_alphas, meta = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks,
            width=width, height=height,
            tile_size=tile_size, render_mode="RGB", packed=False,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = render_colors.sum()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bwd_times.append(t1 - t0)
    
    # Measure full iteration
    full_times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        render_colors, render_alphas, meta = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks,
            width=width, height=height,
            tile_size=tile_size, render_mode="RGB", packed=False,
        )
        loss = render_colors.sum()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        full_times.append(t1 - t0)
    
    results = {
        "fwd_median_ms": float(np.median(fwd_times) * 1000),
        "fwd_mean_ms": float(np.mean(fwd_times) * 1000),
        "fwd_std_ms": float(np.std(fwd_times) * 1000),
        "bwd_median_ms": float(np.median(bwd_times) * 1000),
        "bwd_mean_ms": float(np.mean(bwd_times) * 1000),
        "bwd_std_ms": float(np.std(bwd_times) * 1000),
        "full_median_ms": float(np.median(full_times) * 1000),
        "full_mean_ms": float(np.mean(full_times) * 1000),
        "full_std_ms": float(np.std(full_times) * 1000),
        "n_measurements": n_measure,
        "n_warmup": n_warmup,
        "gpu": torch.cuda.get_device_name(0),
    }
    
    return results

def compute_economics(timing_results, oracle_results):
    """Compute economic bounds for BFC0."""
    
    bwd_ms = timing_results["bwd_median_ms"]
    full_ms = timing_results["full_median_ms"]
    
    # Extract oracle metrics
    o2_global = oracle_results.get("O2_EXACT_VALIDITY_GLOBAL", {})
    o1_global = oracle_results.get("O1_FRONTIER_GLOBAL", {})
    
    removable_fraction = o2_global.get("warp_reduction_bound", 0)
    
    # Optimistic bound: zero overhead
    t_new_bwd_opt = bwd_ms * (1 - removable_fraction)
    t_new_full_opt = full_ms - (bwd_ms - t_new_bwd_opt)
    
    # Conservative bound: charge 50% overhead on compaction
    overhead_factor = 1.5
    t_new_bwd_cons = bwd_ms * (1 - removable_fraction / overhead_factor)
    t_new_full_cons = full_ms - (bwd_ms - t_new_bwd_cons)
    
    # Break-even: what overhead makes it worthwhile
    # Need: bwd_ms * (1 - removable_fraction / (1 + overhead)) < bwd_ms
    # => removable_fraction / (1 + overhead) > 0
    # => overhead < infinity (always beneficial if removable_fraction > 0)
    # More realistic: break-even when compaction cost = savings
    # compaction_cost = bwd_ms * removable_fraction / (1 + overhead)
    # savings = bwd_ms * removable_fraction
    # break-even: overhead = removable_fraction / (1 - removable_fraction) - 1
    
    economics = {
        "baseline_bwd_ms": bwd_ms,
        "baseline_full_ms": full_ms,
        "removable_fraction": removable_fraction,
        "optimistic_bwd_ms": t_new_bwd_opt,
        "optimistic_full_ms": t_new_full_opt,
        "optimistic_bwd_speedup": bwd_ms / t_new_bwd_opt if t_new_bwd_opt > 0 else float('inf'),
        "optimistic_full_speedup": full_ms / t_new_full_opt if t_new_full_opt > 0 else float('inf'),
        "conservative_bwd_ms": t_new_bwd_cons,
        "conservative_full_ms": t_new_full_cons,
        "conservative_bwd_speedup": bwd_ms / t_new_bwd_cons if t_new_bwd_cons > 0 else float('inf'),
        "conservative_full_speedup": full_ms / t_new_full_cons if t_new_full_cons > 0 else float('inf'),
    }
    
    return economics

if __name__ == "__main__":
    out_dir = os.environ.get("BFC0_OUT_DIR", "C:\\Users\\36570\\3dgs-renderer-benchmark\\bfc0_p0\\output")
    
    # Load scene data
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "C:\\Users\\36570\\3dgs-renderer-benchmark\\data\\datasets\\mipnerf360\\room"
    
    sys.path.insert(0, os.path.dirname(__file__))
    from bfc0_trace_collector import collect_trace
    
    print(f"Collecting trace from {scene_dir}...")
    scene_data = collect_trace(scene_dir, max_cameras=1)
    
    print("Measuring baseline timing...")
    timing = measure_baseline_timing(scene_data)
    
    # Save timing
    timing_path = os.path.join(out_dir, "BFC0_TIMING.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    
    print(json.dumps(timing, indent=2))
