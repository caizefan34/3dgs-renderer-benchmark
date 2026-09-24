"""
Phase 17B — C17-1 Tile-Local Bounded Queues
============================================
Step 0: Per-tile intersection capacity analysis.

For each scene (room, bicycle, garden) × tile_size (16, 20, 32):
  1. Run the existing gsplat rasterization (baseline, no modifications)
  2. Extract isect_ids from meta
  3. Decode tile_id from each intersection
  4. Count per-tile intersection counts
  5. Report: mean, median, p95, p99, max, total

Also analyzes:
  - How many tiles would overflow at capacities 512, 1024, 2048, 4096, 8192
  - Whether global materialization is truly eliminated by C17-1's approach

This script modifies NOTHING — it reads the existing renderer output.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework.scene import load_ply

# ── Configuration ──────────────────────────────────────────────────────────
SCENES = ["room", "bicycle", "garden"]
TILE_SIZES = [16, 20, 32]
CAPACITIES_TO_CHECK = [512, 1024, 2048, 4096, 8192]
DEVICE = "cuda"
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase17b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Camera (first camera from the MipNeRF 360 dataset)
# These match the dataset used in earlier phases.
CAMERA_CONFIGS = {
    "room":   {"width": 1080, "height": 1080, "K_scale": 0.9},
    "bicycle": {"width": 1920, "height": 1080, "K_scale": 0.9},
    "garden":  {"width": 1920, "height": 1080, "K_scale": 0.9},
}

# ── Main analysis ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_analysis(scene_name: str, tile_size: int) -> dict:
    """Run baseline forward pass and extract per-tile intersection statistics."""
    print(f"\n{'='*70}")
    print(f"Scene: {scene_name}, tile_size={tile_size}")
    print(f"{'='*70}")
    
    # Device
    torch.cuda.reset_peak_memory_stats()
    
    # Load scene
    ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"
    scene = load_ply(str(ply_path), device=DEVICE)
    
    N = scene["num_points"]
    sh_degree = scene.get("sh_degree", 3)
    
    # Camera setup (same as earlier phases)
    cfg = CAMERA_CONFIGS[scene_name]
    W, H = cfg["width"], cfg["height"]
    fx = W * cfg["K_scale"]
    fy = H * cfg["K_scale"]
    cx, cy = W / 2, H / 2
    
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], 
                     dtype=torch.float32, device=DEVICE).unsqueeze(0)
    
    # Camera at z=4 (standard position used in previous benchmarks)
    viewmat = torch.eye(4, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    
    # Warmup
    from gsplat import rasterization
    _ = rasterization(
        means=scene["xyz"], quats=quats, scales=scales,
        opacities=opacities, colors=scene["shs"],
        viewmats=viewmat, Ks=K,
        width=W, height=H,
        tile_size=tile_size, packed=True,
        sh_degree=sh_degree,
        render_mode="RGB",
    )
    torch.cuda.synchronize()
    
    # Timed run + capture meta
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    renders, alphas, meta = rasterization(
        means=scene["xyz"], quats=quats, scales=scales,
        opacities=opacities, colors=scene["shs"],
        viewmats=viewmat, Ks=K,
        width=W, height=H,
        tile_size=tile_size, packed=True,
        sh_degree=sh_degree,
        render_mode="RGB",
    )
    end_evt.record()
    end_evt.synchronize()
    fwd_ms = start_evt.elapsed_time(end_evt)
    
    # ── Extract intersection data ──────────────────────────────────────
    isect_ids = meta["isect_ids"]  # [n_isects] int64 keys
    n_isects = isect_ids.shape[0]
    
    tile_width = meta["tile_width"]
    tile_height = meta["tile_height"]
    n_tiles = tile_width * tile_height
    tw, th = tile_width, tile_height
    
    # Decode: isect_ids layout (from C17-1 patch / existing code):
    # C1: depth_upper(16) | tile_id(tn) | iid_enc(1)
    # Baseline: depth(32) | tile_id(tn) | iid_enc(1)
    # Both encode tile_id in the same bit position after depth
    # Let's decode generically using the current gsplat key format
    # Current gsplat: bits [32+tile_n_bits-1:32] = tile_id
    
    tile_n_bits = max(int(math.ceil(math.log2(n_tiles))), 1)
    image_n_bits = 1  # single camera
    
    # Decode tile_id from isect_ids
    # Current (C1-patched) layout: depth_upper(16) | tile_id(tn) | iid_enc
    # tile_id is at bits [16+tn-1:16]
    # Baseline layout would have tile_id at bits [32+tn-1:32]
    # Try both and see which gives valid tile_ids
    tile_ids_c1 = (isect_ids >> 16) & ((1 << tile_n_bits) - 1)
    tile_ids_baseline = (isect_ids >> 32) & ((1 << tile_n_bits) - 1)
    
    tile_ids_c1_np = tile_ids_c1.cpu().numpy().astype(np.int64)
    tile_ids_baseline_np = tile_ids_baseline.cpu().numpy().astype(np.int64)
    
    max_c1 = tile_ids_c1_np.max() if len(tile_ids_c1_np) > 0 else 0
    max_bl = tile_ids_baseline_np.max() if len(tile_ids_baseline_np) > 0 else 0
    
    print(f"  tile_id range (C1 decode): [{tile_ids_c1_np.min()}, {max_c1}]")
    print(f"  tile_id range (baseline decode): [{tile_ids_baseline_np.min()}, {max_bl}]")
    
    # Auto-detect which key format is in use
    if max_c1 < n_tiles:
        tile_ids_from_key = tile_ids_c1_np
        key_format = "C1"
    elif max_bl < n_tiles:
        tile_ids_from_key = tile_ids_baseline_np
        key_format = "baseline"
    else:
        # Fallback: try direct unpacking of key
        print("  WARNING: Could not auto-detect key format, trying raw bit inspection")
        tile_ids_from_key = tile_ids_c1_np  # Assume C1
        key_format = "C1 (fallback)"
    
    print(f"  Detected key format: {key_format}")
    
    # Verify tile_id range
    max_tid = tile_ids_from_key.max() if len(tile_ids_from_key) > 0 else 0
    min_tid = tile_ids_from_key.min() if len(tile_ids_from_key) > 0 else 0
    print(f"  tile_width={tile_width}, tile_height={tile_height}, n_tiles={n_tiles}")
    print(f"  tile_n_bits={tile_n_bits}")
    print(f"  tile_id range: [{min_tid}, {max_tid}] (valid: [0, {n_tiles-1}])")
    assert max_tid < n_tiles, f"tile_id {max_tid} >= n_tiles {n_tiles}"
    assert min_tid >= 0, f"tile_id {min_tid} < 0"
    
    # ── Per-tile intersection counts ───────────────────────────────────
    tile_counts = np.bincount(tile_ids_from_key, minlength=n_tiles).astype(np.int64)
    
    # Count tiles with zero intersections
    zero_tiles = np.sum(tile_counts == 0)
    
    # Statistics
    stats = {
        "n_isects": int(n_isects),
        "n_tiles": n_tiles,
        "zero_tiles": int(zero_tiles),
        "mean": float(np.mean(tile_counts)),
        "median": float(np.median(tile_counts)),
        "p95": float(np.percentile(tile_counts, 95)),
        "p99": float(np.percentile(tile_counts, 99)),
        "max": int(np.max(tile_counts)),
        "min": int(np.min(tile_counts[tile_counts > 0])) if np.any(tile_counts > 0) else 0,
        "std": float(np.std(tile_counts)),
    }
    print(f"  n_isects={n_isects:,}")
    print(f"  Per-tile: mean={stats['mean']:.1f}, median={stats['median']:.1f}, "
          f"p95={stats['p95']:.1f}, p99={stats['p99']:.1f}, max={stats['max']:,}, "
          f"min={stats['min']}")
    print(f"  Tiles with 0 intersections: {zero_tiles:,} / {n_tiles:,}")
    
    # ── Overflow analysis ──────────────────────────────────────────────
    overflow_results = {}
    for cap in CAPACITIES_TO_CHECK:
        overflow_mask = tile_counts > cap
        n_overflow = int(np.sum(overflow_mask))
        overflow_tiles = np.where(overflow_mask)[0].tolist()
        overflow_isects = int(np.sum(tile_counts[overflow_mask] - cap))
        total_overflow = int(np.sum(tile_counts[overflow_mask]))
        
        result = {
            "capacity": cap,
            "overflow_tile_count": n_overflow,
            "overflow_tile_ratio": float(n_overflow / n_tiles),
            "total_intersections_in_overflow_tiles": total_overflow,
            "dropped_intersections_if_truncated": overflow_isects,
            "overflow_tile_ids": overflow_tiles[:50] if len(overflow_tiles) > 50 else overflow_tiles,
            "overflow_tile_ids_truncated": len(overflow_tiles) > 50,
        }
        overflow_results[str(cap)] = result
        
        if n_overflow > 0:
            print(f"  Capacity {cap:,}: {n_overflow:,}/{n_tiles:,} tiles overflow "
                  f"({result['overflow_tile_ratio']*100:.2f}%), "
                  f"would drop {overflow_isects:,} isects")
        else:
            print(f"  Capacity {cap:,}: NO OVERFLOW")
    
    # ── Memory estimate ────────────────────────────────────────────────
    # Per-tile queue: each entry = {int32 gaussian_idx, float depth} = 8 bytes
    mem_per_tile_buf = {str(cap): cap * n_tiles * 8 for cap in CAPACITIES_TO_CHECK}
    # Current global buffers:
    # isect_ids: int64 = 8 bytes per isect
    # flatten_ids: int32 = 4 bytes per isect
    # sorted copies (double buffer): 8+4 = 12 per isect
    # CUB temp storage: typically ~2x the sort data
    global_isect_mem = n_isects * 8  # int64
    global_flatten_mem = n_isects * 4  # int32
    global_sorted_mem = n_isects * (8 + 4)  # double-buffer
    cub_temp_est = n_isects * (8 + 4) * 2  # rough CUB estimate
    total_global_mem = global_isect_mem + global_flatten_mem + global_sorted_mem
    
    memory = {
        "global_isect_ids_bytes": global_isect_mem,
        "global_flatten_ids_bytes": global_flatten_mem,
        "global_sorted_bytes": global_sorted_mem,
        "global_total_bytes": total_global_mem,
        "cub_temp_estimate_bytes": cub_temp_est,
        "per_tile_queue_bytes": {str(cap): mem_per_tile_buf[str(cap)] for cap in CAPACITIES_TO_CHECK},
    }
    
    result = {
        "scene": scene_name,
        "tile_size": tile_size,
        "image_width": W,
        "image_height": H,
        "n_gaussians": N,
        "forward_time_ms": fwd_ms,
        "stats": stats,
        "overflow": overflow_results,
        "memory": memory,
    }
    
    return result


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}
    
    for scene in SCENES:
        for tile_size in TILE_SIZES:
            key = f"{scene}_t{tile_size}"
            try:
                result = run_analysis(scene, tile_size)
                all_results[key] = result
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results[key] = {"error": str(e)}
    
    # Save
    output_path = OUTPUT_DIR / "c17_1_queue_capacity.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    # ── Summary table ───────────────────────────────────────────────────
    print("\n\n" + "="*90)
    print("SUMMARY — Per-Tile Intersection Statistics")
    print("="*90)
    header = f"{'Scene':>10} | {'tile':>4} | {'n_isects':>10} | {'n_tiles':>7} | {'mean':>7} | {'median':>6} | {'p95':>7} | {'p99':>7} | {'max':>7} | {'global_MB':>10} | {'mem_4096_MB':>11}"
    print(header)
    print("-"*90)
    for key, r in all_results.items():
        if "error" in r:
            print(f"{key:>10} | ERROR: {r['error']}")
            continue
        s = r["stats"]
        mem_mb = r["memory"]["global_total_bytes"] / (1024*1024)
        mem_queue_mb = r["memory"]["per_tile_queue_bytes"]["4096"] / (1024*1024)
        print(f"{r['scene']:>10} | {r['tile_size']:>4} | {s['n_isects']:>10,} | {s['n_tiles']:>7,} | {s['mean']:>7.1f} | {s['median']:>6.1f} | {s['p95']:>7.1f} | {s['p99']:>7.1f} | {s['max']:>7,} | {mem_mb:>8.1f}MB | {mem_queue_mb:>8.1f}MB")
    
    # ── Overflow summary ────────────────────────────────────────────────
    print("\n\nOVERFLOW ANALYSIS")
    print("-"*90)
    overflow_header = f"{'Scene':>10} | {'tile':>4} | " + " | ".join(f"{c:>5} overflow" for c in CAPACITIES_TO_CHECK)
    print(overflow_header)
    print("-"*90)
    for key, r in all_results.items():
        if "error" in r:
            continue
        parts = [f"{r['scene']:>10}", f"{r['tile_size']:>4}"]
        for cap in CAPACITIES_TO_CHECK:
            ov = r["overflow"][str(cap)]
            if ov["overflow_tile_count"] > 0:
                parts.append(f"{ov['overflow_tile_count']:>4}({ov['overflow_tile_ratio']*100:.1f}%)")
            else:
                parts.append(f"{'OK':>10}")
        print(" | ".join(parts))
