"""
Phase 17B — C17-1: Per-Tile Intersection Capacity Analysis
==========================================================
Uses the REAL first camera from each MipNeRF 360 scene.
Scene PLY = SfM point clouds.

For each scene (room, bicycle, garden) × tile_size (16, 20, 32):
  1. Load scene PLY + first training camera via benchmark_framework
  2. Run baseline gsplat rasterization
  3. Decode per-tile intersection counts from isect_ids
  4. Report: mean, median, p95, p99, max, overflow behavior
"""

import json, math, os, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework.scene import load_ply
from benchmark_framework.cameras import load_cameras_from_json, resize_cameras

DEVICE = "cuda"
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase17b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZES = [16, 20, 32]
CAPACITIES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
RESOLUTIONS = {"room": (1080, 1080), "bicycle": (1920, 1080), "garden": (1920, 1080)}


def get_first_camera(scene_name: str, target_w: int, target_h: int):
    """Load first real camera from the MipNeRF 360 cameras.json, resized."""
    cam_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "cameras.json"
    cameras = load_cameras_from_json(str(cam_path), device=DEVICE)
    cameras = resize_cameras(cameras, target_w, target_h)
    return cameras[0]


def decode_tile_ids(isect_ids: torch.Tensor, n_tiles: int) -> np.ndarray:
    """Decode tile_ids from isect_ids, auto-detecting key format."""
    tile_n_bits = max(int(math.ceil(math.log2(n_tiles))), 1)
    mask = (1 << tile_n_bits) - 1
    
    t_c1 = ((isect_ids >> 16) & mask).cpu().numpy().astype(np.int64)
    t_bl = ((isect_ids >> 32) & mask).cpu().numpy().astype(np.int64)
    
    max_c1 = t_c1.max() if len(t_c1) > 0 else 0
    max_bl = t_bl.max() if len(t_bl) > 0 else 0
    
    if max_c1 < n_tiles and max_c1 > 0:
        return t_c1
    elif max_bl < n_tiles and max_bl > 0:
        return t_bl
    else:
        print(f"  WARNING: C1 max={max_c1}, BL max={max_bl}, n_tiles={n_tiles}")
        print(f"  First 10 hex: {[hex(x.item()) for x in isect_ids[:10]]}")
        return t_c1


@torch.no_grad()
def run_analysis(scene_name: str, tile_size: int):
    print(f"\n{'='*60}")
    print(f"Scene: {scene_name}, tile_size={tile_size}")
    print(f"{'='*60}")
    
    torch.cuda.reset_peak_memory_stats()
    
    # Load scene
    ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"
    scene = load_ply(str(ply_path), device=DEVICE)
    sh_degree = scene.get("sh_degree", 3)
    
    # Camera
    target_w, target_h = RESOLUTIONS[scene_name]
    cam = get_first_camera(scene_name, target_w, target_h)
    W, H = cam.image_width, cam.image_height
    
    # Build viewmat for gsplat (c2w is camera_center, use viewmatrix)
    viewmat = cam.viewmatrix.unsqueeze(0).to(DEVICE)
    K = cam.K.unsqueeze(0).to(DEVICE)
    
    print(f"  Resolution: {W}x{H}, camera: {cam.image_name}")
    
    # Prepare
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    
    from gsplat import rasterization
    
    # Warmup
    _, _, meta = rasterization(
        means=scene["xyz"], quats=quats, scales=scales,
        opacities=opacities, colors=scene["shs"],
        viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=tile_size, packed=True,
        sh_degree=sh_degree, render_mode="RGB")
    torch.cuda.synchronize()
    
    # Timed forward
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    renders, alphas, meta = rasterization(
        means=scene["xyz"], quats=quats, scales=scales,
        opacities=opacities, colors=scene["shs"],
        viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=tile_size, packed=True,
        sh_degree=sh_degree, render_mode="RGB")
    end.record()
    end.synchronize()
    fwd_ms = start.elapsed_time(end)
    
    # Extract
    isect_ids = meta["isect_ids"]
    n_isects = isect_ids.shape[0]
    tile_width = meta["tile_width"]
    tile_height = meta["tile_height"]
    n_tiles = tile_width * tile_height
    nnz = meta.get("gaussian_ids", torch.tensor([0])).shape[0]
    
    print(f"  nnz={nnz:,}, n_isects={n_isects:,}, tiles={tile_width}x{tile_height}={n_tiles}")
    print(f"  Forward: {fwd_ms:.2f}ms")
    
    # Decode tile IDs
    tile_ids = decode_tile_ids(isect_ids, n_tiles)
    unique_tiles = len(np.unique(tile_ids))
    print(f"  Unique tiles covered: {unique_tiles:,} / {n_tiles:,}")
    
    # Per-tile counts
    tile_counts = np.bincount(tile_ids, minlength=n_tiles).astype(np.int64)
    non_zero = tile_counts[tile_counts > 0]
    
    stats = {
        "n_isects": int(n_isects),
        "n_tiles": n_tiles,
        "tiles_with_isects": int(np.sum(tile_counts > 0)),
        "tiles_without_isects": int(np.sum(tile_counts == 0)),
        "mean_per_tile": float(np.mean(non_zero)) if len(non_zero) > 0 else 0,
        "median_per_tile": float(np.median(non_zero)) if len(non_zero) > 0 else 0,
        "p95": float(np.percentile(non_zero, 95)) if len(non_zero) > 0 else 0,
        "p99": float(np.percentile(non_zero, 99)) if len(non_zero) > 0 else 0,
        "max": int(np.max(tile_counts)),
        "std": float(np.std(non_zero)) if len(non_zero) > 0 else 0,
        "p99_9": float(np.percentile(non_zero, 99.9)) if len(non_zero) > 0 else 0,
    }
    
    print(f"  Active tiles: mean={stats['mean_per_tile']:.1f}, median={stats['median_per_tile']:.0f}, "
          f"p95={stats['p95']:.0f}, p99={stats['p99']:.0f}, p99.9={stats['p99_9']:.0f}, max={stats['max']:,}")
    
    # Distribution histogram
    hist_bins = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000]
    hist = {}
    for i in range(len(hist_bins)-1):
        lo, hi = hist_bins[i], hist_bins[i+1]
        cnt = int(np.sum((tile_counts >= lo) & (tile_counts < hi)))
        hist[f"[{lo},{hi})"] = cnt
    cnt_ge = int(np.sum(tile_counts >= hist_bins[-1]))
    hist[f"[{hist_bins[-1]},inf)"] = cnt_ge
    print(f"  Distribution: {hist}")
    
    # Overflow analysis
    overflow = {}
    for cap in CAPACITIES:
        ov_mask = tile_counts > cap
        n_ov = int(np.sum(ov_mask))
        overflow[str(cap)] = {
            "capacity": cap,
            "overflow_tile_count": n_ov,
            "overflow_tile_ratio": float(n_ov / n_tiles) if n_tiles > 0 else 0,
            "overflow_active_tile_ratio": float(n_ov / max(1, int(np.sum(tile_counts > 0)))),
            "total_isects_in_overflow_tiles": int(np.sum(tile_counts[ov_mask])),
            "dropped_if_truncated": int(np.sum(tile_counts[ov_mask] - cap)),
        }
        if n_ov > 0:
            print(f"  Cap {cap:>5}: {n_ov:>4} tiles overflow "
                  f"({n_ov/n_tiles*100:.2f}% all, {n_ov/max(1,int(np.sum(tile_counts>0)))*100:.1f}% active), "
                  f"would drop {overflow[str(cap)]['dropped_if_truncated']:,}")
    
    # Memory comparison
    global_mem = n_isects * (8 + 4)  # isect_ids(int64) + flatten_ids(int32)
    sorted_mem = n_isects * (8 + 4)  # sorted copies (double buffer)
    cub_est = n_isects * (8 + 4) * 2  # rough CUB temp estimate
    
    per_tile_queue = {}
    for cap in CAPACITIES:
        per_tile_queue[str(cap)] = n_tiles * cap * 8  # 8 bytes per entry (int32 + float32)
    
    result = {
        "scene": scene_name,
        "tile_size": tile_size,
        "image_width": W,
        "image_height": H,
        "n_gaussians": scene["num_points"],
        "nnz": int(nnz),
        "forward_time_ms": fwd_ms,
        "stats": stats,
        "histogram": hist,
        "overflow": overflow,
        "memory_estimate_bytes": {
            "global_isect_ids": n_isects * 8,
            "global_flatten_ids": n_isects * 4,
            "global_sorted_copies": sorted_mem,
            "cub_temp_storage_estimate": cub_est,
            "global_total_current": global_mem + sorted_mem + cub_est,
            "per_tile_queue": per_tile_queue,
        },
    }
    return result


if __name__ == "__main__":
    results = {}
    for scene in ["room", "bicycle", "garden"]:
        for ts in TILE_SIZES:
            key = f"{scene}_t{ts}"
            try:
                r = run_analysis(scene, ts)
                results[key] = r
            except Exception as e:
                print(f"  ERROR {key}: {e}")
                import traceback; traceback.print_exc()
                results[key] = {"error": str(e)}
    
    out_path = OUTPUT_DIR / "c17_1_queue_capacity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Summary
    print("\n" + "="*110)
    print("PER-TILE INTERSECTION CAPACITY SUMMARY")
    print("="*110)
    h = f"{'Scene':>10} | {'tile':>4} | {'n_isects':>10} | {'tiles_on':>6} | {'mean':>7} | {'med':>5} | {'p95':>7} | {'p99':>7} | {'max':>8} | {'global':>7} | {'q4096':>7}"
    print(h)
    print("-"*110)
    for k, r in results.items():
        if "error" in r:
            print(f"{k:>15}: {r['error']}")
            continue
        s = r["stats"]
        gm = r["memory_estimate_bytes"]["global_total_current"] / (1024*1024)
        qm = r["memory_estimate_bytes"]["per_tile_queue"]["4096"] / (1024*1024)
        print(f"{r['scene']:>10} | {r['tile_size']:>4} | {s['n_isects']:>10,} | {s['tiles_with_isects']:>6,} | {s['mean_per_tile']:>7.1f} | {s['median_per_tile']:>5.0f} | {s['p95']:>7.0f} | {s['p99']:>7.0f} | {s['max']:>8,} | {gm:>6.1f}M | {qm:>6.1f}M")
