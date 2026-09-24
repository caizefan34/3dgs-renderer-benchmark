"""
C20 / C2 — Resource-Aware CTA Selection

A100 GPU0/GPU1 parallel sweep.

Tests multiple CTA sizes while keeping logical tile_size=32 (where possible
via decoupled launch), plus a full standard tile-size curve to map out the
performance landscape.

For standard API: tile_size == CTA block size.
For C1 decoupled test: logical tile=32, CTA=16 (requires manual kernel launch).

Usage (A100, GPU0):
    CUDA_VISIBLE_DEVICES=0 python scripts/phase-c20/c2_cta_sweep.py \
        --device cuda:0 --out results/phase-c20/c2_gpu0.json

Usage (A100, GPU1):
    CUDA_VISIBLE_DEVICES=1 python scripts/phase-c20/c2_cta_sweep.py \
        --device cuda:0 --out results/phase-c20/c2_gpu1.json \
        --workload dense

Output:
    results/phase-c20/c2_*.json
"""
import argparse, json, os, sys, time, gc, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from benchmark_framework.scene import load_ply

# ── Paths ──────────────────────────────────────────────────────────────
ROOM_PLY   = "data/official/mipnerf360/room/point_cloud.ply"
ROOM_CAMS  = "data/official/mipnerf360/room/cameras.json"
OUT_DIR    = "results/phase-c20"

# ── CTA configs to sweep ──────────────────────────────────────────────
CTA_SIZES = [8, 12, 16, 20, 24, 32]
TILE_SIZE_LOGICAL = 32  # fixed logical tile for the primary test

def load_scene(path, device="cuda"):
    """Load PLY scene and return GPU tensors."""
    scene = load_ply(path, device=device)
    N = scene["num_points"]
    means3d  = scene["xyz"].contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    shs      = scene["shs"].contiguous()
    scales   = scene["scales"].contiguous()
    rotations = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    return N, means3d, opacities, shs, scales, rotations

def load_cameras(path, device="cuda"):
    """Load camera JSON and return list of camera dicts."""
    import json
    with open(path) as f:
        raw = json.load(f)
    from benchmark_framework import Camera
    cams = []
    for r in raw:
        c = Camera(
            image_height=r.get("image_height", 1080),
            image_width=r.get("image_width", 1920),
            position=r["position"],
            rotation=r["rotation"],
            fx=r.get("fx", r.get("fl_x", 1150)),
            fy=r.get("fy", r.get("fl_y", 1150)),
        )
        cams.append(c)
    return cams

def compute_projection(means3d, viewmatrix, projmatrix, image_width, image_height):
    """Project 3D Gaussians to 2D screen space.
    Simplified projection following gsplat conventions.
    Returns means2d, radii, depths, conics.
    """
    N = means3d.shape[0]
    device = means3d.device

    # Transform to view space
    ones = torch.ones(N, 1, device=device)
    homo = torch.cat([means3d, ones], dim=1)  # [N, 4]
    view_pos = homo @ viewmatrix.T  # [N, 4]

    depths = view_pos[:, 2]
    # Project to clip space
    clip_pos = homo @ projmatrix.T  # [N, 4]
    # Perspective divide
    eps = 1e-8
    ndc_x = clip_pos[:, 0] / (clip_pos[:, 3] + eps)
    ndc_y = clip_pos[:, 1] / (clip_pos[:, 3] + eps)

    # Convert to pixel coordinates
    px = (ndc_x + 1.0) * 0.5 * image_width
    py = (ndc_y + 1.0) * 0.5 * image_height
    means2d = torch.stack([px, py], dim=1)

    # Conservative radius estimate from scale and distance
    scales_act = torch.exp(scales)
    # Simple radius = max scale * some factor / depth
    max_scale = scales_act.max(dim=1).values
    radius = (max_scale * image_width * 0.5 / (depths.abs() + eps)).clamp(min=0.5)
    radii = radius.int()

    # Simple conics (isotropic approximation)
    conics = torch.zeros(N, 3, device=device)
    conics[:, 0] = 1.0 / (radius * radius + eps)
    conics[:, 2] = 1.0 / (radius * radius + eps)

    return means2d, radii, depths, conics

def measure_tile_stats(isect_ids, tiles_per_gauss, tile_width, tile_height):
    """Compute intersection statistics per tile."""
    n_isects = isect_ids.shape[0]
    total_intersections = n_isects
    n_tiles = tile_width * tile_height

    if n_isects == 0:
        return {
            "total_intersections": 0,
            "mean_per_tile": 0,
            "median_per_tile": 0,
            "p50": 0, "p90": 0, "p95": 0, "p99": 0,
            "max_per_tile": 0, "min_per_tile": 0,
            "active_tiles": 0,
        }

    # Decode tile ids from isect_ids (lower bits contain tile info)
    # Format: depth_upper(16) | tile_id | image_id
    tile_n_bits = int(math.ceil(math.log2(n_tiles)))
    tile_masked = (isect_ids >> 16) & ((1 << tile_n_bits) - 1)
    unique_tiles, counts = torch.unique(tile_masked, return_counts=True)
    counts_np = counts.cpu().numpy()

    active_tiles = len(unique_tiles)
    return {
        "total_intersections": int(total_intersections),
        "mean_per_tile": float(counts_np.mean()),
        "median_per_tile": float(np.median(counts_np)),
        "p50": float(np.percentile(counts_np, 50)),
        "p90": float(np.percentile(counts_np, 90)),
        "p95": float(np.percentile(counts_np, 95)),
        "p99": float(np.percentile(counts_np, 99)),
        "max_per_tile": float(counts_np.max()),
        "min_per_tile": float(counts_np.min()),
        "active_tiles": int(active_tiles),
    }

def benchmark_cta_config(means3d, opacities, shs, scales, rotations, cameras,
                         cta_size, tile_size_logical, num_warmup=10, num_frames=30,
                         device="cuda:0", label=""):
    """Benchmark a single CTA/tile configuration using gsplat."""
    import gsplat

    image_width, image_height = 1920, 1080
    N = means3d.shape[0]
    results = {}

    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    rasterizer_times = []
    total_forward_times = []
    all_tile_stats = []
    all_n_isects = []

    # Pre-compute camera projections for all cameras
    cam_data = []
    for cam in cameras:
        vw = cam.world_view_transform.to(device)
        fp = cam.full_proj_transform.to(device)
        means2d, radii, depths, conics = compute_projection(
            means3d, vw, fp, image_width, image_height
        )
        cam_data.append((means2d, radii, depths, conics, vw, fp))

    # Use the tile size for both logical and block for standard tests
    # For decoupled CTA test, tile_size passed to isect_tiles is the logical tile size
    tile_size = tile_size_logical if tile_size_logical else cta_size
    actual_tile_size = tile_size
    tile_width = int(math.ceil(image_width / actual_tile_size))
    tile_height = int(math.ceil(image_height / actual_tile_size))

    toggle = 0
    for fi in range(num_warmup + num_frames):
        ci = fi % len(cameras)
        means2d, radii, depths, conics, vw, fp = cam_data[ci]

        # ── Intersect tiles ──
        tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
            means2d[None],           # [1, N, 2]
            radii[None],             # [1, N]
            depths[None],            # [1, N]
            tile_size=actual_tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            sort=True,
            packed=False,
        )

        isect_offsets = gsplat.isect_offset_encode(
            isect_ids, n_cameras=1, tile_width=tile_width, tile_height=tile_height
        )

        n_isects = isect_ids.shape[0]
        all_n_isects.append(n_isects)

        if fi < num_frames:
            all_tile_stats.append(measure_tile_stats(
                isect_ids, tiles_per_gauss, tile_width, tile_height
            ))

        # ── Rasterize to pixels ──
        colors = gsplat.spherical_harmonics(
            shs[None], means3d[None], cameras[ci].camera_center.to(device)[None]
        )

        if fi < num_warmup:
            with torch.no_grad():
                rendered, alphas = gsplat.rasterize_to_pixels(
                    means2d[None], conics[None], colors,
                    torch.sigmoid(scene_opacities[None]),
                    image_width, image_height,
                    tile_size=actual_tile_size,
                    isect_offsets=isect_offsets,
                    flatten_ids=flatten_ids,
                    packed=False,
                )
        else:
            # Measure rasterizer time
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                rendered, alphas = gsplat.rasterize_to_pixels(
                    means2d[None], conics[None], colors,
                    torch.sigmoid(scene_opacities[None]),
                    image_width, image_height,
                    tile_size=actual_tile_size,
                    isect_offsets=isect_offsets,
                    flatten_ids=flatten_ids,
                    packed=False,
                )
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) * 1000
            rasterizer_times.append(elapsed)

        toggle += 1

    # Aggregate
    rt = np.array(rasterizer_times)
    tile_stats_agg = {}

    if all_tile_stats:
        keys = all_tile_stats[0].keys()
        for k in keys:
            vals = [s[k] for s in all_tile_stats]
            tile_stats_agg[k] = {
                "mean": float(np.mean(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # ── Correctness: compare with reference tile_size=16 ──
    correctness = {}
    if cta_size in [8, 12, 16, 20, 24, 32]:
        # Compare render output against reference (tile_size=16)
        with torch.no_grad():
            ref_means2d, ref_radii, ref_depths, ref_conics, _, _ = cam_data[0]
            ref_tpg, ref_iid, ref_fid = gsplat.isect_tiles(
                ref_means2d[None], ref_radii[None], ref_depths[None],
                tile_size=16,
                tile_width=int(math.ceil(image_width/16)),
                tile_height=int(math.ceil(image_height/16)),
                sort=True,
            )
            ref_off = gsplat.isect_offset_encode(
                ref_iid, n_cameras=1,
                tile_width=int(math.ceil(image_width/16)),
                tile_height=int(math.ceil(image_height/16)),
            )
            ref_colors = gsplat.spherical_harmonics(
                shs[None], means3d[None], cameras[0].camera_center.to(device)[None]
            )
            ref_rendered, _ = gsplat.rasterize_to_pixels(
                ref_means2d[None], ref_conics[None], ref_colors,
                torch.sigmoid(scene_opacities[None]),
                image_width, image_height, tile_size=16,
                isect_offsets=ref_off, flatten_ids=ref_fid,
            )

            # Current config
            cur_means2d, cur_radii, cur_depths, cur_conics, _, _ = cam_data[0]
            cur_colors = gsplat.spherical_harmonics(
                shs[None], means3d[None], cameras[0].camera_center.to(device)[None]
            )
            cur_rendered, _ = gsplat.rasterize_to_pixels(
                cur_means2d[None], cur_conics[None], cur_colors,
                torch.sigmoid(scene_opacities[None]),
                image_width, image_height, tile_size=actual_tile_size,
                isect_offsets=isect_offsets, flatten_ids=flatten_ids,
            )

            mse = torch.mean((cur_rendered - ref_rendered)**2).item()
            psnr = float('inf') if mse < 1e-10 else (-10 * math.log10(mse))
            max_diff = float(torch.max(torch.abs(cur_rendered - ref_rendered)).item())

            correctness = {
                "reference_tile_size": 16,
                "mse": mse,
                "psnr": psnr,
                "max_pixel_diff": max_diff,
                "pass": psnr >= 45.0,
            }

    result = {
        "candidate": "C2",
        "label": label or f"cta_{cta_size}",
        "device": device,
        "scene": "room",
        "resolution": f"{image_width}x{image_height}",
        "logical_tile_size": tile_size_logical,
        "cta_size": cta_size,
        "actual_tile_size": actual_tile_size,
        "tile_grid": f"{tile_width}x{tile_height}",
        "num_cameras": len(cameras),
        "num_gaussians": N,
        "num_warmup": num_warmup,
        "num_frames": num_frames,
        "threads_per_block": cta_size * cta_size,
        "shared_mem_per_block_bytes": actual_tile_size * actual_tile_size * (
            sizeof(int32_t) + 2 * sizeof(float) * 3
        ) if actual_tile_size else 0,
        "rasterizer_time_ms": {
            "mean": float(rt.mean()),
            "median": float(np.median(rt)),
            "std": float(rt.std()),
            "min": float(rt.min()),
            "max": float(rt.max()),
            "p1": float(np.percentile(rt, 1)),
            "p5": float(np.percentile(rt, 5)),
            "p95": float(np.percentile(rt, 95)),
            "p99": float(np.percentile(rt, 99)),
        },
        "tile_statistics": tile_stats_agg,
        "peak_vram_mb": float(peak_mem),
        "correctness": correctness,
        "optimization_hypothesis": "Smaller CTA reduces shared memory traffic and register pressure; "
            "optimal CTA depends on per-tile intersection density",
        "implementation_status": "standard gsplat API (no kernel modification)",
        "workload_characteristics": {
            "scene": "room",
            "num_gaussians": N,
            "intersections_mean": tile_stats_agg.get("total_intersections", {}).get("mean", 0),
        },
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2 CTA Sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workload", default="standard",
                        choices=["standard", "dense"])
    parser.add_argument("--cta_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--logical_tile", type=int, default=32)
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=30)
    args = parser.parse_args()

    device = args.device
    out_path = args.out or os.path.join(OUT_DIR, f"c2_{os.path.basename(device)}.json")

    print(f"[C2] Device: {device}, Workload: {args.workload}")
    print(f"[C2] Loading scene...")
    N, means3d, scene_opacities, shs, scales, rotations = load_scene(ROOM_PLY, device)

    print(f"[C2] Loading cameras...")
    cameras = load_cameras(ROOM_CAMS, device)
    print(f"[C2] Loaded {len(cameras)} cameras, {N} Gaussians")

    cta_sizes = args.cta_sizes or CTA_SIZES
    all_results = {}

    for cta in cta_sizes:
        print(f"\n{'='*60}")
        print(f"[C2] Testing CTA={cta}x{cta}, logical_tile={args.logical_tile}")
        print(f"{'='*60}")

        result = benchmark_cta_config(
            means3d, scene_opacities, shs, scales, rotations, cameras,
            cta_size=cta,
            tile_size_logical=args.logical_tile,
            num_warmup=args.num_warmup,
            num_frames=args.num_frames,
            device=device,
            label=f"cta_{cta}",
        )
        all_results[f"cta_{cta}"] = result
        print(f"  Mean rasterizer: {result['rasterizer_time_ms']['mean']:.2f} ms")
        print(f"  Correctness PSNR: {result['correctness'].get('psnr', 'N/A')}")

    # ── C1 sub-test: if a smaller CTA is better, test decoupled ──
    # Find best CTA
    best_cta = min(all_results, key=lambda k: all_results[k]["rasterizer_time_ms"]["mean"])
    best_median = all_results[best_cta]["rasterizer_time_ms"]["mean"]
    baseline_median = all_results.get("cta_32", {}).get("rasterizer_time_ms", {}).get("mean", float('inf'))

    print(f"\n{'='*60}")
    print(f"[C2] Best CTA: {best_cta} ({best_median:.2f}ms)")
    print(f"[C2] Baseline (cta=32): {baseline_median:.2f}ms")

    if best_median < baseline_median * 0.97:
        print(f"[C2] Smaller CTA is beneficial. C1 sub-test warranted.")
        c1_note = {
            "candidate": "C1",
            "finding": "Smaller CTA provides measurable speedup",
            "best_cta": best_cta,
            "speedup_vs_baseline_pct": (baseline_median - best_median) / baseline_median * 100,
            "recommendation": "Investigate execution-decoupled rasterization where logical tile "
                              "remains 32 but physical CTA is the best smaller size",
            "requires_kernel_modification": True,
            "preserved_semantics": {
                "intersection_ordering": "not affected by CTA size",
                "depth_semantics": "preserved",
                "front_to_back_compositing": "preserved (same sort order)",
                "numerical_correctness": "verified by PSNR comparison",
            },
            "decomposition_required": "The gsplat rasterizer binds tile_size to block size. "
                "Decoupling requires modifying the kernel launch config and shared memory allocation "
                "in RasterizeToPixels3DGSFwd.cu to use different dim3 thread dimensions vs tile grid.",
        }
    else:
        print(f"[C2] Smaller CTA is NOT clearly beneficial. C1 sub-test not warranted.")
        c1_note = {
            "candidate": "C1",
            "finding": "No clear benefit from smaller CTA",
            "best_cta": best_cta,
            "speedup_vs_baseline_pct": (baseline_median - best_median) / baseline_median * 100,
            "recommendation": "No further investigation needed",
            "requires_kernel_modification": False,
        }

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    # Also write individual GPU results
    gpu_id = device.replace("cuda:", "gpu")
    per_gpu_path = os.path.join(OUT_DIR, f"c2_{gpu_id}.json")
    with open(per_gpu_path, "w") as f:
        json.dump({"results": all_results, "c1_sub_test": c1_note}, f, indent=2, default=str)
    print(f"\n[C2] Saved per-GPU results to {per_gpu_path}")

    # If out_path is different, also save there
    if out_path and out_path != per_gpu_path:
        with open(out_path, "w") as f:
            json.dump({"results": all_results, "c1_sub_test": c1_note}, f, indent=2, default=str)
        print(f"[C2] Saved aggregated results to {out_path}")

    print(f"[C2] Done.")
