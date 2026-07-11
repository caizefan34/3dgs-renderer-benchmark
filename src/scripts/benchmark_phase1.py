"""
Phase 1 Benchmark: all renderers (including TC-GS) on identical scene and cameras.
Tests: speedy_splat, diff_gaussian, tc_gs, gsplat
"""
import sys, torch, time, gc, json, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "benchmark_framework"))
from benchmark_framework import load_ply, load_cameras_from_json

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "scene.ply")
CAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cameras.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")

scene = load_ply(SCENE, device="cuda")
cameras = load_cameras_from_json(CAMS, device="cuda")
N_CAMS = len(cameras)
print(f"Loaded {N_CAMS} cameras, {scene['num_points']} gaussians")

means3d = scene["xyz"].contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
scales = scene["scales"].contiguous()
rotations = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()

def create_renderer(rname, cam):
    if rname == "speedy_splat":
        from speedy_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        s = GaussianRasterizationSettings(
            image_height=1080, image_width=1920,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=3, campos=cam.camera_center,
            prefiltered=False, debug=False,
        )
        return GaussianRasterizer(s), True
    else:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        s = GaussianRasterizationSettings(
            image_height=1080, image_width=1920,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=3, campos=cam.camera_center,
            prefiltered=False, debug=False, antialiasing=False,
        )
        return GaussianRasterizer(s), False

def render_frame(rname, rast, needs_scores):
    m2 = torch.zeros_like(means3d[:, :2])
    if needs_scores:
        scores = torch.ones(means3d.shape[0], device="cuda")
        out, _, _ = rast(means3D=means3d, means2D=m2, opacities=opacities,
            scores=scores, shs=shs, colors_precomp=None,
            scales=scales, rotations=rotations, cov3D_precomp=None)
    else:
        out, _, _ = rast(means3D=means3d, means2D=m2, opacities=opacities,
            shs=shs, colors_precomp=None,
            scales=scales, rotations=rotations, cov3D_precomp=None)
    return out

all_results = {}

for rname in ["speedy_splat", "diff_gaussian", "tc_gs", "gsplat"]:
    print()
    print("=" * 60)
    print(f"  BENCHMARK: {rname}")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Constructor timing
    print("  [Phase 1] Measuring rasterizer construction time ({N_CAMS} cameras)...", end=" ", flush=True)
    construct_times = []
    for ci in range(min(N_CAMS, 50)):
        cam = cameras[ci]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rast, needs_scores = create_renderer(rname, cam)
        torch.cuda.synchronize()
        construct_times.append((time.perf_counter() - t0) * 1000)
        del rast
        gc.collect()
    ct = np.array(construct_times)
    print(f"mean={ct.mean():.2f}ms, median={np.median(ct):.2f}ms")
    
    # Render timing
    print("  [Phase 2] Measuring pure render time (200 frames)...", end=" ", flush=True)
    render_times = []
    peak_mem = 0
    
    # Warmup
    cam0 = cameras[0]
    rast, needs_scores = create_renderer(rname, cam0)
    for _ in range(50):
        torch.cuda.synchronize()
        with torch.no_grad():
            render_frame(rname, rast, needs_scores)
        torch.cuda.synchronize()
    del rast
    gc.collect()
    torch.cuda.empty_cache()
    
    # Benchmark frames
    for fi in range(200):
        cam = cameras[fi % N_CAMS]
        rast, needs_scores = create_renderer(rname, cam)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            render_frame(rname, rast, needs_scores)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        render_times.append(elapsed)
        mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if mem > peak_mem:
            peak_mem = mem
        del rast
        gc.collect()
        if (fi + 1) % 50 == 0:
            print(f"{fi+1}..", end="", flush=True)
    print(" done")
    
    t = np.array(render_times)
    t_sorted = np.sort(t)
    trim = max(1, len(t) // 10)
    t_stable = t_sorted[trim:-trim] if len(t) > 2 * trim else t[5:]
    
    log = {
        "renderer": rname,
        "num_frames": 200,
        "warmup": 50,
        "num_gaussians": scene["num_points"],
        "peak_memory_mb": round(float(peak_mem), 1),
        "construct_mean_ms": round(float(ct.mean()), 3),
        "construct_median_ms": round(float(np.median(ct)), 3),
        "mean_ms": round(float(t.mean()), 3),
        "median_ms": round(float(np.median(t)), 3),
        "min_ms": round(float(t.min()), 3),
        "max_ms": round(float(t.max()), 3),
        "std_ms": round(float(t.std()), 3),
        "p10_ms": round(float(np.percentile(t, 10)), 3),
        "p25_ms": round(float(np.percentile(t, 25)), 3),
        "p75_ms": round(float(np.percentile(t, 75)), 3),
        "p90_ms": round(float(np.percentile(t, 90)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "stable_mean_ms": round(float(t_stable.mean()), 3),
        "stable_median_ms": round(float(np.median(t_stable)), 3),
        "mean_fps": round(1000.0 / float(t.mean()), 1),
        "median_fps": round(1000.0 / float(np.median(t)), 1),
        "stable_mean_fps": round(1000.0 / float(t_stable.mean()), 1),
        "stable_median_fps": round(1000.0 / float(np.median(t_stable)), 1),
    }
    all_results[rname] = log
    
    print(f'    Median:      {log["median_ms"]:7.2f} ms = {log["median_fps"]:7.1f} FPS')
    print(f'    Mean:        {log["mean_ms"]:7.2f} ms = {log["mean_fps"]:7.1f} FPS')
    print(f'    P99:         {log["p99_ms"]:7.2f} ms')
    print(f'    Peak Mem:    {log["peak_memory_mb"]:7.1f} MB')

# Final ranking
print()
print("=" * 70)
print("  FINAL RANKING (sorted by median render-only latency)")
print("=" * 70)

ranked = sorted(all_results.values(), key=lambda l: l["median_ms"])
fastest = ranked[0]

for i, log in enumerate(ranked):
    rname = log["renderer"]
    med = log["median_ms"]
    fps = log["median_fps"]
    p99 = log["p99_ms"]
    mem = log["peak_memory_mb"]
    if i == 0:
        tag = "  <<< FASTEST >>>"
    else:
        pct = ((med / ranked[0]["median_ms"]) - 1) * 100
        tag = f"  ({pct:+.1f}% vs fastest)"
    print(f"  #{i+1}: {rname:20s}  median={med:6.2f}ms = {fps:6.1f}FPS  P99={p99:6.2f}ms  Mem={mem:.0f}MB{tag}")

output = {
    "metadata": {
        "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU (8.55 GB)",
        "cuda": "13.0 (driver 13.1, toolkit 13.3)",
        "pytorch": "2.12.1+cu130",
        "num_gaussians": scene["num_points"],
        "resolution": "1920x1080",
        "scene": "Synthetic 400K GS, SH degree 3",
        "cameras": f"{N_CAMS} fixed orbit views",
        "date": "2026-07-11",
        "fastest_renderer": fastest["renderer"],
        "baseline_renderer": "diff_gaussian",
    },
    "results": all_results,
}

with open(os.path.join(OUT_DIR, "benchmark_results_phase1.json"), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {os.path.join(OUT_DIR, 'benchmark_results_phase1.json')}")
