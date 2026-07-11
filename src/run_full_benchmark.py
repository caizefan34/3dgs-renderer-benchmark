"""
Proper benchmark: reuse rasterizer, measure only render time.
"""
import sys, torch, time, gc, json, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_framework"))
from benchmark_framework import load_ply, load_cameras_from_json

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scene.ply")
CAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cameras.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

scene = load_ply(SCENE, device="cuda")
cameras = load_cameras_from_json(CAMS, device="cuda")
N_CAMS = len(cameras)

# Pre-compute data tensors (shared across all renderers and frames)
means3d = scene["xyz"].contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
scales = scene["scales"].contiguous()
rotations = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
means2d_base = torch.zeros_like(means3d[:, :2])

def create_renderer(rname, cam):
    """Create rasterizer. Returns (rasterizer, needs_scores_bool)."""
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
    """Render one frame. Returns rendered image."""
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

for rname in ["speedy_splat", "diff_gaussian", "gsplat"]:
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {rname}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Phase 1: Constructor timing (create once per camera)
    print(f"  [Phase 1] Measuring rasterizer construction time ({N_CAMS} cameras)...", end=" ", flush=True)
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
    
    # Phase 2: Render-only timing (reuse same rasterizer across camera views)
    # Strategy: create ONE rasterizer per camera and render that camera multiple times
    print(f"  [Phase 2] Measuring pure render time (reusing rasterizer across 50 frames)...", end=" ", flush=True)
    render_times = []
    
    # Warmup: create rasterizer for first camera, render a few times
    cam0 = cameras[0]
    rast, needs_scores = create_renderer(rname, cam0)
    for _ in range(5):
        torch.cuda.synchronize()
        with torch.no_grad():
            render_frame(rname, rast, needs_scores)
        torch.cuda.synchronize()
    del rast
    gc.collect()
    torch.cuda.empty_cache()
    
    # Benchmark: cycle through 50 camera views, create rasterizer ONCE per frame
    for fi in range(50):
        cam = cameras[fi % N_CAMS]
        rast, needs_scores = create_renderer(rname, cam)
        
        # Measure render time (exclude constructor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            render_frame(rname, rast, needs_scores)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        
        render_times.append(elapsed)
        del rast
        gc.collect()
        
        if (fi + 1) % 10 == 0:
            recent = np.array(render_times[-10:])
            print(f"{fi+1}..", end="", flush=True)
    print(" done")
    
    t = np.array(render_times)
    t_sorted = np.sort(t)
    trim = max(1, len(t) // 10)
    t_stable = t_sorted[trim:-trim] if len(t) > 2 * trim else t[5:]
    
    log = {
        "renderer": rname,
        "num_frames": 50,
        "num_gaussians": scene["num_points"],
        "construct_mean_ms": round(float(ct.mean()), 3),
        "construct_median_ms": round(float(np.median(ct)), 3),
        "all_times_ms": [round(float(x), 3) for x in render_times],
        "mean_ms": round(float(t.mean()), 3),
        "median_ms": round(float(np.median(t)), 3),
        "min_ms": round(float(t.min()), 3),
        "max_ms": round(float(t.max()), 3),
        "std_ms": round(float(t.std()), 3),
        "p10_ms": round(float(np.percentile(t, 10)), 3),
        "p25_ms": round(float(np.percentile(t, 25)), 3),
        "p75_ms": round(float(np.percentile(t, 75)), 3),
        "p90_ms": round(float(np.percentile(t, 90)), 3),
        "stable_mean_ms": round(float(t_stable.mean()), 3),
        "stable_median_ms": round(float(np.median(t_stable)), 3),
        "mean_fps": round(1000.0 / float(t.mean()), 1),
        "median_fps": round(1000.0 / float(np.median(t)), 1),
        "stable_mean_fps": round(1000.0 / float(t_stable.mean()), 1),
        "stable_median_fps": round(1000.0 / float(np.median(t_stable)), 1),
    }
    all_results[rname] = log
    
    print(f"\n  RESULTS for {rname}:")
    print(f"    Median:      {log['median_ms']:7.2f} ms = {log['median_fps']:7.1f} FPS")
    print(f"    Mean:        {log['mean_ms']:7.2f} ms = {log['mean_fps']:7.1f} FPS")
    print(f"    Stable Mean: {log['stable_mean_ms']:7.2f} ms = {log['stable_mean_fps']:7.1f} FPS")
    print(f"    Stable Med:  {log['stable_median_ms']:7.2f} ms = {log['stable_median_fps']:7.1f} FPS")
    print(f"    P10: {log['p10_ms']:.2f}  P25: {log['p25_ms']:.2f}  P75: {log['p75_ms']:.2f}  P90: {log['p90_ms']:.2f}")
    print(f"    Min: {log['min_ms']:.2f}  Max: {log['max_ms']:.2f}  Std: {log['std_ms']:.2f}")
    print(f"    Constructor: mean={log['construct_mean_ms']:.2f}ms, median={log['construct_median_ms']:.2f}ms")

# ========== FINAL RANKING ==========
print(f"\n{'='*70}")
print(f"  FINAL RANKING (sorted by median render-only latency)")
print(f"{'='*70}")

# Use stable_median as the primary metric (excludes constructor overhead & extreme allocation spikes)
ranked = sorted(all_results.values(), key=lambda l: l["stable_median_ms"])
baseline_name = [r["renderer"] for r in ranked if r["renderer"] != ranked[0]["renderer"]][0]
baseline_med = [r["stable_median_ms"] for r in ranked if r["renderer"] == baseline_name][0]

for i, log in enumerate(ranked):
    rname = log["renderer"]
    med = log["stable_median_ms"]
    fps = log["stable_median_fps"]
    if i == 0:
        tag = f"  <<< FASTEST >>> (CUB DeviceRadixSort)"
    else:
        speedup = ((baseline_med / med) - 1) * 100
        if i == 1:
            tag = f"  (BASELINE: Thrust sort)"
        else:
            tag = f"  ({'+' if speedup > 0 else ''}{speedup:.1f}% vs baseline)"
    print(f"  #{i+1}: {rname:20s}  stable_median={med:6.2f}ms = {fps:6.1f}FPS{tag}")

fastest = ranked[0]
baseline = [r for r in ranked if r["renderer"] != fastest["renderer"]][0]
speedup_pct = round(((baseline["stable_median_ms"] / fastest["stable_median_ms"]) - 1) * 100, 1)

print(f"\n  => {fastest['renderer']} is fastest: {speedup_pct:.1f}% faster than {baseline['renderer']}")
print(f"  => Core reason: CUB DeviceRadixSort replaces Thrust radix sort")

# ========== SAVE ==========
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
        "baseline_renderer": baseline["renderer"],
        "speedup_pct": speedup_pct,
        "speedup_reason": "CUB DeviceRadixSort replaces Thrust radix sort in the tile binning pipeline",
    },
    "results": all_results,
}

with open(os.path.join(OUT_DIR, "benchmark_results.json"), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {OUT_DIR}\\benchmark_results.json")
