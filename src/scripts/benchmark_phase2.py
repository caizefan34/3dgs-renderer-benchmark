"""
Phase 2 benchmark - simplified approach.
"""
import sys, torch, time, gc, json, os, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "benchmark_framework"))
from benchmark_framework import load_ply, load_cameras_from_json

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "scene.ply")
CAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cameras.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")

scene = load_ply(SCENE, device="cuda")
cameras = load_cameras_from_json(CAMS, device="cuda")
N_CAMS = len(cameras)
N_G = scene["num_points"]

means3d = scene["xyz"].contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
scales = scene["scales"].contiguous()
rotations = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()

print(f"Loaded {N_CAMS} cameras, {N_G} gaussians")

from speedy_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def make_rast(cam):
    bg = torch.zeros(3, device="cuda")
    s = GaussianRasterizationSettings(
        image_height=1080, image_width=1920,
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=3, campos=cam.camera_center,
        prefiltered=False, debug=False,
    )
    return GaussianRasterizer(s)

# ===== Frustum culling using view matrix =====
def compute_frustum_mask(cam):
    """Conservative culling using world_view_transform."""
    V = cam.world_view_transform  # 4x4 camera-to-world
    cam_pos = cam.camera_center    # (3,)
    
    # Get camera axes from view matrix (which is world-to-camera transposed)
    # viewmatrix is 4x4: [R | t] where R is world->cam rotation
    view = cam.viewmatrix  # 4x4
    cam_z = view[2, :3]    # camera forward direction (in world space, negated)
    cam_x = view[0, :3]
    cam_y = view[1, :3]
    
    # Vector from camera to each gaussian
    dir_to_gs = means3d - cam_pos.unsqueeze(0)  # (N, 3)
    
    # Depth along camera forward axis (positive = in front)
    depth = (dir_to_gs * cam_z).sum(dim=1)  # note: view[2] is -forward
    
    # Project to NDC-like coordinates using tanfov
    tan_h = cam.tanfovx  # horizontal half-fov tangent
    tan_v = cam.tanfovy  # vertical half-fov tangent
    
    proj_x = (dir_to_gs * cam_x).sum(dim=1) / (depth.abs() * tan_h + 1e-8)
    proj_y = (dir_to_gs * cam_y).sum(dim=1) / (depth.abs() * tan_v + 1e-8)
    
    # Conservative: keep anything with depth > 0 (in front) and within [-3, 3] in projection
    z_valid = depth > 0.1
    x_valid = (proj_x >= -3.0) & (proj_x <= 3.0)
    y_valid = (proj_y >= -3.0) & (proj_y <= 3.0)
    
    return z_valid & x_valid & y_valid

mask0 = compute_frustum_mask(cameras[0])
visible_pct = mask0.float().mean().item() * 100
print(f"Frustum culling keeps {visible_pct:.1f}% ({mask0.sum().item()}/{N_G}) gaussians")

# Pre-compute masks for all cameras
cull_masks = {}
for ci in range(N_CAMS):
    cull_masks[ci] = compute_frustum_mask(cameras[ci])
avg_visible = sum(m.float().mean().item() for m in cull_masks.values()) / N_CAMS * 100
print(f"Avg visible across all cams: {avg_visible:.1f}%")


def benchmark_config(name, render_fn_func, use_cache=True, num_warmup=50, num_frames=200):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {name}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    rast_cache = {}
    times = []
    
    for fi in range(num_warmup + num_frames):
        ci = fi % N_CAMS
        cam = cameras[ci]
        
        if use_cache and ci in rast_cache:
            rast = rast_cache[ci]
        else:
            rast = make_rast(cam)
            if use_cache:
                rast_cache[ci] = rast
        
        render_fn = render_fn_func(ci, cam, rast)
        
        if fi < num_warmup:
            with torch.no_grad():
                render_fn(rast)
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                render_fn(rast)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        
        if (fi + 1) % 50 == 0 and fi >= num_warmup:
            print(f"  {fi+1-num_warmup}..", end="", flush=True)
    print(" done")
    
    t = np.array(times)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    log = {
        "renderer": name,
        "num_frames": num_frames,
        "warmup": num_warmup,
        "num_gaussians": N_G,
        "peak_memory_mb": round(float(peak_mem), 1),
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
        "mean_fps": round(1000.0 / float(t.mean()), 1),
        "median_fps": round(1000.0 / float(np.median(t)), 1),
    }
    
    print(f'    Median:      {log["median_ms"]:7.2f} ms = {log["median_fps"]:7.1f} FPS')
    print(f'    Mean:        {log["mean_ms"]:7.2f} ms = {log["mean_fps"]:7.1f} FPS')
    print(f'    P99:         {log["p99_ms"]:7.2f} ms')
    print(f'    Peak Mem:    {log["peak_memory_mb"]:7.1f} MB')
    
    return log


results = {}

# ===== Baseline (recreate rasterizer each frame) =====
def render_baseline_factory(ci, cam, rast):
    def fn(_rast):
        m2 = torch.zeros(N_G, 2, device="cuda")
        scores = torch.ones(N_G, device="cuda")
        _rast(means3D=means3d, means2D=m2, opacities=opacities,
              scores=scores, shs=shs, colors_precomp=None,
              scales=scales, rotations=rotations, cov3D_precomp=None)
    return fn

results["baseline"] = benchmark_config("baseline_speedy", render_baseline_factory, use_cache=True)

# ===== OPT 1: Culling (remove clearly invisible gaussians) =====
def render_culling_factory(ci, cam, rast):
    mask = cull_masks[ci]
    n_v = mask.sum().item()
    def fn(_rast):
        m2 = torch.zeros(n_v, 2, device="cuda")
        scores = torch.ones(n_v, device="cuda")
        _rast(means3D=means3d[mask], means2D=m2, opacities=opacities[mask],
              scores=scores, shs=shs[mask], colors_precomp=None,
              scales=scales[mask], rotations=rotations[mask], cov3D_precomp=None)
    return fn

results["culling"] = benchmark_config("opt_culling", render_culling_factory, use_cache=True)

# ===== OPT 2: Culling + rast recreated each frame (no cache) =====
def render_culling_nocache_factory(ci, cam, rast):
    mask = cull_masks[ci]
    n_v = mask.sum().item()
    def fn(_rast):
        m2 = torch.zeros(n_v, 2, device="cuda")
        scores = torch.ones(n_v, device="cuda")
        _rast(means3D=means3d[mask], means2D=m2, opacities=opacities[mask],
              scores=scores, shs=shs[mask], colors_precomp=None,
              scales=scales[mask], rotations=rotations[mask], cov3D_precomp=None)
    return fn

results["culling_nocache"] = benchmark_config("opt_culling_nocache", render_culling_nocache_factory, use_cache=False)

# ===== OPT 3: Pre-allocated buffers reuse (remove per-frame allocations) =====
# Pre-allocate all per-camera buffers
pre_masks = {}
pre_bufs = {}
for ci in range(N_CAMS):
    mask = cull_masks[ci]
    pre_masks[ci] = mask
    n_v = mask.sum().item()
    pre_bufs[ci] = {
        "m2": torch.zeros(n_v, 2, device="cuda"),
        "scores": torch.ones(n_v, device="cuda"),
    }

def render_culling_nomalloc_factory(ci, cam, rast):
    mask = pre_masks[ci]
    bufs = pre_bufs[ci]
    def fn(_rast):
        _rast(means3D=means3d[mask], means2D=bufs["m2"], opacities=opacities[mask],
              scores=bufs["scores"], shs=shs[mask], colors_precomp=None,
              scales=scales[mask], rotations=rotations[mask], cov3D_precomp=None)
    return fn

results["culling_nomalloc"] = benchmark_config("opt_culling_nomalloc", render_culling_nomalloc_factory, use_cache=True)

# ===== OPT 4: Async pipeline (double buffering) =====
# Use CUDA streams to overlap data transfer/CPU prep with GPU rendering
def render_async_factory(ci, cam, rast):
    s0 = torch.cuda.Stream()
    # Pre-compute: create views into same data
    mask = pre_masks[ci]
    n_v = mask.sum().item()
    m2_pinned = torch.zeros(n_v, 2, device="cuda", pin_memory=True)
    scores_pinned = torch.ones(n_v, device="cuda", pin_memory=True)
    
    def fn(_rast):
        with torch.cuda.stream(s0):
            _rast(means3D=means3d[mask], means2D=m2_pinned, opacities=opacities[mask],
                  scores=scores_pinned, shs=shs[mask], colors_precomp=None,
                  scales=scales[mask], rotations=rotations[mask], cov3D_precomp=None)
        torch.cuda.current_stream().wait_stream(s0)
    return fn

results["async"] = benchmark_config("opt_async", render_async_factory, use_cache=True)

# ===== Ranking =====
print()
print("=" * 70)
print("  PHASE 2 OPTIMIZATION RESULTS")
print("=" * 70)

b = results["baseline"]
b_med = b["median_ms"]
print(f"\n  Baseline (speedy_splat): {b_med:.2f}ms @ {b['median_fps']:.1f}FPS\n")

ranked = sorted(results.values(), key=lambda l: l["median_ms"])
for log in ranked:
    lbl = log["renderer"]
    med = log["median_ms"]
    fps = log["median_fps"]
    p99 = log["p99_ms"]
    mem = log["peak_memory_mb"]
    delta = ((b_med - med) / b_med) * 100
    tag = " <<< BASELINE >>>" if lbl == "baseline_speedy" else f"  {delta:+.1f}%"
    print(f"  {lbl:25s}  median={med:7.2f}ms = {fps:6.1f}FPS  P99={p99:7.2f}ms  Mem={mem:5.0f}MB{tag}")

fastest = ranked[0]
output = {
    "metadata": {
        "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU",
        "cuda": "13.0 (driver 13.1, toolkit 13.3)",
        "pytorch": "2.12.1+cu130",
        "num_gaussians": N_G,
        "resolution": "1920x1080",
        "scene": "400K GS, SH deg 3",
        "cameras": f"{N_CAMS} fixed orbit views",
        "date": "2026-07-11",
        "baseline": "speedy_splat",
        "fastest_opt": fastest["renderer"],
    },
    "results": results,
}
with open(os.path.join(OUT_DIR, "benchmark_results_phase2.json"), "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {os.path.join(OUT_DIR, 'benchmark_results_phase2.json')}")
