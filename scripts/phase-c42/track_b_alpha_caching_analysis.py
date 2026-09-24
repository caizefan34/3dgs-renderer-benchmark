#!/usr/bin/env python3
"""
Track B: Quantify alpha recomputation cost in backward.

Strategy: We can't easily modify the CUDA kernel, but we CAN measure
the backward kernel's compute intensity by comparing:
1. Normal backward (with exp() recomputation)
2. A synthetic "cheap backward" where we precompute alpha and pass it

Since we can't modify the kernel directly, we estimate the exp() cost by:
- Measuring the backward kernel time at different GS counts (scales with intersections)
- Computing the exp() throughput on A100 and estimating how many exp() calls happen
- Using the PyTorch profiler to measure the kernel time and estimating exp fraction

Alternative: Count the number of intersections (visible gaussian-pixel pairs)
and multiply by the known exp() latency on A100.
"""
import sys, torch, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from gsplat import rasterization, fully_fused_projection, isect_tiles, isect_offset_encode
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
import torch.nn.functional as F

torch.manual_seed(42)
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
DEVICE = "cuda"

dataset = GTDataset(scene="room", repo_root=repo, resolution="1080p", device=DEVICE)
sfm = load_initial_checkpoint("room", repo, device=DEVICE)
n = sfm["xyz"].shape[0]
print(f"Gaussians: {n:,}")

model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device=DEVICE)
model.init_from_sfm(
    xyz=sfm["xyz"],
    opacity_logit=torch.logit(torch.full((n,1),0.1,device=DEVICE)),
    scales_log=sfm.get("scales"),
    rotations_raw=sfm.get("rotations"),
    shs=sfm.get("shs"))
model.set_sh_degree(3)

cam = dataset.get_camera(0)
gt = dataset.get_gt_image(0)
vm = cam.viewmatrix.unsqueeze(0)
K = cam.K.unsqueeze(0)
W, H = cam.image_width, cam.image_height
tile_size = 16
tw = (W + tile_size - 1) // tile_size
th = (H + tile_size - 1) // tile_size

# Measure intersection count
data = model.forward()
radii, means2d, depths, conics, _ = fully_fused_projection(
    means=data["xyz"], covars=None, quats=data["rotations"], scales=data["scales"],
    viewmats=vm, Ks=K, width=W, height=H, eps2d=0.1)
_, iids, fids = isect_tiles(means2d, radii, depths, tile_size, tw, th, sort=True, packed=False)
n_isects = iids.shape[0]
n_visible_gaussians = len(torch.unique(fids))
print(f"Total intersections (gaussian-tile pairs): {n_isects:,}")
print(f"Visible Gaussians (intersecting any tile): {n_visible_gaussians:,}")
print(f"Pixels per tile: {tile_size**2}")
print(f"Estimated gaussian-pixel pairs (upper bound): {n_isects * tile_size**2:,}")

# Each gaussian-pixel pair in backward requires:
# - 1x __expf() (special function unit)
# - ~6x FMUL/FADD (sigma computation)
# - ~10x FMUL/FADD (gradient computation)
# - 8x atomicAdd (v_rgb[3], v_conic[3], v_xy[2], v_opacity[1] = 9, but v_rgb is 3)

# A100 specs:
# - 108 SMs, each with 4 SFUs, each SFU does 1 exp/log/rsqrt per cycle
# - Clock: ~1.41 GHz
# - Peak exp throughput: 108 * 4 * 1.41e9 = 6.1e11 exp/s = 610 Gexp/s
# - But __expf uses MUFU.EX2 instruction, 1 per SFU per cycle

A100_SMs = 108
A100_SFU_per_SM = 4
A100_clock_GHz = 1.41
peak_exp_per_s = A100_SMs * A100_SFU_per_SM * A100_clock_GHz * 1e9
print(f"\nA100 peak exp throughput: {peak_exp_per_s/1e9:.1f} Gexp/s")

# Time backward kernel
def time_func(fn, n_iter=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter

# Full fwd+bwd timing
def do_fwd_bwd():
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    d2 = model.forward()
    r, a, l = rasterization(
        means=d2["xyz"], quats=d2["rotations"], scales=d2["scales"],
        opacities=d2["opacity"], colors=d2["shs"],
        viewmats=vm, Ks=K, width=W, height=H,
        tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    loss = F.l1_loss(r, gt.unsqueeze(0))
    loss.backward()

def do_fwd_only():
    with torch.no_grad():
        d2 = model.forward()
        r, a, l = rasterization(
            means=d2["xyz"], quats=d2["rotations"], scales=d2["scales"],
            opacities=d2["opacity"], colors=d2["shs"],
            viewmats=vm, Ks=K, width=W, height=H,
            tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
        return r

t_fwd = time_func(do_fwd_only, n_iter=30, warmup=5)
t_fwd_bwd = time_func(do_fwd_bwd, n_iter=20, warmup=5)
t_bwd = t_fwd_bwd - t_fwd

# From PyTorch profiler: rasterize_to_pixels_3dgs_bwd_kernel = 63.5% of total CUDA backward
# But total CUDA backward = 10.375 ms, timed backward = t_bwd
# The bwd kernel is 6.587 ms out of 10.375 ms CUDA = 63.5%
# Scale to our measurement
bwd_kernel_pct = 0.635
estimated_bwd_kernel_ms = t_bwd * bwd_kernel_pct

print(f"\n{'='*60}")
print(f"Backward timing breakdown")
print(f"{'='*60}")
print(f"  Forward (rasterize): {t_fwd:.3f} ms")
print(f"  Fwd+Bwd:             {t_fwd_bwd:.3f} ms")
print(f"  Backward (total):    {t_bwd:.3f} ms")
print(f"  Bwd kernel (est 63.5%): {estimated_bwd_kernel_ms:.3f} ms")

# Estimate exp() cost
# Each intersection contributes to up to 256 pixels (tile_size^2)
# But not all pixels in a tile are inside the gaussian's footprint
# The backward kernel iterates over ALL gaussians in the tile for ALL pixels
# So exp() count = n_isects * tile_size^2 (upper bound)
# But early termination via last_ids reduces this

# From the source: the backward iterates from back to front, and stops at bin_final
# So the actual exp() count is roughly: sum over pixels of (number of contributing gaussians)
# This is approximately: render_alphas.sum() * avg_gaussians_per_pixel (rough estimate)

# More precisely: n_isects * tile_size^2 * (fraction of gaussians that are valid)
# With early termination, this is reduced. Let's use the upper bound.

exp_calls_upper = n_isects * tile_size**2
exp_calls_lower = n_isects  # at least 1 exp per intersection (for 1 pixel)

# exp() time estimate
exp_time_upper_ms = exp_calls_upper / peak_exp_per_s * 1000
exp_time_lower_ms = exp_calls_lower / peak_exp_per_s * 1000

print(f"\n{'='*60}")
print(f"exp() recomputation cost estimate")
print(f"{'='*60}")
print(f"  Intersections: {n_isects:,}")
print(f"  exp() calls (upper, all pixels): {exp_calls_upper:,}")
print(f"  exp() calls (lower, 1 per isect): {exp_calls_lower:,}")
print(f"  Peak exp throughput: {peak_exp_per_s/1e9:.1f} Gexp/s")
print(f"  exp() time (upper): {exp_time_upper_ms:.3f} ms")
print(f"  exp() time (lower): {exp_time_lower_ms:.3f} ms")
print(f"  Bwd kernel time: {estimated_bwd_kernel_ms:.3f} ms")
print(f"  exp() fraction (upper): {exp_time_upper_ms/estimated_bwd_kernel_ms*100:.1f}%")
print(f"  exp() fraction (lower): {exp_time_lower_ms/estimated_bwd_kernel_ms*100:.1f}%")

# Memory cost of caching alpha per intersection
# Each intersection needs: alpha (4 bytes) + vis (4 bytes) = 8 bytes
# Or just vis = 4 bytes
cache_size_bytes = n_isects * 4  # just vis
cache_size_MB = cache_size_bytes / 1e6
print(f"\n{'='*60}")
print(f"Alpha/vis caching memory cost")
print(f"{'='*60}")
print(f"  Cache size (vis only): {cache_size_MB:.1f} MB ({n_isects:,} entries)")
print(f"  Cache size (alpha+vis): {cache_size_MB*2:.1f} MB")
print(f"  A100 memory: 40 GB")
print(f"  Memory overhead: {cache_size_MB/40000*100:.3f}%")

# Potential backward speedup if exp() is eliminated
exp_time_midpoint = (exp_time_upper_ms + exp_time_lower_ms) / 2
bwd_speedup_pct = exp_time_midpoint / estimated_bwd_kernel_ms * 100
# E2e speedup: backward is ~18% of total training step (from Track 0v3)
# But if we include SSIM, backward is ~18/98 = 18.3%
bwd_fraction_of_total = t_bwd / t_fwd_bwd  # rough
e2e_speedup_pct = exp_time_midpoint / t_fwd_bwd * 100

print(f"\n{'='*60}")
print(f"Potential speedup from alpha caching")
print(f"{'='*60}")
print(f"  exp() time (midpoint): {exp_time_midpoint:.3f} ms")
print(f"  Backward kernel speedup: {bwd_speedup_pct:.1f}%")
print(f"  E2E speedup (vs fwd+bwd only): {e2e_speedup_pct:.1f}%")
print(f"  With C42 (scale=0.75, total=66 ms): {exp_time_midpoint/66*100:.1f}%")
print(f"  Without C42 (scale=1.0, total=98 ms): {exp_time_midpoint/98*100:.1f}%")

# Decision
print(f"\n{'='*60}")
print(f"DECISION")
print(f"{'='*60}")
if e2e_speedup_pct > 5:
    print(f"  KEEP: E2E speedup {e2e_speedup_pct:.1f}% > 5% threshold")
elif e2e_speedup_pct > 3:
    print(f"  MARGINAL: E2E speedup {e2e_speedup_pct:.1f}% (between 3-5%)")
else:
    print(f"  DROP: E2E speedup {e2e_speedup_pct:.1f}% < 3% threshold")

results = {
    "scene": "room",
    "n_gaussians": n,
    "n_isects": int(n_isects),
    "n_visible_gaussians": int(n_visible_gaussians),
    "timing": {
        "forward_ms": t_fwd,
        "fwd_bwd_ms": t_fwd_bwd,
        "backward_ms": t_bwd,
        "estimated_bwd_kernel_ms": estimated_bwd_kernel_ms,
    },
    "exp_estimate": {
        "peak_exp_throughput_Gexp_s": peak_exp_per_s / 1e9,
        "exp_calls_upper": int(exp_calls_upper),
        "exp_calls_lower": int(exp_calls_lower),
        "exp_time_upper_ms": exp_time_upper_ms,
        "exp_time_lower_ms": exp_time_lower_ms,
        "exp_time_midpoint_ms": exp_time_midpoint,
        "exp_fraction_upper": exp_time_upper_ms / estimated_bwd_kernel_ms,
        "exp_fraction_lower": exp_time_lower_ms / estimated_bwd_kernel_ms,
    },
    "caching": {
        "cache_size_MB": cache_size_MB,
        "memory_overhead_pct": cache_size_MB / 40000 * 100,
        "bwd_kernel_speedup_pct": bwd_speedup_pct,
        "e2e_speedup_pct": e2e_speedup_pct,
    }
}

save_path = repo / "results" / "a100" / "phase-c42" / "track_b_alpha_caching_analysis.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
