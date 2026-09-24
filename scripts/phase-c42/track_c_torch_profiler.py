#!/usr/bin/env python3
"""
Track C: Backward kernel profiling using PyTorch profiler + nvprof.
Collects kernel-level timing for backward pass.
"""
import sys, torch, json
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
import torch.nn.functional as F

torch.manual_seed(42)
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
dataset = GTDataset(scene="room", repo_root=repo, resolution="1080p", device="cuda")
sfm = load_initial_checkpoint("room", repo, device="cuda")
n = sfm["xyz"].shape[0]
print(f"Gaussians: {n:,}")

model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device="cuda")
model.init_from_sfm(
    xyz=sfm["xyz"],
    opacity_logit=torch.logit(torch.full((n,1),0.1,device="cuda")),
    scales_log=sfm.get("scales"),
    rotations_raw=sfm.get("rotations"),
    shs=sfm.get("shs"))
model.set_sh_degree(3)

cam = dataset.get_camera(0)
gt = dataset.get_gt_image(0)
vm = cam.viewmatrix.unsqueeze(0)
K = cam.K.unsqueeze(0)

# Warmup
for _ in range(5):
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    d = model.forward()
    r, a, l = rasterization(
        means=d["xyz"], quats=d["rotations"], scales=d["scales"],
        opacities=d["opacity"], colors=d["shs"],
        viewmats=vm, Ks=K, width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    loss = F.l1_loss(r, gt.unsqueeze(0))
    loss.backward()
torch.cuda.synchronize()

# Time backward only (CUDA events)
from gsplat import fully_fused_projection, isect_tiles, isect_offset_encode

def time_func(fn, n=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n

d = model.forward()

# Forward projection
def do_proj():
    return fully_fused_projection(
        means=d["xyz"], covars=None, quats=d["rotations"], scales=d["scales"],
        viewmats=vm, Ks=K, width=cam.image_width, height=cam.image_height, eps2d=0.1)

t_proj = time_func(do_proj)
radii, means2d, depths, conics, _ = do_proj()

tile_size = 16
tw = (cam.image_width + tile_size - 1) // tile_size
th = (cam.image_height + tile_size - 1) // tile_size

def do_isect():
    return isect_tiles(means2d, radii, depths, tile_size, tw, th, sort=True, packed=False)

t_isect = time_func(do_isect)
_, iids, fids = do_isect()

def do_offset():
    return isect_offset_encode(iids, 1, tw, th)

t_offset = time_func(do_offset)

# Full forward
def do_fwd():
    with torch.no_grad():
        d2 = model.forward()
        r, a, l = rasterization(
            means=d2["xyz"], quats=d2["rotations"], scales=d2["scales"],
            opacities=d2["opacity"], colors=d2["shs"],
            viewmats=vm, Ks=K, width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
        return r, a

t_fwd = time_func(do_fwd)

# Full fwd+bwd
def do_fwd_bwd():
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    d2 = model.forward()
    r, a, l = rasterization(
        means=d2["xyz"], quats=d2["rotations"], scales=d2["scales"],
        opacities=d2["opacity"], colors=d2["shs"],
        viewmats=vm, Ks=K, width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    loss = F.l1_loss(r, gt.unsqueeze(0))
    loss.backward()

t_fwd_bwd = time_func(do_fwd_bwd, n=20, warmup=5)
t_bwd = t_fwd_bwd - t_fwd

print(f"\n{'='*60}")
print(f"Backward kernel profiling (room, {n:,} Gaussians)")
print(f"{'='*60}")
print(f"  Projection:  {t_proj:.3f} ms")
print(f"  Isect+sort:  {t_isect:.3f} ms")
print(f"  Offset:      {t_offset:.3f} ms")
print(f"  Forward:     {t_fwd:.3f} ms")
print(f"  Fwd+Bwd:     {t_fwd_bwd:.3f} ms")
print(f"  Backward:    {t_bwd:.3f} ms ({t_bwd/t_fwd_bwd*100:.1f}% of total)")

# PyTorch profiler for kernel breakdown
print(f"\n{'='*60}")
print("PyTorch profiler: backward kernel breakdown")
print(f"{'='*60}")

for p in model.parameters():
    if p.grad is not None: p.grad = None
d2 = model.forward()
r, a, l = rasterization(
    means=d2["xyz"], quats=d2["rotations"], scales=d2["scales"],
    opacities=d2["opacity"], colors=d2["shs"],
    viewmats=vm, Ks=K, width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
loss = F.l1_loss(r, gt.unsqueeze(0))

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    loss.backward()
torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

# Save results
results = {
    "scene": "room",
    "n_gaussians": n,
    "timing": {
        "projection_ms": t_proj,
        "isect_sort_ms": t_isect,
        "offset_ms": t_offset,
        "forward_ms": t_fwd,
        "fwd_bwd_ms": t_fwd_bwd,
        "backward_ms": t_bwd,
        "backward_pct": t_bwd / t_fwd_bwd * 100,
    }
}
save_path = repo / "results" / "a100" / "phase-c42" / "track_c_backward_profile.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
