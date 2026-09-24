#!/usr/bin/env python3
"""
Track A: Batch size (tile_size) scaling experiment.

Since batch_size = tile_size^2 in gsplat's rasterizer, changing tile_size
directly changes the batch size:
  tile_size=8  → batch_size=64
  tile_size=16 → batch_size=256 (baseline)
  tile_size=32 → batch_size=1024

Measure: forward time, backward time, gradient correctness.

Gradient correctness: compare model gradients after backward with
tile_size=X vs tile_size=16 (baseline). Use max_abs_gradient_difference.
"""
import sys, torch, json, time, copy
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")

from gsplat import rasterization
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

def create_model():
    torch.manual_seed(42)
    m = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device=DEVICE)
    m.init_from_sfm(
        xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((n, 1), 0.1, device=DEVICE)),
        scales_log=sfm.get("scales"),
        rotations_raw=sfm.get("rotations"),
        shs=sfm.get("shs"))
    m.set_sh_degree(3)
    return m

def render_with_tile(model, cam, tile_size):
    data = model.forward()
    r, a, l = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=tile_size, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    return r[0].clamp(0, 1), a[0], l[0]

def time_func(fn, n_iter=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter

cam = dataset.get_camera(0)
gt = dataset.get_gt_image(0)

results = {"scene": "room", "n_gaussians": n, "tile_sizes": []}

# Baseline: tile_size=16, get reference gradients
print("\n=== Computing reference gradients (tile_size=16) ===")
model_ref = create_model()
d = model_ref.forward()
r, a, l = rasterization(
    means=d["xyz"], quats=d["rotations"], scales=d["scales"],
    opacities=d["opacity"], colors=d["shs"],
    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
    width=cam.image_width, height=cam.image_height,
    tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
loss = F.l1_loss(r, gt.unsqueeze(0))
loss.backward()
ref_grads = {}
for name, p in model_ref.named_parameters():
    if p.grad is not None:
        ref_grads[name] = p.grad.clone()
print(f"  Reference loss: {loss.item():.6f}")
print(f"  Reference gradients: {len(ref_grads)} params")

for tile_size in [8, 16, 32]:
    print(f"\n=== tile_size={tile_size} (batch_size={tile_size**2}) ===")
    try:
        model = create_model()

        # Time forward
        def do_fwd():
            with torch.no_grad():
                d2 = model.forward()
                r, a, l = rasterization(
                    means=d2["xyz"], quats=d2["rotations"], scales=d2["scales"],
                    opacities=d2["opacity"], colors=d2["shs"],
                    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                    width=cam.image_width, height=cam.image_height,
                    tile_size=tile_size, packed=False, sh_degree=3,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB")
                return r

        t_fwd = time_func(do_fwd, n_iter=30, warmup=5)

        # Time fwd+bwd
        def do_fwd_bwd():
            for p in model.parameters():
                if p.grad is not None: p.grad = None
            d2 = model.forward()
            r, a, l = rasterization(
                means=d2["xyz"], quats=d2["rotations"], scales=d2["scales"],
                opacities=d2["opacity"], colors=d2["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=tile_size, packed=False, sh_degree=3,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB")
            loss = F.l1_loss(r, gt.unsqueeze(0))
            loss.backward()

        t_fwd_bwd = time_func(do_fwd_bwd, n_iter=20, warmup=5)
        t_bwd = t_fwd_bwd - t_fwd

        # Gradient correctness
        do_fwd_bwd()  # one more to get gradients
        max_diff = 0.0
        mean_diff = 0.0
        n_params = 0
        for name, p in model.named_parameters():
            if name in ref_grads and p.grad is not None:
                diff = (p.grad - ref_grads[name]).abs()
                max_diff = max(max_diff, diff.max().item())
                mean_diff += diff.mean().item()
                n_params += 1
        mean_diff /= max(n_params, 1)

        # Check rendering correctness
        with torch.no_grad():
            pred = do_fwd()
            render_diff = (pred - r[0].clamp(0, 1)).abs().max().item()

        print(f"  Forward:     {t_fwd:.3f} ms")
        print(f"  Fwd+Bwd:     {t_fwd_bwd:.3f} ms")
        print(f"  Backward:    {t_bwd:.3f} ms ({t_bwd/t_fwd_bwd*100:.1f}%)")
        print(f"  Max grad diff vs baseline: {max_diff:.2e}")
        print(f"  Mean grad diff vs baseline: {mean_diff:.2e}")
        print(f"  Max render diff vs baseline: {render_diff:.2e}")

        # Speedup vs baseline
        if tile_size != 16:
            # Will be computed after baseline
            pass

        results["tile_sizes"].append({
            "tile_size": tile_size,
            "batch_size": tile_size ** 2,
            "forward_ms": t_fwd,
            "fwd_bwd_ms": t_fwd_bwd,
            "backward_ms": t_bwd,
            "backward_pct": t_bwd / t_fwd_bwd * 100,
            "max_grad_diff": max_diff,
            "mean_grad_diff": mean_diff,
            "max_render_diff": render_diff,
            "grad_correct": max_diff < 1e-4,
        })

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        results["tile_sizes"].append({
            "tile_size": tile_size,
            "batch_size": tile_size ** 2,
            "error": str(e),
        })
        import traceback
        traceback.print_exc()

# Compute speedups
baseline_bwd = None
for ts in results["tile_sizes"]:
    if ts["tile_size"] == 16:
        baseline_bwd = ts.get("backward_ms")
        break
if baseline_bwd:
    for ts in results["tile_sizes"]:
        if "backward_ms" in ts:
            ts["bwd_speedup_vs_16"] = (baseline_bwd - ts["backward_ms"]) / baseline_bwd * 100

print(f"\n{'='*60}")
print("SUMMARY: Track A - Batch size (tile_size) scaling")
print(f"{'='*60}")
print(f"{'tile_size':>10} {'batch':>8} {'fwd_ms':>8} {'bwd_ms':>8} {'speedup':>8} {'grad_diff':>12} {'correct':>8}")
print("-" * 70)
for ts in results["tile_sizes"]:
    if "error" in ts:
        print(f"{ts['tile_size']:>10} {ts['batch_size']:>8} ERROR: {ts['error'][:40]}")
    else:
        sp = ts.get("bwd_speedup_vs_16", 0)
        print(f"{ts['tile_size']:>10} {ts['batch_size']:>8} {ts['forward_ms']:>8.2f} {ts['backward_ms']:>8.2f} {sp:>+7.1f}% {ts['max_grad_diff']:>12.2e} {'PASS' if ts['grad_correct'] else 'FAIL':>8}")

save_path = repo / "results" / "a100" / "phase-c42" / "track_a_tile_size.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
