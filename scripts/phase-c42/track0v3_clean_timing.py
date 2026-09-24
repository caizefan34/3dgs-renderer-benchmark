#!/usr/bin/env python3
"""
Track 0v3: Clean steady-state timing of C42 SSIM downscale.

Trains for 2000 iters to stabilize GS count, then does PURE timing
of render + loss (with SSIM at each scale) + backward, WITHOUT
any densification/pruning overhead.

This gives clean C42 speedup measurements.
"""
import sys, torch, json, time, math, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss
import torch.nn.functional as F

torch.manual_seed(42)
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
DEVICE = "cuda"

dataset = GTDataset(scene="room", repo_root=repo, resolution="1080p", device=DEVICE)
sfm = load_initial_checkpoint("room", repo, device=DEVICE)
n = sfm["xyz"].shape[0]
print(f"Gaussians: {n:,}")

# Train model for 2000 iters to stabilize
print("Training 2000 iters to stabilize GS count...")
model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device=DEVICE)
model.init_from_sfm(
    xyz=sfm["xyz"],
    opacity_logit=torch.logit(torch.full((n,1),0.1,device=DEVICE)),
    scales_log=sfm.get("scales"),
    rotations_raw=sfm.get("rotations"),
    shs=sfm.get("shs"))
model.set_sh_degree(3)

lr_params = [
    {"params": [model.xyz], "lr": 1.6e-4, "name": "xyz"},
    {"params": [model.rotations], "lr": 1e-3, "name": "rotations"},
    {"params": [model.scales], "lr": 5e-3, "name": "scales"},
    {"params": [model.opacity], "lr": 5e-2, "name": "opacity"},
    {"params": [model.shs], "lr": 2.5e-3, "name": "shs"},
]
optimizer = torch.optim.Adam(lr_params, eps=1e-15)

n_cams = len(dataset)
cam_indices = list(range(n_cams))
np.random.shuffle(cam_indices)

for iter_idx in range(2000):
    model.train()
    ci = cam_indices[iter_idx % n_cams]
    cam = dataset.get_camera(ci)
    gt = dataset.get_gt_image(ci)
    data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    loss = F.l1_loss(r, gt.unsqueeze(0))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if iter_idx >= 500 and iter_idx < 2000 and iter_idx % 100 == 0:
        model.accumulate_positional_gradient()
        model.densification(grad_threshold=0.002)
        model.prune(opacity_threshold=0.05)
    if iter_idx > 0 and iter_idx % 1000 == 0:
        if model.sh_degree < 3:
            model.set_sh_degree(model.sh_degree + 1)
    optimizer.step()

torch.cuda.synchronize()
gs_count = model.xyz.shape[0]
print(f"GS count after training: {gs_count:,}")

# Now do clean timing at each scale
# Use 5 cameras, 20 iters each, no densification
EVAL_CAMS = [0, 50, 100, 150, 200]
N_TIMING = 30

def time_scale(scale):
    times = {"render": [], "ssim": [], "backward": [], "total": []}
    for ci in EVAL_CAMS:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Warmup
        for _ in range(3):
            for p in model.parameters():
                if p.grad is not None: p.grad = None
            data = model.forward()
            r, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
            if scale < 1.0:
                pred_s = F.interpolate(r[0].unsqueeze(0).permute(0,3,1,2), scale_factor=scale, mode="area").squeeze(0).permute(1,2,0)
                gt_s = F.interpolate(gt.unsqueeze(0).permute(0,3,1,2), scale_factor=scale, mode="area").squeeze(0).permute(1,2,0)
            else:
                pred_s, gt_s = r[0], gt
            loss = 0.8 * F.l1_loss(r[0].clamp(0,1), gt) + 0.2 * d_ssim_loss(pred_s, gt_s)
            loss.backward()
        torch.cuda.synchronize()

        for _ in range(N_TIMING):
            for p in model.parameters():
                if p.grad is not None: p.grad = None

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Render
            torch.cuda.synchronize()
            t_r0 = time.perf_counter()
            data = model.forward()
            r, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
            torch.cuda.synchronize()
            t_r1 = time.perf_counter()

            # SSIM loss
            torch.cuda.synchronize()
            t_s0 = time.perf_counter()
            if scale < 1.0:
                pred_s = F.interpolate(r[0].unsqueeze(0).permute(0,3,1,2), scale_factor=scale, mode="area").squeeze(0).permute(1,2,0)
                gt_s = F.interpolate(gt.unsqueeze(0).permute(0,3,1,2), scale_factor=scale, mode="area").squeeze(0).permute(1,2,0)
            else:
                pred_s, gt_s = r[0], gt
            l1 = F.l1_loss(r[0].clamp(0,1), gt)
            dsim = d_ssim_loss(pred_s, gt_s)
            loss = 0.8 * l1 + 0.2 * dsim
            torch.cuda.synchronize()
            t_s1 = time.perf_counter()

            # Backward
            torch.cuda.synchronize()
            t_b0 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t_b1 = time.perf_counter()

            t_total = t_b1 - t0
            times["render"].append((t_r1 - t_r0) * 1000)
            times["ssim"].append((t_s1 - t_s0) * 1000)
            times["backward"].append((t_b1 - t_b0) * 1000)
            times["total"].append(t_total * 1000)

    return {k: np.mean(v) for k, v in times.items()}

results = {"scene": "room", "gs_count": gs_count, "scales": {}}
baseline_total = None

for scale in [1.0, 0.875, 0.75, 0.625, 0.5]:
    print(f"\nTiming scale={scale}...")
    t = time_scale(scale)
    if baseline_total is None:
        baseline_total = t["total"]
    speedup = (baseline_total - t["total"]) / baseline_total * 100
    t["e2e_speedup_pct"] = speedup
    results["scales"][str(scale)] = t
    print(f"  Render: {t['render']:.2f} ms  SSIM: {t['ssim']:.2f} ms  Bwd: {t['backward']:.2f} ms  Total: {t['total']:.2f} ms  Speedup: {speedup:+.1f}%")

print(f"\n{'='*70}")
print("SUMMARY: Track 0v3 - Clean C42 timing (no densification overhead)")
print(f"{'='*70}")
print(f"GS count: {gs_count:,}")
print(f"{'Scale':>8} {'Render':>10} {'SSIM':>10} {'Backward':>10} {'Total':>10} {'Speedup':>10}")
print("-" * 65)
for scale in [1.0, 0.875, 0.75, 0.625, 0.5]:
    t = results["scales"][str(scale)]
    print(f"{scale:>8} {t['render']:>10.2f} {t['ssim']:>10.2f} {t['backward']:>10.2f} {t['total']:>10.2f} {t['e2e_speedup_pct']:>+9.1f}%")

save_path = repo / "results" / "a100" / "phase-c42" / "track0v3_clean_timing.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
