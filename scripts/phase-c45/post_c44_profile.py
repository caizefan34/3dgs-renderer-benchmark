#!/usr/bin/env python3
"""
Post-C44 pipeline re-profiling.

After separable SSIM (75ms -> 3.4ms), the bottleneck shifts.
This script measures the new distribution:
- Render (forward rasterization)
- Separable SSIM loss
- Backward (total + kernel breakdown)
- L1 loss
- Optimizer step
- Densification/pruning overhead

Uses the separable SSIM implementation from C44 Track B v4.
"""
import sys, torch, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
import torch.nn.functional as F
from gsplat import rasterization, fully_fused_projection, isect_tiles, isect_offset_encode
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss

DEVICE = "cuda"
torch.manual_seed(42)
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")

# Separable SSIM from C44 Track B v4
class SeparableSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        self.kernel_h = kernel_1d.expand(3, 1, window_size).contiguous().unsqueeze(2)  # [3,1,1,11]
        self.kernel_v = kernel_1d.expand(3, 1, window_size).contiguous().unsqueeze(3)  # [3,1,11,1]
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.stack([pred[0], target[0], pred[0]**2, target[0]**2, pred[0]*target[0]], dim=0)
        blurred = F.conv2d(stacked, self.kernel_h.unsqueeze(0).repeat(15,1,1,1).contiguous(),
                          padding=(0, self.padding), groups=15)
        blurred = F.conv2d(blurred, self.kernel_v.unsqueeze(0).repeat(15,1,1,1).contiguous() if self.kernel_v.ndim==3 else 
                          self.kernel_v.squeeze(2).unsqueeze(0).repeat(15,1,1,1).contiguous(),
                          padding=(self.padding, 0), groups=15)
        # Actually let me be more careful with kernel shapes
        # kernel_h: [3,1,1,11] for horizontal conv (height=1, width=11)
        # kernel_v: [3,1,11,1] for vertical conv (height=11, width=1)
        # For groups=15 on [5,3,H,W], we need [15,1,1,11] and [15,1,11,1]
        mu_pred = blurred[0]
        mu_target = blurred[1]
        blur_pred_sq = blurred[2]
        blur_target_sq = blurred[3]
        blur_pred_target = blurred[4]
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        sigma_pred_sq = blur_pred_sq - mu_pred_sq
        sigma_target_sq = blur_target_sq - mu_target_sq
        sigma_pred_target = blur_pred_target - mu_pred_target
        ssim_map = (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2) / (
            (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))
        return 1.0 - ssim_map.mean()


# Simpler correct implementation
class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # 1D horizontal kernel: [15, 1, 1, 11] for groups=15
        self.k_h = kernel_1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()  # [15,1,1,11]
        # 1D vertical kernel: [15, 1, 11, 1] for groups=15
        self.k_v = kernel_1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0,1,3,2).contiguous()  # [15,1,11,1]
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)  # [1,3,H,W]
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        pred_sq = pred ** 2
        target_sq = target ** 2
        pred_target = pred * target
        stacked = torch.cat([pred, target, pred_sq, target_sq, pred_target], dim=1)  # [1,15,H,W]
        # Horizontal blur
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        # Vertical blur
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_pred = b[:, 0:3]
        mu_target = b[:, 3:6]
        blur_pred_sq = b[:, 6:9]
        blur_target_sq = b[:, 9:12]
        blur_pred_target = b[:, 12:15]
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        sigma_pred_sq = blur_pred_sq - mu_pred_sq
        sigma_target_sq = blur_target_sq - mu_target_sq
        sigma_pred_target = blur_pred_target - mu_pred_target
        ssim_map = (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2) / (
            (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))
        return 1.0 - ssim_map.mean()


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

dataset = GTDataset(scene="room", repo_root=repo, resolution="1080p", device=DEVICE)
sfm = load_initial_checkpoint("room", repo, device=DEVICE)
n = sfm["xyz"].shape[0]
print(f"Gaussians: {n:,}")

model = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device=DEVICE)
model.init_from_sfm(
    xyz=sfm["xyz"],
    opacity_logit=torch.logit(torch.full((n, 1), 0.1, device=DEVICE)),
    scales_log=sfm.get("scales"),
    rotations_raw=sfm.get("rotations"),
    shs=sfm.get("shs"))
model.set_sh_degree(3)

cam = dataset.get_camera(0)
gt = dataset.get_gt_image(0)
vm = cam.viewmatrix.unsqueeze(0)
K = cam.K.unsqueeze(0)
W, H = cam.image_width, cam.image_height

sep_ssim = SepSSIM(device=DEVICE)

# Verify correctness
loss_orig = d_ssim_loss(cam and (lambda: None)() or gt.unsqueeze(0).permute(0,3,1,2).squeeze(0).permute(2,0,1) if False else gt, gt)
# Just test with rendered image
data = model.forward()
r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
    opacities=data["opacity"], colors=data["shs"], viewmats=vm, Ks=K,
    width=W, height=H, tile_size=16, packed=False, sh_degree=3,
    radius_clip=0.0, eps2d=0.1, render_mode="RGB")
pred = r[0].clamp(0, 1)

loss_orig = d_ssim_loss(pred, gt)
loss_sep = sep_ssim(pred, gt)
print(f"SSIM correctness: orig={loss_orig.item():.8f} sep={loss_sep.item():.8f} diff={abs(loss_sep.item()-loss_orig.item()):.2e}")

# Warmup
for _ in range(5):
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    data = model.forward()
    r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"], viewmats=vm, Ks=K,
        width=W, height=H, tile_size=16, packed=False, sh_degree=3,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    pred = r[0].clamp(0, 1)
    l1 = F.l1_loss(pred, gt)
    dsim = sep_ssim(pred, gt)
    loss = 0.8 * l1 + 0.2 * dsim
    loss.backward()
torch.cuda.synchronize()

# Component timing
print(f"\n{'='*70}")
print("Post-C44 Pipeline Component Timing")
print(f"{'='*70}")

# 1. Forward render only
def do_render():
    with torch.no_grad():
        data = model.forward()
        r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"], viewmats=vm, Ks=K,
            width=W, height=H, tile_size=16, packed=False, sh_degree=3,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB")
        return r

t_render = time_func(do_render, n=30, warmup=5)

# 2. L1 loss only
def do_l1():
    pred = do_render()
    return F.l1_loss(pred[0].clamp(0,1), gt)

t_l1 = time_func(do_l1, n=30, warmup=5) - t_render

# 3. Separable SSIM only
def do_ssim():
    pred = do_render()
    return sep_ssim(pred[0].clamp(0,1), gt)

t_ssim = time_func(do_ssim, n=30, warmup=5) - t_render

# 4. Original SSIM for comparison
def do_ssim_orig():
    pred = do_render()
    return d_ssim_loss(pred[0].clamp(0,1), gt)

t_ssim_orig = time_func(do_ssim_orig, n=30, warmup=5) - t_render

# 5. Full forward + backward (separable SSIM)
def do_fwd_bwd_sep():
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    data = model.forward()
    r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"], viewmats=vm, Ks=K,
        width=W, height=H, tile_size=16, packed=False, sh_degree=3,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    pred = r[0].clamp(0, 1)
    loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * sep_ssim(pred, gt)
    loss.backward()

t_fwd_bwd_sep = time_func(do_fwd_bwd_sep, n=20, warmup=5)
t_bwd_sep = t_fwd_bwd_sep - t_render - t_ssim - t_l1

# 6. Full forward + backward (original SSIM)
def do_fwd_bwd_orig():
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    data = model.forward()
    r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"], viewmats=vm, Ks=K,
        width=W, height=H, tile_size=16, packed=False, sh_degree=3,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    pred = r[0].clamp(0, 1)
    loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * d_ssim_loss(pred, gt)
    loss.backward()

t_fwd_bwd_orig = time_func(do_fwd_bwd_orig, n=20, warmup=5)
t_bwd_orig = t_fwd_bwd_orig - t_render - t_ssim_orig - t_l1

# 7. Optimizer step
opt = torch.optim.Adam([
    {"params": [model.xyz], "lr": 1.6e-4},
    {"params": [model.rotations], "lr": 1e-3},
    {"params": [model.scales], "lr": 5e-3},
    {"params": [model.opacity], "lr": 5e-2},
    {"params": [model.shs], "lr": 2.5e-3}], eps=1e-15)

do_fwd_bwd_sep()  # populate gradients
def do_optim():
    opt.step()
    opt.zero_grad(set_to_none=True)

t_optim = time_func(do_optim, n=30, warmup=5)

# Print results
print(f"\n  Component breakdown (separable SSIM):")
print(f"  {'Component':<30} {'Time (ms)':>10} {'% of total':>12}")
print(f"  {'-'*55}")
total_sep = t_fwd_bwd_sep
print(f"  {'Render (forward)':<30} {t_render:>10.2f} {t_render/total_sep*100:>11.1f}%")
print(f"  {'L1 loss':<30} {t_l1:>10.2f} {t_l1/total_sep*100:>11.1f}%")
print(f"  {'Separable SSIM loss':<30} {t_ssim:>10.2f} {t_ssim/total_sep*100:>11.1f}%")
print(f"  {'Backward':<30} {t_bwd_sep:>10.2f} {t_bwd_sep/total_sep*100:>11.1f}%")
print(f"  {'Total (fwd+bwd+loss)':<30} {total_sep:>10.2f} {'100.0%':>12}")
print(f"  {'Optimizer step':<30} {t_optim:>10.2f}")

print(f"\n  Comparison (original SSIM):")
print(f"  {'Component':<30} {'Time (ms)':>10}")
print(f"  {'-'*40}")
print(f"  {'Original SSIM loss':<30} {t_ssim_orig:>10.2f}")
print(f"  {'Backward (orig SSIM)':<30} {t_bwd_orig:>10.2f}")
print(f"  {'Total (orig)':<30} {t_fwd_bwd_orig:>10.2f}")
print(f"  {'Total (separable)':<30} {total_sep:>10.2f}")
print(f"  {'Speedup':<30} {t_fwd_bwd_orig/total_sep:>9.2f}x")

# PyTorch profiler for post-C44 backward
print(f"\n{'='*70}")
print("PyTorch profiler: Post-C44 backward (separable SSIM)")
print(f"{'='*70}")

for p in model.parameters():
    if p.grad is not None: p.grad = None
data = model.forward()
r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
    opacities=data["opacity"], colors=data["shs"], viewmats=vm, Ks=K,
    width=W, height=H, tile_size=16, packed=False, sh_degree=3,
    radius_clip=0.0, eps2d=0.1, render_mode="RGB")
pred = r[0].clamp(0, 1)
loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * sep_ssim(pred, gt)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    loss.backward()
torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

# Save
results = {
    "scene": "room", "n_gaussians": n,
    "post_c44": {
        "render_ms": t_render, "l1_ms": t_l1, "ssim_sep_ms": t_ssim,
        "backward_ms": t_bwd_sep, "total_ms": total_sep, "optimizer_ms": t_optim,
        "render_pct": t_render/total_sep*100, "l1_pct": t_l1/total_sep*100,
        "ssim_pct": t_ssim/total_sep*100, "backward_pct": t_bwd_sep/total_sep*100,
    },
    "comparison": {
        "ssim_orig_ms": t_ssim_orig, "backward_orig_ms": t_bwd_orig,
        "total_orig_ms": t_fwd_bwd_orig, "total_sep_ms": total_sep,
        "speedup_x": t_fwd_bwd_orig / total_sep,
    }
}
save_path = repo / "results" / "a100" / "phase-c45" / "post_c44_profile.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
