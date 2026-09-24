#!/usr/bin/env python3
"""
Track B v3: Tolerance test for groups=15 fused SSIM (8.4x faster, 3.6e-4 diff).

Key question: Does the 3.6e-4 cuDNN numerical difference affect training quality?

Test: 100 training iterations with original vs fused SSIM, compare loss curves.
"""
import sys, torch, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
import torch.nn.functional as F
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss

torch.manual_seed(42)
DEVICE = "cuda"
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")

class FusedSSIM:
    """Fused SSIM with groups=15 (depthwise). 8.4x faster, ~3.6e-4 numerical diff from cuDNN."""
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = kernel_1d[:, None] * kernel_1d[None, :]
        self.kernel = kernel.unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        elif pred.ndim == 4:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)
        pred_sq = pred ** 2
        target_sq = target ** 2
        pred_target = pred * target
        stacked = torch.cat([pred, target, pred_sq, target_sq, pred_target], dim=1)
        blurred = F.conv2d(stacked, self.kernel, padding=self.padding, groups=15)
        mu_pred = blurred[:, 0:3]
        mu_target = blurred[:, 3:6]
        blur_pred_sq = blurred[:, 6:9]
        blur_target_sq = blurred[:, 9:12]
        blur_pred_target = blurred[:, 12:15]
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        sigma_pred_sq = blur_pred_sq - mu_pred_sq
        sigma_target_sq = blur_target_sq - mu_target_sq
        sigma_pred_target = blur_pred_target - mu_pred_target
        ssim_map = (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2) / (
            (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))
        return 1.0 - ssim_map.mean()


def time_func(fn, n=100, warmup=10):
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
fused_ssim = FusedSSIM(device=DEVICE)

# Numerical
H, W = 1080, 1920
pred = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)
gt = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)
loss_orig = d_ssim_loss(pred, gt)
loss_fused = fused_ssim(pred, gt)
print(f"Numerical: orig={loss_orig.item():.10f} fused={loss_fused.item():.10f} diff={abs(loss_fused.item()-loss_orig.item()):.2e}")

# Gradient
pred_g = pred.clone().requires_grad_(True)
d_ssim_loss(pred_g, gt).backward()
grad_orig = pred_g.grad.clone()
pred_g2 = pred.clone().requires_grad_(True)
fused_ssim(pred_g2, gt).backward()
grad_fused = pred_g2.grad.clone()
grad_diff = (grad_fused - grad_orig).abs()
print(f"Gradient: max_diff={grad_diff.max().item():.2e} mean_diff={grad_diff.mean().item():.2e}")
print(f"  norm: orig={grad_orig.norm().item():.4f} fused={grad_fused.norm().item():.4f}")

# Timing
t_orig = time_func(lambda: d_ssim_loss(pred, gt))
t_fused = time_func(lambda: fused_ssim(pred, gt))
print(f"Timing: orig={t_orig:.2f}ms fused={t_fused:.2f}ms speedup={t_orig/t_fused:.1f}x")

# 100-iter training comparison
def create_model():
    torch.manual_seed(42)
    m = GaussianModel(num_points=n, sh_degree=3, max_sh_degree=3, device=DEVICE)
    m.init_from_sfm(xyz=sfm["xyz"], opacity_logit=torch.logit(torch.full((n,1),0.1,device=DEVICE)),
                     scales_log=sfm.get("scales"), rotations_raw=sfm.get("rotations"), shs=sfm.get("shs"))
    m.set_sh_degree(3)
    return m

def train_100(ssim_fn, label):
    model = create_model()
    opt = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4}, {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3}, {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3}], eps=1e-15)
    cams = list(range(len(dataset)))
    np.random.shuffle(cams)
    losses = []
    t0 = time.perf_counter()
    for i in range(100):
        ci = cams[i % len(dataset)]
        cam = dataset.get_camera(ci)
        gt_img = dataset.get_gt_image(ci)
        data = model.forward()
        r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"], viewmats=cam.viewmatrix.unsqueeze(0),
            Ks=cam.K.unsqueeze(0), width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=3, radius_clip=0.0, eps2d=0.1, render_mode="RGB")
        pred = r[0].clamp(0, 1)
        loss = 0.8 * F.l1_loss(pred, gt_img) + 0.2 * ssim_fn(pred, gt_img)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 20 == 0:
            losses.append(loss.item())
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.1f}s, losses={['%.4f' % l for l in losses]}")
    return elapsed, losses

print("\n100-iter training comparison:")
t_o, l_o = train_100(d_ssim_loss, "Original")
t_f, l_f = train_100(fused_ssim, "Fused g15")

train_speedup = (t_o - t_f) / t_o * 100
loss_diff = abs(l_f[-1] - l_o[-1])
print(f"\n  Training speedup: {train_speedup:.1f}%")
print(f"  Final loss diff: {loss_diff:.6f} ({'acceptable' if loss_diff < 0.01 else 'TOO LARGE'})")
print(f"  Loss trajectory diff at each checkpoint: {[abs(l_f[j]-l_o[j]) for j in range(len(l_o))]}")

print(f"\n{'='*60}")
print(f"DECISION")
print(f"{'='*60}")
if train_speedup > 5 and loss_diff < 0.01:
    print(f"  KEEP: {train_speedup:.1f}% training speedup, loss diff {loss_diff:.6f}")
else:
    print(f"  REVIEW: speedup={train_speedup:.1f}%, loss_diff={loss_diff:.6f}")

results = {
    "numerical": {"orig": loss_orig.item(), "fused": loss_fused.item(), "diff": abs(loss_fused.item()-loss_orig.item())},
    "gradient": {"max_diff": grad_diff.max().item(), "mean_diff": grad_diff.mean().item()},
    "timing": {"orig_ms": t_orig, "fused_ms": t_fused, "speedup_x": t_orig/t_fused},
    "training": {"orig_s": t_o, "fused_s": t_f, "speedup_pct": train_speedup,
                 "loss_orig": l_o, "loss_fused": l_f, "loss_diff": loss_diff},
}
save_path = repo / "results" / "a100" / "phase-c44" / "track_b_fused_v3_tolerance.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {save_path}")
