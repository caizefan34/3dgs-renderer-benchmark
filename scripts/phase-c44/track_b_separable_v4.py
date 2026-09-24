#!/usr/bin/env python3
"""
Track B v4: Fast + exact SSIM via manual depthwise convolution.

Instead of relying on cuDNN's grouped conv (which has numerical differences),
implement the Gaussian blur as a manual depthwise convolution using unfold+matmul
or using torch's unfold-based approach.

Alternatively: use conv2d with groups=3 but batch all 5 inputs into [5,3,H,W]
and call conv2d ONCE with groups=3. This gives exact same algorithm as original.

v2 showed this only gives 1.03x because cuDNN doesn't parallelize across batch.
But we can try: torch.backends.cudnn.enabled = False for this conv,
or use a custom unfold-based depthwise conv.

Key insight: The 11x11 Gaussian kernel is separable! It's outer(1d, 1d).
So we can do 2 × 1D convolutions (11x1 then 1x11) instead of 1 × 2D (11x11).
This reduces FLOPs from 11²×H×W×C = 121×H×W×C to 2×11×H×W×C = 22×H×W×C (5.5x fewer).
"""
import sys, torch, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
import torch.nn.functional as F
from loss import d_ssim_loss

DEVICE = "cuda"
torch.manual_seed(42)

class SeparableFusedSSIM:
    """SSIM with separable Gaussian blur (2 × 1D conv) + batch-stacked for exactness.
    
    The 11x11 Gaussian kernel = outer(k1d, k1d). By the associative property
    of convolution, conv2d(x, outer(k1d,k1d)) = conv1d(conv1d(x, k1d, axis=H), k1d, axis=W).
    
    This reduces work by 5.5x AND allows using the same groups=3 per conv.
    """
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # 1D kernels for separable conv: [3, 1, 11] and [3, 1, 11]
        self.kernel_h = kernel_1d.expand(3, 1, window_size).contiguous()  # horizontal
        self.kernel_v = kernel_1d.expand(3, 1, window_size).contiguous()  # vertical
        self.padding = window_size // 2

    def blur(self, x):
        """Apply separable Gaussian blur to [B,3,H,W]."""
        # Horizontal: conv2d with [3,1,1,11] kernel (height=1, width=11)
        kernel_1x11 = self.kernel_h.unsqueeze(2)  # [3, 1, 1, 11]
        # Vertical: conv2d with [3,1,11,1] kernel (height=11, width=1)
        kernel_11x1 = self.kernel_v.unsqueeze(3)  # [3, 1, 11, 1]
        
        # Apply horizontal then vertical (order doesn't matter for separable)
        x = F.conv2d(x, kernel_1x11, padding=(0, self.padding), groups=3)
        x = F.conv2d(x, kernel_11x1, padding=(self.padding, 0), groups=3)
        return x

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        elif pred.ndim == 4:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        # Stack all 5 inputs along batch: [5, 3, H, W]
        stacked = torch.stack([pred, target, pred**2, target**2, pred*target], dim=0).squeeze(1)
        # Wait: pred is [1,3,H,W], stack gives [5,1,3,H,W], squeeze(1) → [5,3,H,W]
        # Actually: pred is [1,3,H,W] after permute. Let me handle this properly.
        if pred.shape[0] == 1:
            stacked = torch.stack([
                pred[0], target[0], pred[0]**2, target[0]**2, pred[0]*target[0]
            ], dim=0)  # [5, 3, H, W]
        else:
            # Batch case: not typical for 3DGS, skip for now
            stacked = torch.stack([pred, target, pred**2, target**2, pred*target], dim=0).squeeze(1)

        blurred = self.blur(stacked)  # [5, 3, H, W]

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

H, W = 1080, 1920
pred = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)
gt = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)
pred_50 = F.interpolate(pred.unsqueeze(0).permute(0,3,1,2), scale_factor=0.5, mode="area").squeeze(0).permute(1,2,0)
gt_50 = F.interpolate(gt.unsqueeze(0).permute(0,3,1,2), scale_factor=0.5, mode="area").squeeze(0).permute(1,2,0)

sep_ssim = SeparableFusedSSIM(device=DEVICE)

# Correctness
print("=" * 70)
print("Track B v4: Separable Fused SSIM (2×1D conv, batch-stacked)")
print("=" * 70)
loss_orig = d_ssim_loss(pred, gt)
loss_sep = sep_ssim(pred, gt)
diff = abs(loss_sep.item() - loss_orig.item())
print(f"  Original: {loss_orig.item():.10f}")
print(f"  Separable: {loss_sep.item():.10f}")
print(f"  Diff: {diff:.2e}  {'PASS' if diff < 1e-6 else 'FAIL' if diff > 1e-4 else 'CLOSE'}")

loss_orig_50 = d_ssim_loss(pred_50, gt_50)
loss_sep_50 = sep_ssim(pred_50, gt_50)
diff_50 = abs(loss_sep_50.item() - loss_orig_50.item())
print(f"  Scale 0.5: diff={diff_50:.2e}  {'PASS' if diff_50 < 1e-6 else 'FAIL' if diff_50 > 1e-4 else 'CLOSE'}")

# Gradient
pred_g = pred.clone().requires_grad_(True)
d_ssim_loss(pred_g, gt).backward()
grad_orig = pred_g.grad.clone()
pred_g2 = pred.clone().requires_grad_(True)
sep_ssim(pred_g2, gt).backward()
grad_sep = pred_g2.grad.clone()
grad_diff = (grad_sep - grad_orig).abs()
print(f"  Gradient max diff: {grad_diff.max().item():.2e}")

# Timing
print(f"\n{'='*70}")
print("Timing")
print(f"{'='*70}")
for _ in range(5):
    _ = d_ssim_loss(pred, gt)
    _ = sep_ssim(pred, gt)
torch.cuda.synchronize()

t_orig = time_func(lambda: d_ssim_loss(pred, gt))
t_sep = time_func(lambda: sep_ssim(pred, gt))
t_orig_50 = time_func(lambda: d_ssim_loss(pred_50, gt_50))
t_sep_50 = time_func(lambda: sep_ssim(pred_50, gt_50))

print(f"  Scale 1.0: orig={t_orig:.2f}ms  sep={t_sep:.2f}ms  speedup={t_orig/t_sep:.1f}x")
print(f"  Scale 0.5: orig={t_orig_50:.2f}ms  sep={t_sep_50:.2f}ms  speedup={t_orig_50/t_sep_50:.1f}x")

# E2E
total_10 = 98.33
total_05 = 40.27
total_sep_10 = total_10 - 75.85 + t_sep
total_sep_05 = total_05 - 21.27 + t_sep_50
total_combined = 4.57 + t_sep_50 + 14.40

print(f"\n  E2E (sep, scale=1.0):  {total_sep_10:.1f}ms → +{(total_10-total_sep_10)/total_10*100:.1f}%")
print(f"  E2E (sep, scale=0.5):  {total_sep_05:.1f}ms → +{(total_10-total_sep_05)/total_10*100:.1f}%")
print(f"  Combined (C42+sep):    {total_combined:.1f}ms → +{(total_10-total_combined)/total_10*100:.1f}%")

# 100-iter training comparison
print(f"\n{'='*70}")
print("100-iter training comparison")
print(f"{'='*70}")
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
dataset = GTDataset(scene="room", repo_root=repo, resolution="1080p", device=DEVICE)
sfm = load_initial_checkpoint("room", repo, device=DEVICE)
n_gs = sfm["xyz"].shape[0]

def create_model():
    torch.manual_seed(42)
    m = GaussianModel(num_points=n_gs, sh_degree=3, max_sh_degree=3, device=DEVICE)
    m.init_from_sfm(xyz=sfm["xyz"], opacity_logit=torch.logit(torch.full((n_gs,1),0.1,device=DEVICE)),
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

t_o, l_o = train_100(d_ssim_loss, "Original")
t_s, l_s = train_100(sep_ssim, "Separable")

train_speedup = (t_o - t_s) / t_o * 100
loss_diff = abs(l_s[-1] - l_o[-1])
print(f"\n  Training speedup: {train_speedup:.1f}%")
print(f"  Final loss diff: {loss_diff:.6f} ({'acceptable' if loss_diff < 0.01 else 'TOO LARGE'})")

print(f"\n{'='*60}")
print(f"DECISION")
print(f"{'='*60}")
if train_speedup > 5 and loss_diff < 0.01:
    print(f"  KEEP: {train_speedup:.1f}% training speedup, loss diff {loss_diff:.6f}")
elif train_speedup > 5:
    print(f"  REVIEW: speedup={train_speedup:.1f}% but loss_diff={loss_diff:.6f}")
else:
    print(f"  DROP: speedup={train_speedup:.1f}%")

results = {
    "numerical": {"orig": loss_orig.item(), "separable": loss_sep.item(), "diff": diff},
    "gradient": {"max_diff": grad_diff.max().item()},
    "timing": {"orig_10": t_orig, "sep_10": t_sep, "speedup_10": t_orig/t_sep,
               "orig_05": t_orig_50, "sep_05": t_sep_50, "speedup_05": t_orig_50/t_sep_50},
    "training": {"orig_s": t_o, "sep_s": t_s, "speedup_pct": train_speedup,
                 "loss_orig": l_o, "loss_sep": l_s, "loss_diff": loss_diff},
}
save_path = repo / "results" / "a100" / "phase-c44" / "track_b_separable_v4.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {save_path}")
