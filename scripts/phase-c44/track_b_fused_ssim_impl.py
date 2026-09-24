#!/usr/bin/env python3
"""
Track B: Fused SSIM implementation and validation.

Fusion strategy:
1. Precompute Gaussian kernel ONCE (not every call)
2. Concatenate 5 inputs [pred, target, pred², target², pred*target] → 15 channels
3. Single grouped conv2d (groups=15) → 5 blurred outputs simultaneously
4. Fuse all elementwise ops into minimal intermediate tensors

This replaces 5 conv2d launches + ~15 elementwise launches
with 1 conv2d + ~5 elementwise launches.

Validation:
- Numerical correctness vs original d_ssim_loss
- Actual timing improvement
"""
import sys, torch, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
import torch.nn.functional as F
from loss import d_ssim_loss

DEVICE = "cuda"
torch.manual_seed(42)

class FusedSSIMLoss:
    """Fused SSIM loss with precomputed kernel and single conv2d."""

    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, n_channels=3, device="cuda"):
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.n_channels = n_channels

        # Precompute kernel ONCE
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = kernel_1d[:, None] * kernel_1d[None, :]  # [11, 11]

        # Fused conv2d: 5 groups, each group = 3-channel RGB blur (depthwise within group)
        # groups=5, in_channels=15, out_channels=15, each group: in=3, out=3
        # Weight shape: [15, 1, 11, 11] — each of 15 output channels has 1 input channel
        # With groups=5, each group has 3 input channels and 3 output channels
        # Weight per group: [3, 1, 11, 11] — this IS depthwise within each group
        # This exactly matches the original groups=3 behavior per blur operation

        self.kernel = kernel.unsqueeze(0).repeat(15, 1, 1, 1).contiguous()  # [15, 1, 11, 11]
        self.padding = window_size // 2
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def __call__(self, pred, target):
        """Compute D-SSIM loss. pred/target: [H,W,3] or [B,3,H,W]."""
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        elif pred.ndim == 4:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        # Compute elementwise products (3 tensors)
        pred_sq = pred ** 2
        target_sq = target ** 2
        pred_target = pred * target

        # Concatenate: [pred, target, pred², target², pred*target] → [1, 15, H, W]
        stacked = torch.cat([pred, target, pred_sq, target_sq, pred_target], dim=1)

        # Single fused conv2d (groups=5, each group = 3ch RGB depthwise blur)
        blurred = F.conv2d(stacked, self.kernel, padding=self.padding, groups=5)

        # Split back into 5 groups of 3 channels
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

        ssim_map = (
            (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)
        ) / (
            (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)
        )

        return 1.0 - ssim_map.mean()

    def __call__scaled(self, pred, target, scale):
        """SSIM with C42 downscale applied before computation."""
        if scale < 1.0:
            if pred.ndim == 3:
                pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
                target = target.unsqueeze(0).permute(0, 3, 1, 2)
            pred_s = F.interpolate(pred, scale_factor=scale, mode="area")
            target_s = F.interpolate(target, scale_factor=scale, mode="area")
            pred_s = pred_s.squeeze(0).permute(1, 2, 0)
            target_s = target_s.squeeze(0).permute(1, 2, 0)
            return self(pred_s, target_s)
        return self(pred, target)


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

# Test data
H, W = 1080, 1920
pred = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)
gt = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)
pred_50 = F.interpolate(pred.unsqueeze(0).permute(0,3,1,2), scale_factor=0.5, mode="area").squeeze(0).permute(1,2,0)
gt_50 = F.interpolate(gt.unsqueeze(0).permute(0,3,1,2), scale_factor=0.5, mode="area").squeeze(0).permute(1,2,0)

# Initialize fused SSIM
fused_ssim = FusedSSIMLoss(device=DEVICE)

# 1. Numerical correctness
print("=" * 70)
print("Track B: Fused SSIM — Numerical Correctness")
print("=" * 70)

loss_orig = d_ssim_loss(pred, gt)
loss_fused = fused_ssim(pred, gt)
diff = (loss_fused - loss_orig).abs().item()
print(f"  Original d_ssim_loss: {loss_orig.item():.10f}")
print(f"  Fused d_ssim_loss:    {loss_fused.item():.10f}")
print(f"  Absolute difference:  {diff:.2e}")
print(f"  Relative difference:  {diff/loss_orig.item()*100:.6f}%")
print(f"  Correct: {'PASS' if diff < 1e-5 else 'FAIL'}")

# Test at scale 0.5
loss_orig_50 = d_ssim_loss(pred_50, gt_50)
loss_fused_50 = fused_ssim(pred_50, gt_50)
diff_50 = (loss_fused_50 - loss_orig_50).abs().item()
print(f"\n  At scale=0.5:")
print(f"  Original: {loss_orig_50.item():.10f}")
print(f"  Fused:    {loss_fused_50.item():.10f}")
print(f"  Diff:     {diff_50:.2e}")
print(f"  Correct: {'PASS' if diff_50 < 1e-5 else 'FAIL'}")

# 2. Timing comparison
print(f"\n{'='*70}")
print("Track B: Fused SSIM — Timing Comparison")
print(f"{'='*70}")

# Warmup
for _ in range(5):
    _ = d_ssim_loss(pred, gt)
    _ = fused_ssim(pred, gt)
torch.cuda.synchronize()

t_orig_10 = time_func(lambda: d_ssim_loss(pred, gt), n=100, warmup=10)
t_fused_10 = time_func(lambda: fused_ssim(pred, gt), n=100, warmup=10)
t_orig_50 = time_func(lambda: d_ssim_loss(pred_50, gt_50), n=100, warmup=10)
t_fused_50 = time_func(lambda: fused_ssim(pred_50, gt_50), n=100, warmup=10)

print(f"  Scale 1.0 (1080p):")
print(f"    Original: {t_orig_10:.2f} ms")
print(f"    Fused:    {t_fused_10:.2f} ms")
print(f"    Speedup:  {t_orig_10/t_fused_10:.2f}x ({(1-t_fused_10/t_orig_10)*100:.1f}% faster)")
print(f"  Scale 0.5 (540p):")
print(f"    Original: {t_orig_50:.2f} ms")
print(f"    Fused:    {t_fused_50:.2f} ms")
print(f"    Speedup:  {t_orig_50/t_fused_50:.2f}x ({(1-t_fused_50/t_orig_50)*100:.1f}% faster)")

# 3. E2E speedup estimate
print(f"\n{'='*70}")
print("Track B: E2E Speedup Estimate")
print(f"{'='*70}")

# From C43 Track 0v3 clean timing:
# scale=1.0: total=98.33ms, ssim=75.85ms
# scale=0.5: total=40.27ms, ssim=21.27ms
total_10 = 98.33
total_05 = 40.27
ssim_orig_10 = 75.85
ssim_orig_05 = 21.27

# With fused SSIM:
total_fused_10 = total_10 - ssim_orig_10 + t_fused_10
total_fused_05 = total_05 - ssim_orig_05 + t_fused_50

e2e_10 = (total_10 - total_fused_10) / total_10 * 100
e2e_05 = (total_05 - total_fused_05) / total_05 * 100

# Combined: Fused SSIM + C42 scale=0.5
# total = render(4.57) + fused_ssim(t_fused_50) + backward(~14)
total_combined = 4.57 + t_fused_50 + 14.40  # from C43 data
e2e_combined = (total_10 - total_combined) / total_10 * 100

print(f"  Baseline (scale=1.0, original SSIM): {total_10:.1f} ms")
print(f"  Fused SSIM (scale=1.0):              {total_fused_10:.1f} ms  → {e2e_10:+.1f}%")
print(f"  C42 only (scale=0.5, original SSIM): {total_05:.1f} ms  → +{(total_10-total_05)/total_10*100:.1f}%")
print(f"  Fused SSIM (scale=0.5):              {total_fused_05:.1f} ms  → {e2e_05:+.1f}%")
print(f"  Combined (C42 0.5 + fused SSIM):     {total_combined:.1f} ms  → {e2e_combined:+.1f}%")

# 4. PyTorch profiler on fused version
print(f"\n{'='*70}")
print("PyTorch profiler: Fused SSIM at scale=1.0")
print(f"{'='*70}")

for _ in range(5):
    _ = fused_ssim(pred, gt)
torch.cuda.synchronize()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    _ = fused_ssim(pred, gt)
torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# 5. Decision
print(f"\n{'='*70}")
print("DECISION")
print(f"{'='*70}")
if e2e_10 > 5:
    print(f"  KEEP: E2E speedup {e2e_10:.1f}% > 5% threshold")
    print(f"  Combined with C42: {e2e_combined:.1f}% E2E speedup")
else:
    print(f"  DROP: E2E speedup {e2e_10:.1f}% < 5% threshold")

# Save
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
results = {
    "correctness": {
        "original_loss": loss_orig.item(),
        "fused_loss": loss_fused.item(),
        "abs_diff": diff,
        "rel_diff_pct": diff / loss_orig.item() * 100,
        "correct": diff < 1e-5,
    },
    "timing": {
        "original_10_ms": t_orig_10,
        "fused_10_ms": t_fused_10,
        "speedup_10x": t_orig_10 / t_fused_10,
        "original_05_ms": t_orig_50,
        "fused_05_ms": t_fused_50,
        "speedup_05x": t_orig_50 / t_fused_50,
    },
    "e2e": {
        "baseline_total_ms": total_10,
        "fused_10_total_ms": total_fused_10,
        "fused_10_speedup_pct": e2e_10,
        "c42_only_total_ms": total_05,
        "c42_only_speedup_pct": (total_10 - total_05) / total_10 * 100,
        "fused_05_total_ms": total_fused_05,
        "fused_05_speedup_pct": e2e_05,
        "combined_total_ms": total_combined,
        "combined_speedup_pct": e2e_combined,
    }
}
save_path = repo / "results" / "a100" / "phase-c44" / "track_b_fused_ssim_validation.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
