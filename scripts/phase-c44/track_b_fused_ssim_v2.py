#!/usr/bin/env python3
"""
Track B v2: Fused SSIM — correct numerical match.

Strategy: Instead of changing groups, stack 5 inputs along the BATCH dimension
and use the original groups=3 conv2d. This forces cuDNN to use the exact same
algorithm as the original, just batched.

[1,3,H,W] × 5 separate conv2d → [5,3,H,W] × 1 batched conv2d
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
    """Fused SSIM with batch-stacked conv2d for exact numerical match."""

    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, n_channels=3, device="cuda"):
        self.window_size = window_size
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = kernel_1d[:, None] * kernel_1d[None, :]
        self.kernel = kernel.expand(n_channels, 1, window_size, window_size).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        elif pred.ndim == 4:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        pred_sq = pred ** 2
        target_sq = target ** 2
        pred_target = pred * target

        # Stack along batch: [5, 3, H, W]
        stacked = torch.stack([pred, target, pred_sq, target_sq, pred_target], dim=0)
        # Single conv2d with groups=3 (same as original per-call)
        blurred = F.conv2d(stacked.squeeze(1) if stacked.shape[2] != 3 else stacked.reshape(5, 3, pred.shape[2], pred.shape[3]),
                          self.kernel, padding=self.padding, groups=3)
        
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

        ssim_map = (
            (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)
        ) / (
            (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)
        )

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

fused_ssim = FusedSSIMLoss(device=DEVICE)

# Correctness
print("=" * 70)
print("Track B v2: Fused SSIM (batch-stacked) — Correctness")
print("=" * 70)
loss_orig = d_ssim_loss(pred, gt)
loss_fused = fused_ssim(pred, gt)
diff = (loss_fused - loss_orig).abs().item()
print(f"  Original: {loss_orig.item():.10f}")
print(f"  Fused:    {loss_fused.item():.10f}")
print(f"  Diff:     {diff:.2e}")
print(f"  Rel diff: {diff/loss_orig.item()*100:.8f}%")
print(f"  Correct:  {'PASS' if diff < 1e-6 else 'FAIL' if diff > 1e-4 else 'CLOSE'}")

loss_orig_50 = d_ssim_loss(pred_50, gt_50)
loss_fused_50 = fused_ssim(pred_50, gt_50)
diff_50 = (loss_fused_50 - loss_orig_50).abs().item()
print(f"\n  Scale 0.5: orig={loss_orig_50.item():.10f} fused={loss_fused_50.item():.10f} diff={diff_50:.2e}")
print(f"  Correct:  {'PASS' if diff_50 < 1e-6 else 'FAIL' if diff_50 > 1e-4 else 'CLOSE'}")

# Timing
print(f"\n{'='*70}")
print("Track B v2: Timing")
print(f"{'='*70}")
for _ in range(5):
    _ = d_ssim_loss(pred, gt)
    _ = fused_ssim(pred, gt)
torch.cuda.synchronize()

t_orig = time_func(lambda: d_ssim_loss(pred, gt))
t_fused = time_func(lambda: fused_ssim(pred, gt))
t_orig_50 = time_func(lambda: d_ssim_loss(pred_50, gt_50))
t_fused_50 = time_func(lambda: fused_ssim(pred_50, gt_50))

print(f"  Scale 1.0: orig={t_orig:.2f} ms  fused={t_fused:.2f} ms  speedup={t_orig/t_fused:.2f}x")
print(f"  Scale 0.5: orig={t_orig_50:.2f} ms  fused={t_fused_50:.2f} ms  speedup={t_orig_50/t_fused_50:.2f}x")

# E2E
total_10 = 98.33
total_05 = 40.27
total_fused_10 = total_10 - 75.85 + t_fused
total_fused_05 = total_05 - 21.27 + t_fused_50
total_combined = 4.57 + t_fused_50 + 14.40

print(f"\n  E2E (scale=1.0, fused):  {total_fused_10:.1f} ms → +{(total_10-total_fused_10)/total_10*100:.1f}%")
print(f"  E2E (scale=0.5, fused):  {total_fused_05:.1f} ms → +{(total_10-total_fused_05)/total_10*100:.1f}%")
print(f"  Combined (C42+fused):    {total_combined:.1f} ms → +{(total_10-total_combined)/total_10*100:.1f}%")

# Profiler
print(f"\n{'='*70}")
print("Profiler: Fused SSIM v2")
print(f"{'='*70}")
for _ in range(5): _ = fused_ssim(pred, gt)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    _ = fused_ssim(pred, gt)
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# Save
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
results = {
    "correctness": {"abs_diff": diff, "rel_diff_pct": diff/loss_orig.item()*100, "correct": diff < 1e-6},
    "timing": {"orig_10": t_orig, "fused_10": t_fused, "speedup_10": t_orig/t_fused,
               "orig_05": t_orig_50, "fused_05": t_fused_50, "speedup_05": t_orig_50/t_fused_50},
    "e2e": {"fused_10_total": total_fused_10, "fused_10_pct": (total_10-total_fused_10)/total_10*100,
            "fused_05_total": total_fused_05, "fused_05_pct": (total_10-total_fused_05)/total_10*100,
            "combined_total": total_combined, "combined_pct": (total_10-total_combined)/total_10*100}
}
save_path = repo / "results" / "a100" / "phase-c44" / "track_b_fused_ssim_v2_validation.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
