#!/usr/bin/env python3
"""
Track B: Fused SSIM CUDA optimization analysis.

Profile the d_ssim_loss computation to understand:
1. Kernel launch count
2. Intermediate tensor memory
3. CUDA time breakdown per operation
4. Potential fusion opportunities

Do NOT implement fusion unless estimated gain >5%.
"""
import sys, torch, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")
import torch.nn.functional as F
from loss import d_ssim_loss

torch.manual_seed(42)
DEVICE = "cuda"

# Create representative tensors at 1080p
H, W = 1080, 1920
pred = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)
gt = torch.rand(H, W, 3, device=DEVICE).clamp(0, 1)

# Also test at 0.75x and 0.5x scales
H75, W75 = int(H * 0.75), int(W * 0.75)
H50, W50 = H // 2, W // 2
pred_75 = F.interpolate(pred.unsqueeze(0).permute(0,3,1,2), size=(H75, W75), mode="area").squeeze(0).permute(1,2,0)
gt_75 = F.interpolate(gt.unsqueeze(0).permute(0,3,1,2), size=(H75, W75), mode="area").squeeze(0).permute(1,2,0)
pred_50 = F.interpolate(pred.unsqueeze(0).permute(0,3,1,2), size=(H50, W50), mode="area").squeeze(0).permute(1,2,0)
gt_50 = F.interpolate(gt.unsqueeze(0).permute(0,3,1,2), size=(H50, W50), mode="area").squeeze(0).permute(1,2,0)

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

# Time SSIM at each scale
print("=" * 70)
print("Track B: SSIM loss profiling")
print("=" * 70)

for label, p, g in [("1.0", pred, gt), ("0.75", pred_75, gt_75), ("0.5", pred_50, gt_50)]:
    t = time_func(lambda: d_ssim_loss(p, g), n=100, warmup=10)
    print(f"  SSIM scale={label} ({p.shape[0]}x{p.shape[1]}): {t:.3f} ms")

# Profile with PyTorch profiler
print(f"\n{'='*70}")
print("PyTorch profiler: SSIM at scale=1.0 (1080p)")
print(f"{'='*70}")

# Warmup
for _ in range(5):
    loss = d_ssim_loss(pred, gt)
torch.cuda.synchronize()

# Profile
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    loss = d_ssim_loss(pred, gt)
torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# Count kernel launches
ka = prof.key_averages()
total_kernels = sum(1 for e in ka if e.count > 0)
total_cuda_time = sum(e.self_device_time_total for e in ka)
print(f"\nTotal CUDA kernel entries: {total_kernels}")
print(f"Total CUDA time: {total_cuda_time:.1f} us = {total_cuda_time/1000:.3f} ms")

# Analyze fusion opportunities
print(f"\n{'='*70}")
print("Fusion opportunity analysis")
print(f"{'='*70}")

# Current SSIM does:
# 1. Build Gaussian kernel (small ops)
# 2. blur(pred) = conv2d(pred, kernel)     → mu_pred
# 3. blur(target) = conv2d(target, kernel) → mu_target
# 4. mu_pred_sq = mu_pred ** 2
# 5. mu_target_sq = mu_target ** 2
# 6. mu_pred_target = mu_pred * mu_target
# 7. blur(pred**2) = conv2d(pred**2, kernel) → for sigma_pred_sq
# 8. blur(target**2) = conv2d(target**2, kernel) → for sigma_target_sq
# 9. blur(pred*target) = conv2d(pred*target, kernel) → for sigma_pred_target
# 10. sigma_pred_sq = blur(pred**2) - mu_pred_sq
# 11. sigma_target_sq = blur(target**2) - mu_target_sq
# 12. sigma_pred_target = blur(pred*target) - mu_pred_target
# 13. ssim_map = (2*mu_pred_target + C1) * (2*sigma_pred_target + C2) / ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
# 14. return 1 - ssim_map.mean()

# Fusion ideas:
# A. Precompute kernel ONCE (not every call) → saves ~5 small ops
# B. Concatenate [pred, target, pred^2, target^2, pred*target] and do ONE conv2d
#    with groups=3, 5 output channels per group → 1 conv2d instead of 5
# C. Fuse all elementwise ops into a single custom kernel

# Estimate savings from each:
# A. Kernel rebuild: ~0.1ms (negligible)
# B. 5 conv2d → 1 conv2d: saves 4 kernel launches + 4 intermediate writes
# C. ~10 elementwise → 1 kernel: saves ~9 kernel launches

# Measure individual conv2d time
kernel_1d = torch.exp(-(torch.arange(11, device=DEVICE, dtype=torch.float32) - 5) ** 2 / (2 * 1.5 ** 2))
kernel_1d = kernel_1d / kernel_1d.sum()
kernel = kernel_1d[:, None] * kernel_1d[None, :]
kernel = kernel.expand(3, 1, 11, 11).contiguous()

pred_chw = pred.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
gt_chw = gt.unsqueeze(0).permute(0, 3, 1, 2)

t_conv = time_func(lambda: F.conv2d(pred_chw, kernel, padding=5, groups=3), n=100, warmup=10)
print(f"  Single conv2d (1080p, 11x11, groups=3): {t_conv:.3f} ms")
print(f"  5 conv2d (current): {t_conv * 5:.3f} ms")

# Test fused approach: concatenate 5 inputs, 1 conv2d
pred_sq = pred_chw ** 2
gt_sq = gt_chw ** 2
pred_gt = pred_chw * gt_chw
# Stack along channel: [pred, target, pred^2, target^2, pred*target] = 15 channels
stacked = torch.cat([pred_chw, gt_chw, pred_sq, gt_sq, pred_gt], dim=1)  # [1, 15, H, W]
# Kernel: expand to 15 output channels, groups=3 (each RGB group has 5 outputs)
kernel_fused = kernel.unsqueeze(0).repeat(5, 1, 1, 1, 1).reshape(15, 1, 11, 11).contiguous()

t_fused_conv = time_func(lambda: F.conv2d(stacked, kernel_fused, padding=5, groups=15), n=100, warmup=10)
print(f"  Fused conv2d (15ch, groups=15): {t_fused_conv:.3f} ms")
print(f"  Estimated fused SSIM (1 conv + elementwise): {t_fused_conv + 0.5:.3f} ms")

# Measure elementwise overhead
t_elem = time_func(lambda: pred_chw ** 2 + gt_chw ** 2 + pred_chw * gt_chw, n=100, warmup=10)
print(f"  Elementwise (sq + mul): {t_elem:.3f} ms")

# Measure interpolate (for C42 downscale)
t_interp = time_func(lambda: F.interpolate(pred_chw, size=(H50, W50), mode="area"), n=100, warmup=10)
print(f"  Interpolate (1080p → 540p): {t_interp:.3f} ms")

# Estimate memory
pred_bytes = pred_chw.numel() * 4
gt_bytes = gt_chw.numel() * 4
# 5 intermediate conv outputs: 5 × [1, 3, H, W] × 4 bytes
conv_out_bytes = 5 * pred_bytes
# Pre-fusion elementwise: pred^2, gt^2, pred*gt = 3 × [1, 3, H, W]
elem_bytes = 3 * pred_bytes
# Stacked tensor: 15 × [1, H, W] × 4 bytes
stacked_bytes = 5 * pred_bytes  # same as 5 × [1, 3, H, W]
print(f"\n  Memory per SSIM call:")
print(f"    Input tensors: {(pred_bytes + gt_bytes) / 1e6:.1f} MB")
print(f"    Conv intermediates (5): {conv_out_bytes / 1e6:.1f} MB")
print(f"    Elementwise intermediates: {elem_bytes / 1e6:.1f} MB")
print(f"    Total intermediates: {(conv_out_bytes + elem_bytes) / 1e6:.1f} MB")

# Summary
print(f"\n{'='*70}")
print("Fusion gain estimate")
print(f"{'='*70}")
t_ssim_full = time_func(lambda: d_ssim_loss(pred, gt), n=100, warmup=10)
t_ssim_50 = time_func(lambda: d_ssim_loss(pred_50, gt_50), n=100, warmup=10)

# Fused estimate: 1 conv2d (fused) + elementwise + interpolate
t_fused_estimate = t_fused_conv + t_elem + t_interp + 0.5  # +0.5 for SSIM map computation
fused_gain = (t_ssim_full - t_fused_estimate) / t_ssim_full * 100

# E2E estimate (from C43: total iter at scale=1.0 = 98.33 ms)
total_iter_10 = 98.33
total_iter_05 = 40.27
e2e_gain_10 = (t_ssim_full - t_fused_estimate) / total_iter_10 * 100
e2e_gain_05 = (t_ssim_full - t_fused_estimate) / total_iter_05 * 100  # if fused at scale=0.5

print(f"  Current SSIM (1.0): {t_ssim_full:.2f} ms")
print(f"  Current SSIM (0.5): {t_ssim_50:.2f} ms")
print(f"  Fused SSIM estimate: {t_fused_estimate:.2f} ms")
print(f"  SSIM speedup: {fused_gain:.1f}%")
print(f"  E2E speedup (scale=1.0, total=98ms): {e2e_gain_10:.1f}%")
print(f"  E2E speedup (scale=0.5, total=40ms): {e2e_gain_05:.1f}%")
print(f"  Note: Fused SSIM at scale=0.5 would be even faster")

# Decision
if e2e_gain_10 > 5:
    print(f"\n  DECISION: Implement fusion (E2E gain {e2e_gain_10:.1f}% > 5%)")
elif e2e_gain_10 > 3:
    print(f"\n  DECISION: MARGINAL (E2E gain {e2e_gain_10:.1f}%, between 3-5%)")
else:
    print(f"\n  DECISION: DROP (E2E gain {e2e_gain_10:.1f}% < 3%)")

# Save
repo = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
results = {
    "ssim_time_10": t_ssim_full,
    "ssim_time_05": t_ssim_50,
    "single_conv2d": t_conv,
    "fused_conv2d": t_fused_conv,
    "elementwise": t_elem,
    "interpolate": t_interp,
    "fused_estimate": t_fused_estimate,
    "ssim_speedup_pct": fused_gain,
    "e2e_gain_10_pct": e2e_gain_10,
    "e2e_gain_05_pct": e2e_gain_05,
    "memory_intermediates_MB": (conv_out_bytes + elem_bytes) / 1e6,
}
save_path = repo / "results" / "a100" / "phase-c44" / "track_b_ssim_profile.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
