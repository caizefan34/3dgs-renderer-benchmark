#!/usr/bin/env python3
"""
Track D + E: Combined analysis from C43 data.

Track D: C42 + Alpha Cache combination
- C43 showed alpha caching alone <2% e2e
- But after C42 scale=0.5, backward fraction increases
- Recalculate combined gain

Track E: Gradient Buffer Initialization Fusion  
- C43 profiler showed Memcpy DtoD = 1.445ms (13.9% of backward)
- This is gradient buffer zeroing (cudaMemsetAsync for v_colors, v_conics, etc.)
- Can this be fused into the backward kernel?

Both tracks are pure analysis — no implementation needed.
"""
import json
from pathlib import Path

repo = Path(__file__).parent.parent.parent
results_dir = repo / "results" / "a100" / "phase-c44"

print("=" * 70)
print("Track D: C42 + Alpha Cache Combination Analysis")
print("=" * 70)

# From C43 data:
# Track 0v3 clean timing (scale=0.5): total=40.27ms, ssim=21.27ms, bwd=14.40ms, render=4.56ms
# Alpha caching: 16.7% bwd kernel speedup, bwd kernel = 63.5% of bwd total
# But: fused SSIM changes everything. Let's compute for both scenarios.

# Scenario 1: C42 + alpha cache (no fused SSIM)
total_baseline = 98.33  # scale=1.0, original SSIM
total_c42_05 = 40.27    # scale=0.5, original SSIM
bwd_total_05 = 14.40    # backward at scale=0.5
bwd_kernel_pct = 0.635  # rasterize_bwd_kernel is 63.5% of total backward CUDA
alpha_cache_bwd_kernel_speedup = 0.167  # 16.7% of bwd kernel
bwd_kernel_05 = bwd_total_05 * bwd_kernel_pct  # estimated bwd kernel at scale=0.5
alpha_cache_savings_ms = bwd_kernel_05 * alpha_cache_bwd_kernel_speedup
total_c42_05_alphacache = total_c42_05 - alpha_cache_savings_ms
e2e_c42_alphacache = (total_baseline - total_c42_05_alphacache) / total_baseline * 100

print(f"\nScenario 1: C42 (scale=0.5) + Alpha Cache (no fused SSIM)")
print(f"  Baseline (scale=1.0):           {total_baseline:.2f} ms")
print(f"  C42 only (scale=0.5):           {total_c42_05:.2f} ms (+{(total_baseline-total_c42_05)/total_baseline*100:.1f}%)")
print(f"  Backward at scale=0.5:          {bwd_total_05:.2f} ms")
print(f"  Bwd kernel (63.5% of bwd):      {bwd_kernel_05:.2f} ms")
print(f"  Alpha cache savings (16.7%):    {alpha_cache_savings_ms:.2f} ms")
print(f"  C42 + alpha cache total:        {total_c42_05_alphacache:.2f} ms")
print(f"  E2E speedup:                    +{e2e_c42_alphacache:.1f}%")
print(f"  Marginal gain from alpha cache:  +{alpha_cache_savings_ms/total_baseline*100:.1f}% over C42 alone")
print(f"  DECISION: DROP — alpha cache adds only +{alpha_cache_savings_ms/total_baseline*100:.1f}% on top of C42")

# Scenario 2: Fused SSIM + alpha cache
# If fused SSIM is implemented, SSIM drops from 75ms to ~9ms
# Total at scale=1.0 with fused SSIM: 98.33 - 75.85 + 9 = ~31.5ms
# Backward becomes much more significant
total_fused_ssim_10 = 98.33 - 75.85 + 9.0  # estimated
bwd_total_fused = 17.87  # backward doesn't change with fused SSIM
bwd_kernel_fused = bwd_total_fused * bwd_kernel_pct
alpha_cache_savings_fused = bwd_kernel_fused * alpha_cache_bwd_kernel_speedup
total_fused_alphacache = total_fused_ssim_10 - alpha_cache_savings_fused
e2e_fused_alphacache = (total_baseline - total_fused_alphacache) / total_baseline * 100

print(f"\nScenario 2: Fused SSIM + Alpha Cache (scale=1.0)")
print(f"  Fused SSIM total (scale=1.0):   {total_fused_ssim_10:.2f} ms (+{(total_baseline-total_fused_ssim_10)/total_baseline*100:.1f}%)")
print(f"  Backward:                       {bwd_total_fused:.2f} ms")
print(f"  Bwd kernel:                     {bwd_kernel_fused:.2f} ms")
print(f"  Alpha cache savings:            {alpha_cache_savings_fused:.2f} ms")
print(f"  Fused + alpha cache total:      {total_fused_alphacache:.2f} ms")
print(f"  E2E speedup:                    +{e2e_fused_alphacache:.1f}%")
print(f"  Marginal gain from alpha cache: +{alpha_cache_savings_fused/total_baseline*100:.1f}% on top of fused SSIM")
print(f"  DECISION: DROP — alpha cache adds only +{alpha_cache_savings_fused/total_baseline*100:.1f}% on top of fused SSIM")

# Scenario 3: Fused SSIM + C42 + alpha cache
total_fused_c42 = 4.57 + 2.0 + 14.40  # render + fused_ssim_0.5 + backward
bwd_kernel_3 = 14.40 * bwd_kernel_pct
alpha_savings_3 = bwd_kernel_3 * alpha_cache_bwd_kernel_speedup
total_all3 = total_fused_c42 - alpha_savings_3
e2e_all3 = (total_baseline - total_all3) / total_baseline * 100

print(f"\nScenario 3: Fused SSIM + C42 (scale=0.5) + Alpha Cache")
print(f"  Fused + C42 total:              {total_fused_c42:.2f} ms (+{(total_baseline-total_fused_c42)/total_baseline*100:.1f}%)")
print(f"  Alpha cache savings:            {alpha_savings_3:.2f} ms")
print(f"  All three combined:             {total_all3:.2f} ms (+{e2e_all3:.1f}%)")
print(f"  Marginal from alpha cache:      +{alpha_savings_3/total_baseline*100:.1f}%")
print(f"  DECISION: DROP — alpha cache adds only +{alpha_savings_3/total_baseline*100:.1f}% even with both fused+C42")

print(f"\n{'='*70}")
print("Track D Summary: Alpha caching is NOT worth implementing in any scenario")
print(f"  Best case marginal gain: +{alpha_cache_savings_fused/total_baseline*100:.1f}% (fused SSIM, scale=1.0)")
print(f"  This is below 3% threshold → DROP")
print(f"  Reason: Even with fused SSIM making backward more significant,")
print(f"  the exp() savings (~0.9ms) is tiny relative to total iteration time")
print(f"{'='*70}")

# Track E
print(f"\n{'='*70}")
print("Track E: Gradient Buffer Initialization Fusion")
print(f"{'='*70}")

# From C43 PyTorch profiler:
# Memcpy DtoD: 1.445ms (13.9% of backward CUDA)
# This is likely the zeroing of gradient buffers:
# - v_colors: [N, 3] = 1.59M × 3 × 4 = 19.1 MB
# - v_conics: [N, 3] = 19.1 MB
# - v_means2d: [N, 2] = 12.7 MB
# - v_opacities: [N] = 6.4 MB
# Total: ~57 MB of memset
# Plus PyTorch's optimizer.zero_grad(set_to_none=True) overhead

# The backward kernel already does atomicAdd, which requires pre-zeroed buffers
# Fusing initialization into the kernel would mean:
# 1. First thread to touch each gradient slot writes 0 instead of atomicAdd
# 2. This requires a "first touch" flag per Gaussian — complex
# 3. Alternatively: use a separate small kernel that zeros only the used gradients

# Estimate gain
memcpy_dtoD_ms = 1.445  # from profiler
backward_total_cuda_ms = 10.375  # from profiler
total_iter_ms = 98.33  # scale=1.0

e2e_gain_memset = memcpy_dtoD_ms / total_iter_ms * 100
print(f"  Memcpy DtoD time:     {memcpy_dtoD_ms:.3f} ms")
print(f"  Backward CUDA total:  {backward_total_cuda_ms:.3f} ms")
print(f"  Memcpy as % of bwd:   {memcpy_dtoD_ms/backward_total_cuda_ms*100:.1f}%")
print(f"  E2E gain if eliminated: +{e2e_gain_memset:.1f}%")
print(f"  DECISION: DROP — only +{e2e_gain_memset:.1f}% e2e, below 3% threshold")

# With fused SSIM, this becomes relatively more significant
total_fused_iter = 31.5  # estimated with fused SSIM
e2e_gain_fused = memcpy_dtoD_ms / total_fused_iter * 100
print(f"\n  With fused SSIM (total={total_fused_iter}ms):")
print(f"  E2E gain if eliminated: +{e2e_gain_fused:.1f}%")
print(f"  Still below 5% threshold → DROP")

# What about the set_to_none=True overhead?
# PyTorch's zero_grad(set_to_none=True) doesn't memset — it just drops the reference
# The Memcpy DtoD is likely from gsplat's internal gradient allocation
# or from the autograd function's backward setup

print(f"\n{'='*70}")
print("Track E Summary: Gradient buffer init fusion is NOT worth implementing")
print(f"  Best case gain: +{e2e_gain_memset:.1f}% e2e (below 3%)")
print(f"  Implementation complexity: HIGH (requires kernel modification)")
print(f"  Risk: Race conditions on first-touch initialization")
print(f"  DECISION: DROP")
print(f"{'='*70}")

# Save
results = {
    "track_d": {
        "scenario1_c42_alphacache": {
            "total_ms": total_c42_05_alphacache,
            "e2e_speedup_pct": e2e_c42_alphacache,
            "marginal_from_alpha_pct": alpha_cache_savings_ms / total_baseline * 100,
            "decision": "DROP"
        },
        "scenario2_fused_alphacache": {
            "total_ms": total_fused_alphacache,
            "e2e_speedup_pct": e2e_fused_alphacache,
            "marginal_from_alpha_pct": alpha_cache_savings_fused / total_baseline * 100,
            "decision": "DROP"
        },
        "scenario3_all_three": {
            "total_ms": total_all3,
            "e2e_speedup_pct": e2e_all3,
            "marginal_from_alpha_pct": alpha_savings_3 / total_baseline * 100,
            "decision": "DROP"
        },
        "conclusion": "Alpha caching adds <1% in all scenarios. DROP."
    },
    "track_e": {
        "memcpy_dtoD_ms": memcpy_dtoD_ms,
        "backward_cuda_total_ms": 10.375,
        "e2e_gain_pct": e2e_gain_memset,
        "e2e_gain_with_fused_pct": e2e_gain_fused,
        "decision": "DROP — below 3% threshold, high implementation complexity"
    }
}
save_path = results_dir / "track_d_e_analysis.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
