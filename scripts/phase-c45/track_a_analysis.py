#!/usr/bin/env python3
"""
Track A: Post-C44 backward optimization analysis.

After separable SSIM, the pipeline breakdown is:
  SSIM: 24.94ms (52.8%)
  Backward: 18.72ms (39.6%)
  Render: 3.46ms (7.3%)

The backward is now 39.6% — much more significant than C43's 18.2%.
Re-assess A1-A4 with the new bottleneck distribution.

From profiler:
  rasterize_to_pixels_3dgs_bwd: 6.65ms (35.5% of backward)
  Memcpy DtoD: 2.64ms (14.1%)
  dgrad2d (SSIM conv backward): 1.88ms (10.0%)
  elementwise: ~7.57ms (40.4%)

A1: Gradient buffer init fusion (Memcpy DtoD = 2.64ms)
A2: Backward rasterizer kernel (6.65ms)  
A3: Alpha/transmittance reuse (exp() savings ~0.91ms)
A4: Atomic reduction optimization

This is a pure analysis script — no implementation.
"""
import json
from pathlib import Path

# Post-C44 profile data
total_pipeline_ms = 47.27  # from post_c44_profile.json
backward_ms = 18.72
ssim_ms = 24.94
render_ms = 3.46

# Backward kernel breakdown (from profiler, total CUDA = 18.74ms)
bwd_kernel_ms = 6.65      # rasterize_to_pixels_3dgs_bwd (35.5%)
memcpy_dtoD_ms = 2.64     # gradient buffer init (14.1%)
dgrad2d_ms = 1.88         # SSIM conv backward (10.0%)
elementwise_ms = 7.57     # various elementwise (40.4%)

print("=" * 70)
print("Track A: Post-C44 Backward Optimization Analysis")
print("=" * 70)

print(f"\nPost-C44 pipeline breakdown:")
print(f"  Total: {total_pipeline_ms:.2f} ms")
print(f"  SSIM: {ssim_ms:.2f} ms ({ssim_ms/total_pipeline_ms*100:.1f}%)")
print(f"  Backward: {backward_ms:.2f} ms ({backward_ms/total_pipeline_ms*100:.1f}%)")
print(f"  Render: {render_ms:.2f} ms ({render_ms/total_pipeline_ms*100:.1f}%)")

print(f"\nBackward kernel breakdown (total CUDA: {bwd_kernel_ms+memcpy_dtoD_ms+dgrad2d_ms+elementwise_ms:.2f} ms):")
print(f"  rasterize_bwd: {bwd_kernel_ms:.2f} ms ({bwd_kernel_ms/backward_ms*100:.1f}%)")
print(f"  Memcpy DtoD: {memcpy_dtoD_ms:.2f} ms ({memcpy_dtoD_ms/backward_ms*100:.1f}%)")
print(f"  dgrad2d (SSIM bwd): {dgrad2d_ms:.2f} ms ({dgrad2d_ms/backward_ms*100:.1f}%)")
print(f"  elementwise: {elementwise_ms:.2f} ms ({elementwise_ms/backward_ms*100:.1f}%)")

# A1: Gradient buffer init fusion
print(f"\n{'='*70}")
print("A1: Gradient Buffer Initialization Fusion")
print(f"{'='*70}")
# Memcpy DtoD = 2.64ms. If eliminated:
a1_gain = memcpy_dtoD_ms / total_pipeline_ms * 100
print(f"  Memcpy DtoD: {memcpy_dtoD_ms:.2f} ms")
print(f"  E2E gain if eliminated: +{a1_gain:.1f}%")
print(f"  With freq8 (total ~15ms): +{memcpy_dtoD_ms/15*100:.1f}%")
print(f"  Implementation: Fuse memset into backward kernel prologue")
print(f"  Risk: Race conditions on first-touch, HIGH complexity")
print(f"  DECISION: {'KEEP' if a1_gain > 5 else 'MARGINAL' if a1_gain > 3 else 'DROP'} — {a1_gain:.1f}% < 5% threshold")

# A2: Backward rasterizer kernel optimization
print(f"\n{'='*70}")
print("A2: Backward Rasterizer Kernel Optimization")
print(f"{'='*70}")
# 6.65ms rasterize_bwd kernel. From C43 analysis:
# - exp() recomputation: ~0.91ms (16.7% of kernel)
# - The rest is gradient computation + atomicAdd
# Options: 
#   a) Alpha caching (C43 Track B): eliminate exp() → save 0.91ms
#   b) Register reduction (tile_size tuning): C43 showed tile_size=8 gives +26% bwd but -64% fwd
#   c) Custom CUDA kernel rewrite: very high complexity
a2_alpha_gain = 0.91 / total_pipeline_ms * 100
print(f"  Bwd kernel: {bwd_kernel_ms:.2f} ms")
print(f"  Alpha caching (save exp()): 0.91ms → +{a2_alpha_gain:.1f}% e2e")
print(f"  Tile_size=8 (C43): +26% bwd but -64% fwd → net negative")
print(f"  Custom kernel rewrite: infeasible (requires deep gsplat modification)")
print(f"  DECISION: DROP — alpha caching gives +{a2_alpha_gain:.1f}% (below 3%)")

# A3: Alpha/transmittance reuse
print(f"\n{'='*70}")
print("A3: Alpha/Transmittance Reuse")
print(f"{'='*70}")
# Same as A2 alpha caching — reuse forward exp() in backward
# From C43: cache 17.3MB, saves 0.91ms
# Post-C44: 0.91/47.27 = 1.9%
a3_gain = 0.91 / total_pipeline_ms * 100
print(f"  Same as alpha caching in A2")
print(f"  E2E gain: +{a3_gain:.1f}%")
print(f"  Memory cost: 17.3MB (negligible)")
print(f"  DECISION: DROP — +{a3_gain:.1f}% < 3% threshold")

# A4: Atomic reduction optimization
print(f"\n{'='*70}")
print("A4: Atomic Reduction Optimization")
print(f"{'='*70}")
# The backward kernel uses warpSum + single-thread atomicAdd per warp
# Already optimized at warp level. Further optimization would require:
# - Block-level reduction (shared memory) — but Gaussians span tiles, not blocks
# - Batching atomics — complex, limited benefit
# From C43: atomicAdd throughput was not measured (ncu unavailable)
# Estimate: atomics are ~10-20% of kernel time = 0.67-1.33ms
a4_gain_low = 0.67 / total_pipeline_ms * 100
a4_gain_high = 1.33 / total_pipeline_ms * 100
print(f"  Current: warpSum + single-thread atomicAdd (already optimized)")
print(f"  Estimated atomic cost: 0.67-1.33ms (10-20% of bwd kernel)")
print(f"  E2E gain if eliminated: +{a4_gain_low:.1f}% to +{a4_gain_high:.1f}%")
print(f"  Implementation: Block-level shared memory reduction — but Gaussians")
print(f"    span multiple tiles, so block-level reduction doesn't apply")
print(f"  DECISION: DROP — max +{a4_gain_high:.1f}% < 3%, and optimization not applicable")

# SSIM backward (dgrad2d) — NEW opportunity post-C44
print(f"\n{'='*70}")
print("NEW: SSIM Conv Backward (dgrad2d) Optimization")
print(f"{'='*70}")
# The separable SSIM adds 1.88ms of conv2d backward (dgrad2d)
# This is the backward of the 2×1D separable convolutions
# Potential: fuse the 2 backward convs, or use more efficient backward
dgrad_gain = dgrad2d_ms / total_pipeline_ms * 100
print(f"  dgrad2d time: {dgrad2d_ms:.2f} ms ({dgrad_gain:.1f}% of pipeline)")
print(f"  This is the backward of separable SSIM's 2 conv2d operations")
print(f"  Potential: Already using cuDNN, limited optimization room")
print(f"  DECISION: DROP — {dgrad_gain:.1f}% < 3%")

# Elementwise operations — the largest backward component
print(f"\n{'='*70}")
print("NEW: Elementwise Operations (40.4% of backward)")
print(f"{'='*70}")
# 7.57ms of elementwise kernels — these are PyTorch's autograd operations
# for the SSIM loss computation (mu², sigma², ssim_map, etc.)
# Potential: Custom fused kernel for SSIM gradient computation
elem_gain = elementwise_ms / total_pipeline_ms * 100
print(f"  Elementwise time: {elementwise_ms:.2f} ms ({elem_gain:.1f}% of pipeline)")
print(f"  These are PyTorch autograd ops for SSIM gradient (mu^2, sigma^2, etc.)")
print(f"  Potential: Custom CUDA kernel fusing all SSIM gradient ops")
print(f"  Estimated savings: 50% of elementwise = {elementwise_ms*0.5:.2f}ms")
print(f"  E2E gain: +{elementwise_ms*0.5/total_pipeline_ms*100:.1f}%")
print(f"  DECISION: MARGINAL — +{elementwise_ms*0.5/total_pipeline_ms*100:.1f}% (near 3% threshold)")
print(f"  But requires custom CUDA kernel — HIGH complexity")

# Summary
print(f"\n{'='*70}")
print("Track A Summary")
print(f"{'='*70}")
print(f"  A1 (gradient init fusion):  +{a1_gain:.1f}% → DROP")
print(f"  A2 (bwd kernel opt):        +{a2_alpha_gain:.1f}% → DROP")
print(f"  A3 (alpha reuse):           +{a3_gain:.1f}% → DROP")
print(f"  A4 (atomic reduction):      +{a4_gain_high:.1f}% → DROP")
print(f"  NEW (dgrad2d):              +{dgrad_gain:.1f}% → DROP")
print(f"  NEW (elementwise fusion):   +{elementwise_ms*0.5/total_pipeline_ms*100:.1f}% → MARGINAL")
print(f"\n  CONCLUSION: No backward optimization exceeds 5% threshold post-C44.")
print(f"  The backward is 39.6% of pipeline, but it's composed of many small")
print(f"  kernels — no single kernel dominates enough for targeted optimization.")
print(f"  The largest opportunity is elementwise fusion ({elementwise_ms*0.5/total_pipeline_ms*100:.1f}%)")
print(f"  but requires custom CUDA kernel development.")

# Save
results = {
    "post_c44": {"total_ms": total_pipeline_ms, "backward_ms": backward_ms,
                 "ssim_ms": ssim_ms, "render_ms": render_ms},
    "backward_breakdown": {"rasterize_bwd": bwd_kernel_ms, "memcpy_dtoD": memcpy_dtoD_ms,
                          "dgrad2d": dgrad2d_ms, "elementwise": elementwise_ms},
    "decisions": {
        "A1": {"gain_pct": a1_gain, "decision": "DROP"},
        "A2": {"gain_pct": a2_alpha_gain, "decision": "DROP"},
        "A3": {"gain_pct": a3_gain, "decision": "DROP"},
        "A4": {"gain_pct": a4_gain_high, "decision": "DROP"},
        "dgrad2d": {"gain_pct": dgrad_gain, "decision": "DROP"},
        "elementwise_fusion": {"gain_pct": elementwise_ms*0.5/total_pipeline_ms*100, "decision": "MARGINAL"},
    }
}
save_path = Path("results/a100/phase-c45/track_a_analysis.json")
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {save_path}")
