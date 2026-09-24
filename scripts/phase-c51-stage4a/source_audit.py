"""Phase C51 Stage 4A — Source Audit

Re-verified the exact current CUDA code for the backward rasterization path.
This script documents the data flow and identifies the mask injection points.

Data Flow:
===========

1. Python: rasterization() in rendering.py
   → calls fully_fused_projection() → gets means2d, conics, radii, depths
   → calls isect_tiles() → gets tile_offsets, flatten_ids
   → calls rasterize_to_pixels() → gets render_colors, render_alphas

2. Autograd: _RasterizeToPixels in _wrapper.py
   forward(): calls CUDA "rasterize_to_pixels_3dgs_fwd"
              saves: means2d, conics, colors, opacities, backgrounds, masks,
                     isect_offsets, flatten_ids, render_alphas, last_ids
   backward(): calls CUDA "rasterize_to_pixels_3dgs_bwd"
               produces: v_means2d, v_conics, v_colors, v_opacities, v_means2d_abs

3. Autograd chain: _FullyFusedProjection.backward() in _wrapper.py
   Takes v_means2d, v_conics → produces v_means (xyz.grad), v_quats, v_scales
   This is where xyz.grad is computed from v_means2d.

4. Densification: GaussianModel.accumulate_positional_gradient()
   Reads self.xyz.grad → grad.norm(dim=-1) → accumulates
   GaussianModel.densification() uses accumulated avg vs threshold

Key Insight:
============
The backward kernel computes v_means2d (2D gradient). This flows through
projection backward to become xyz.grad (3D gradient). Densification uses
xyz.grad.norm(dim=-1).

If we skip gradient computation for masked Gaussians in the backward kernel,
their v_means2d = 0, so xyz.grad = 0, and densification won't see them.

Design B1: Skip all gradient compute for masked. Use previous-iteration
           stored gradient norm for densification.
Design B2: For masked Gaussians, still compute v_means2d (needed for
           densification through projection backward), but skip v_colors,
           v_conics, v_opacities computation and their atomicAdds.
Design B3: Same kernel as B1. Ablation for delayed densification.

Mask Injection Points:
======================
1. CUDA kernel (RasterizeToPixels3DGSBwd.cu):
   - Add importance_mask parameter (uint8_t*, [N] for non-packed)
   - Load mask into shared memory during batch loading (line 141 area)
   - After alpha computation (line 193), check mask:
     - mask=1: full gradient computation (existing code)
     - mask=0 + compute_densify_grad=true (B2): compute only v_xy_local
     - mask=0 + compute_densify_grad=false (B1/B3): skip all gradient
   - Always update T and buffer (for forward reconstruction correctness)
   - Skip warpSum + atomicAdd for masked Gaussians (B1/B3)
   - Skip partial warpSum + atomicAdd for B2 (only v_means2d)

2. C++ binding (Rasterization.cpp, Rasterization.h):
   - Add importance_mask and compute_densify_grad to rasterize_to_pixels_3dgs_bwd()

3. Python wrapper (_wrapper.py):
   - Add importance_mask to rasterize_to_pixels()
   - Save in _RasterizeToPixels.forward() via ctx
   - Pass to CUDA in _RasterizeToPixels.backward()

4. High-level API (rendering.py):
   - Add importance_mask to rasterization()
   - Pass to rasterize_to_pixels()

Correctness Invariants:
=======================
- Forward T/buffer reconstruction MUST be exact (mask doesn't affect T/buffer)
- Selected Gaussian gradients MUST be exactly correct (cosine ≈ 1)
- Skipped Gaussian optimizer gradients MUST be exactly zero (B1/B3)
  or only v_means2d computed (B2)
- Densification behavior must be explicitly measured

Files to modify on remote:
===========================
1. /tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu
2. /tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/csrc/Rasterization.h
3. /tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/csrc/Rasterization.cpp
4. /tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/_wrapper.py
5. /tmp/gsplat_baseline/gsplat-1.5.3/gsplat/rendering.py
"""

import json
from pathlib import Path

audit = {
    "phase": "C51-Stage4A",
    "component": "Source Audit",
    "status": "completed",
    "data_flow": {
        "python_api": "rasterization() in rendering.py",
        "autograd_forward": "_RasterizeToPixels.forward() calls CUDA rasterize_to_pixels_3dgs_fwd",
        "autograd_backward": "_RasterizeToPixels.backward() calls CUDA rasterize_to_pixels_3dgs_bwd",
        "projection_backward": "_FullyFusedProjection.backward() transforms v_means2d → v_means (xyz.grad)",
        "densification": "accumulate_positional_gradient() reads xyz.grad.norm(dim=-1)"
    },
    "key_insight": "Skipping gradient computation for masked Gaussians in backward kernel → v_means2d=0 → xyz.grad=0 → densification blind to masked Gaussians",
    "designs": {
        "B1": {
            "kernel_behavior": "Skip ALL gradient compute for masked. T/buffer still updated.",
            "densification": "Use previous-iteration stored gradient norm",
            "skipped_work": "v_rgb, v_conic, v_xy, v_opacity computation + warpSum + atomicAdd",
            "correctness": "Skipped Gaussians: optimizer grad = 0. Densification uses stale gradient."
        },
        "B2": {
            "kernel_behavior": "For masked: compute v_xy_local only (for densification), skip v_rgb/v_conic/v_opacity",
            "densification": "Current-iteration gradient norm (through projection backward)",
            "skipped_work": "v_rgb (3 atomicAdd), v_conic (3 atomicAdd), v_opacity (1 atomicAdd) + their warpSum + arithmetic",
            "kept_work": "v_xy (2 atomicAdd) + its warpSum + v_alpha/v_sigma arithmetic",
            "correctness": "Skipped Gaussians: xyz.grad correct, other grads = 0. Densification uses current gradient."
        },
        "B3": {
            "kernel_behavior": "Same as B1 at kernel level",
            "densification": "Delayed: use previous-iteration gradient norm (ablation)",
            "correctness": "Same as B1. Tests whether delayed densification is acceptable."
        }
    },
    "mask_injection_points": [
        "CUDA kernel: load mask in batch loading, check after alpha, skip gradient compute + atomicAdd",
        "C++ binding: pass importance_mask + compute_densify_grad through",
        "Python wrapper: save mask in autograd ctx, pass in backward",
        "High-level API: pass mask through rasterization() → rasterize_to_pixels()"
    ],
    "correctness_invariants": [
        "Forward T/buffer MUST be exact (mask does not affect T/buffer updates)",
        "Selected Gaussian gradients MUST be exactly correct (cosine ≈ 1.0)",
        "Skipped Gaussian optimizer gradients MUST be exactly zero (B1/B3) or only v_means2d (B2)",
        "Densification behavior must be explicitly measured (clone/split/prune agreement)"
    ],
    "mask_representation": "uint8_t per Gaussian (1 byte, compact). 1=compute, 0=skip.",
    "earliest_safe_branch": "After alpha computation (line 177), before gradient computation (line 193). T and buffer updates must always execute.",
    "warp_divergence": "No divergence: mask is per-Gaussian, all threads in warp process same Gaussian t. Branch is uniform across warp.",
    "files_to_modify": [
        "gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu",
        "gsplat/cuda/csrc/Rasterization.h",
        "gsplat/cuda/csrc/Rasterization.cpp",
        "gsplat/cuda/_wrapper.py",
        "gsplat/rendering.py"
    ]
}

out_path = Path(__file__).parent.parent.parent / "results" / "a100" / "phase-c51-stage4a" / "source_audit.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(audit, indent=2))
print(f"Source audit saved to {out_path}")
