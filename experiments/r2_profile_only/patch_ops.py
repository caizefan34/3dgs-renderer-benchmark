"""Patch Ops.h to add 3 mask params to bwd function declaration."""
import os, shutil

filepath = "/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/cuda/include/Ops.h"
bak = filepath + ".orig_r21"
if not os.path.exists(bak):
    shutil.copy2(filepath, bak)
    print(f"  Backed up to {bak}")

with open(filepath) as f:
    content = f.read()

old = """    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad                 // B2 mode flag
);

// Rasterize 3D Gaussian, but only return the indices of gaussians and pixels."""

new = """    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad,                // B2 mode flag
    // R2.1: per-branch masks
    const at::optional<at::Tensor> r2_geo_mask,     // [N] or [nnz], uint8
    const at::optional<at::Tensor> r2_app_mask,     // [N] or [nnz], uint8
    const at::optional<at::Tensor> r2_opacity_mask  // [N] or [nnz], uint8
);

// Rasterize 3D Gaussian, but only return the indices of gaussians and pixels."""

if old in content:
    content = content.replace(old, new, 1)
    with open(filepath, 'w') as f:
        f.write(content)
    print("  Patched Ops.h")
else:
    print("  WARNING: pattern not found in Ops.h")
