#!/usr/bin/env python3
"""Fix Ops.h forward declaration of intersect_tile to include conics/opacities."""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153"
ops_path = f"{root}/gsplat/cuda/include/Ops.h"

with open(ops_path) as f:
    content = f.read()

old = """std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [..., C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., C, N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented
);"""

new = """std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [..., C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., C, N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., C, N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., C, N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented
);"""

assert old in content, "Cannot find expected text in Ops.h"
content = content.replace(old, new, 1)
with open(ops_path, "w") as f:
    f.write(content)
print("MODIFIED Ops.h: added conics/opacities to intersect_tile declaration")
