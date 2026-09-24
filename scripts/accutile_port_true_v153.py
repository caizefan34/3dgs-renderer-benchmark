#!/usr/bin/env python3
"""Port the TRUE upstream AccuTile (strip-based SnugBox + ellipse intersection)
from gsplat commit 28e794ca onto frozen gsplat v1.5.3.

This is variant A — the actual Speedy-Splat / upstream gsplat algorithm.
It replaces the per-tile predicate (variant P) with:
  1. SnugBox: tight ellipse bounding box
  2. Shorter-side selection
  3. Strip-based processing with ellipse intersection reuse
  4. Contiguous tile-range emission

The AABB fallback path is preserved for when conics/opacities are not provided.

Usage:
    python3 accutile_port_true_v153.py /path/to/gsplat-checkout
"""

import os
import sys


def modify_file(path, old, new):
    with open(path, "r") as f:
        content = f.read()
    if new in content and old not in content:
        print(f"  SKIP (already modified): {path}")
        return
    if old not in content:
        raise RuntimeError(f"Cannot find expected text in {path}")
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"  MODIFIED: {path}")


def port(root):
    cuda_csrc = os.path.join(root, "gsplat", "cuda", "csrc")
    cuda_dir = os.path.join(root, "gsplat", "cuda")
    include_dir = os.path.join(root, "gsplat", "cuda", "include")

    # ── 1. Add GAUSSIAN_EXTEND to Common.h ──
    common_h = os.path.join(include_dir, "Common.h")
    with open(common_h) as f:
        content = f.read()
    if "GAUSSIAN_EXTEND" not in content:
        # Add after ALPHA_THRESHOLD
        if "#define ALPHA_THRESHOLD" in content:
            content = content.replace(
                "#define ALPHA_THRESHOLD",
                "#define GAUSSIAN_EXTEND 3.33f\n#define ALPHA_THRESHOLD",
                1
            )
            with open(common_h, "w") as f:
                f.write(content)
            print(f"  MODIFIED Common.h: added GAUSSIAN_EXTEND")
        else:
            print(f"  WARNING: Could not find ALPHA_THRESHOLD in Common.h")
    else:
        print(f"  SKIP Common.h (GAUSSIAN_EXTEND already present)")

    # ── 2. Rewrite IntersectTile.cu with true AccuTile ──
    isect_cu = os.path.join(cuda_csrc, "IntersectTile.cu")
    with open(isect_cu) as f:
        original = f.read()

    # We need to replace the entire intersect_tile_kernel function and add helpers.
    # Strategy: add helpers after namespace cg, then modify the kernel.

    # 2a. Add true AccuTile helpers after namespace cg
    helpers = """
// ============================================================
// TRUE AccuTile: SnugBox + strip-based ellipse intersection
// Ported from upstream gsplat commit 28e794ca
// Reference: Speedy-Splat https://arxiv.org/pdf/2412.00578
// ============================================================

inline __device__ float2 accutile_ellipse_intersection(
    float A, float B, float C, float disc, float t, float2 p, bool isY, float coord)
{
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? A : C;

    float h = coord - p_u;
    float sqrt_term = sqrtf(disc * h * h + t * coeff);

    return {(-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v};
}

inline __device__ uint32_t accutile_process_tiles(
    float A, float B, float C, float disc, float t, float2 p,
    float2 bbox_min, float2 bbox_max,
    float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    uint32_t tile_size, uint32_t tile_width, uint32_t n_tiles,
    bool isY,
    int64_t iid_enc, uint32_t tile_n_bits, int64_t depth_id_enc,
    uint32_t flatten_idx,
    int64_t *isect_ids, int32_t *flatten_ids, int64_t *cur_idx)
{
    float BLOCK = (float)tile_size;

    if (isY) {
        rect_min = {rect_min.y, rect_min.x};
        rect_max = {rect_max.y, rect_max.x};
        bbox_min = {bbox_min.y, bbox_min.x};
        bbox_max = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK;
    if (bbox_min.x <= min_line) {
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    } else {
        intersect_min_line = intersect_max_line;
    }

    #pragma unroll 1
    for (int u = rect_min.x; u < rect_max.x; ++u) {
        max_line = min_line + BLOCK;
        if (max_line <= bbox_max.x) {
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
        }

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)));
        int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1)));

        #pragma unroll 1
        for (int v = min_tile_v; v < max_tile_v; v++) {
            int64_t tile_id = isY ? (int64_t)(u * tile_width + v) : (int64_t)(v * tile_width + u);
            ++tiles_count;

            if (isect_ids != nullptr) {
                isect_ids[*cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[*cur_idx] = static_cast<int32_t>(flatten_idx);
                ++(*cur_idx);
            }
        }

        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }
    return tiles_count;
}
"""
    modify_file(
        isect_cu,
        "namespace cg = cooperative_groups;\n",
        "namespace cg = cooperative_groups;\n" + helpers,
    )

    # 2b. Add conics/opacities to kernel signature
    modify_file(
        isect_cu,
        """    const scalar_t *__restrict__ depths,             // [..., N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [..., N] or [nnz]
    const uint32_t tile_size,""",
        """    const scalar_t *__restrict__ depths,             // [..., N] or [nnz]
    const float *__restrict__ conics,               // [..., N, 3] or [nnz, 3], optional
    const float *__restrict__ opacities,            // [..., N] or [nnz], optional
    const int64_t *__restrict__ cum_tiles_per_gauss, // [..., N] or [nnz]
    const uint32_t tile_size,""",
    )

    # 2c. Replace the AABB tile computation + first_pass/second_pass with true AccuTile path
    # This is the big change: after radius check, add the AccuTile branch
    old_aabb = """    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_radius_x = radius_x / static_cast<float>(tile_size);
    float tile_radius_y = radius_y / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t iid; // image id
    if (packed) {
        // parallelize over nnz
        iid = image_ids[idx];
    } else {
        // parallelize over I * N
        iid = idx / N;
    }
    const int64_t iid_enc = iid << (32 + tile_n_bits);

    // tolerance for negative depth
    int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit
    // int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // image id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [I * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}"""

    new_accutile = """    float2 mean2d = {(float)means2d[2 * idx], (float)means2d[2 * idx + 1]};

    int64_t iid;
    if (packed) {
        iid = image_ids[idx];
    } else {
        iid = idx / N;
    }
    int64_t iid_enc = 0;
    int64_t depth_id_enc = 0;
    if (!first_pass) {
        iid_enc = iid << (32 + tile_n_bits);
        float depth_f = static_cast<float>(depths[idx]);
        depth_id_enc = __float_as_uint(depth_f);
    }

    if (conics != nullptr && opacities != nullptr) {
        // TRUE AccuTile: SnugBox + strip-based ellipse intersection
        const float A = conics[idx * 3];
        const float B = conics[idx * 3 + 1];
        const float C = conics[idx * 3 + 2];
        float disc = B * B - A * C;

        const float opacity = opacities[idx];
        float t = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity / ALPHA_THRESHOLD));

        // Only use AccuTile if inputs are valid
        if (isfinite(A) && isfinite(B) && isfinite(C) && isfinite(t) &&
            A > 0.f && C > 0.f && disc < 0.f && t > 0.f) {

            // SNUGBOX: tight axis-aligned bounding box of the ellipse
            float neg_t_over_disc = -t / disc;
            float x_extent = sqrtf(neg_t_over_disc * C);
            float y_extent = sqrtf(neg_t_over_disc * A);

            float2 bbox_min = {mean2d.x - x_extent, mean2d.y - y_extent};
            float2 bbox_max = {mean2d.x + x_extent, mean2d.y + y_extent};

            float Bx_over_C = B * x_extent / C;
            float By_over_A = B * y_extent / A;
            float2 bbox_argmin = {mean2d.y + Bx_over_C, mean2d.x + By_over_A};
            float2 bbox_argmax = {mean2d.y - Bx_over_C, mean2d.x - By_over_A};

            float tile_size_f = (float)tile_size;
            int2 rect_min = {max(0, min((int)tile_width, (int)(bbox_min.x / tile_size_f))),
                             max(0, min((int)tile_height, (int)(bbox_min.y / tile_size_f)))};
            int2 rect_max = {max(0, min((int)tile_width, (int)(bbox_max.x / tile_size_f + 1.f))),
                             max(0, min((int)tile_height, (int)(bbox_max.y / tile_size_f + 1.f)))};

            int y_span = rect_max.y - rect_min.y;
            int x_span = rect_max.x - rect_min.x;
            if (y_span * x_span == 0) {
                if (first_pass) {
                    tiles_per_gauss[idx] = 0;
                }
                return;
            }

            bool isY = y_span < x_span;
            int64_t cur_idx = first_pass ? 0 : ((idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1]);

            uint32_t count = accutile_process_tiles(
                A, B, C, disc, t, mean2d,
                bbox_min, bbox_max, bbox_argmin, bbox_argmax,
                rect_min, rect_max,
                tile_size, tile_width, tile_width * tile_height,
                isY, iid_enc, tile_n_bits, depth_id_enc, idx,
                first_pass ? nullptr : isect_ids,
                first_pass ? nullptr : flatten_ids,
                &cur_idx);

            if (first_pass) {
                tiles_per_gauss[idx] = static_cast<int32_t>(count);
            }
            return;
        }
    }

    // AABB fallback: used when conics/opacities are not available or invalid
    {
        float tile_radius_x = radius_x / static_cast<float>(tile_size);
        float tile_radius_y = radius_y / static_cast<float>(tile_size);
        float tile_x = mean2d.x / static_cast<float>(tile_size);
        float tile_y = mean2d.y / static_cast<float>(tile_size);

        uint2 tile_min, tile_max;
        tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
        tile_min.y = min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
        tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
        tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

        if (first_pass) {
            tiles_per_gauss[idx] = static_cast<int32_t>(
                (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x));
            return;
        }

        int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
        for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
            for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                int64_t tile_id = i * tile_width + j;
                isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
            }
        }
    }
}"""

    modify_file(isect_cu, old_aabb, new_accutile)

    # 2d. Add conics/opacities to launch_intersect_tile_kernel signature
    modify_file(
        isect_cu,
        """    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {""",
        """    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {""",
    )

    # 2e. Add conics/opacities pointer extraction
    modify_file(
        isect_cu,
        """    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;""",
        """    bool packed = means2d.dim() == 2;

    const float *conics_ptr = nullptr;
    const float *opacities_ptr = nullptr;
    if (conics.has_value()) {
        conics_ptr = conics.value().data_ptr<float>();
        opacities_ptr = opacities.value().data_ptr<float>();
    }

    uint32_t N, nnz;""",
    )

    # 2f. Add conics_ptr/opacities_ptr to kernel launch calls
    modify_file(
        isect_cu,
        """                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,""",
        """                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics_ptr,
                    opacities_ptr,
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,""",
    )

    # ── 3. Intersect.h ──
    intersect_h = os.path.join(cuda_csrc, "Intersect.h")
    modify_file(
        intersect_h,
        """    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,""",
        """    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,""",
    )

    # ── 4. Intersect.cpp ──
    intersect_cpp = os.path.join(cuda_csrc, "Intersect.cpp")

    modify_file(
        intersect_cpp,
        """    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented
) {""",
        """    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented
) {""",
    )

    modify_file(
        intersect_cpp,
        """    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);

    auto opt = depths.options();""",
        """    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    TORCH_CHECK(
        conics.has_value() == opacities.has_value(),
        "AccuTile requires conics and opacities together."
    );
    if (conics.has_value()) {
        CHECK_INPUT(conics.value());
        CHECK_INPUT(opacities.value());
        TORCH_CHECK(
            conics.value().scalar_type() == at::kFloat &&
                opacities.value().scalar_type() == at::kFloat,
            "AccuTile conics and opacities must be float32."
        );
    }

    auto opt = depths.options();""",
    )

    # Add conics/opacities to first launch call
    modify_file(
        intersect_cpp,
        """        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);""",
        """        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            conics,
            opacities,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);""",
    )

    # Add conics/opacities to second launch call
    modify_file(
        intersect_cpp,
        """        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            cum_tiles_per_gauss,
            // outputs
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids)
        );""",
        """        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            conics,
            opacities,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            cum_tiles_per_gauss,
            // outputs
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids)
        );""",
    )

    # ── 5. Ops.h ──
    ops_h = os.path.join(include_dir, "Ops.h")
    modify_file(
        ops_h,
        """std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
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
);""",
        """std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
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
);""",
    )

    # ── 6. _wrapper.py ──
    wrapper_py = os.path.join(cuda_dir, "_wrapper.py")
    modify_file(
        wrapper_py,
        """    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:""",
        """    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
    conics: Optional[Tensor] = None,
    opacities: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:""",
    )

    modify_file(
        wrapper_py,
        """    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("intersect_tile")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        image_ids,
        gaussian_ids,
        I,
        tile_size,
        tile_width,
        tile_height,
        sort,
        segmented,
    )""",
        """    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("intersect_tile")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        conics.contiguous() if conics is not None else None,
        opacities.contiguous() if opacities is not None else None,
        image_ids,
        gaussian_ids,
        I,
        tile_size,
        tile_width,
        tile_height,
        sort,
        segmented,
    )""",
    )

    # ── 7. rendering.py ──
    rendering_py = os.path.join(root, "gsplat", "rendering.py")
    modify_file(
        rendering_py,
        """    covars: Optional[Tensor] = None,
    with_ut: bool = False,""",
        """    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    accutile: bool = False,""",
    )

    modify_file(
        rendering_py,
        """    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )""",
        """    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
        conics=conics if accutile else None,
        opacities=opacities if accutile else None,
    )""",
    )

    # ── 8. setup.py: add --extended-lambda ──
    setup_py = os.path.join(root, "setup.py")
    with open(setup_py) as f:
        content = f.read()
    old = 'nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]'
    if old in content and "--extended-lambda" not in content:
        content = content.replace(old, old + ', "--extended-lambda"', 1)
        with open(setup_py, "w") as f:
            f.write(content)
        print(f"  MODIFIED setup.py: added --extended-lambda")

    # ── 9. _backend.py: add --extended-lambda for JIT ──
    backend_py = os.path.join(cuda_dir, "_backend.py")
    with open(backend_py) as f:
        content = f.read()
    old = 'extra_cuda_cflags += ["-use_fast_math"]'
    if old in content and "--extended-lambda" not in content:
        content = content.replace(old, old + ', "--extended-lambda"', 1)
        with open(backend_py, "w") as f:
            f.write(content)
        print(f"  MODIFIED _backend.py: added --extended-lambda")

    print("\nAll files modified successfully for true AccuTile (variant A).")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
    port(root)
