#!/usr/bin/env python3
"""Port the AccuTile conservative tile-culling algorithm from the Codex
gsplat-1.4.0 patch to the gsplat v1.5.3 API used on the A100.

This is NOT a reimplementation of AccuTile — it applies the exact same
accutile_tile_may_contribute predicate from the Codex patch, wired into
the renamed/refactored v1.5.3 kernel API (IntersectTile.cu instead of
isect_tiles.cu, intersect_tile_kernel instead of isect_tiles, etc.).

Usage:
    python3 accutile_port_v153.py /path/to/gsplat-checkout
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

    # ── 1. IntersectTile.cu: add predicate + conics/opacities to kernel ──

    isect_cu = os.path.join(cuda_csrc, "IntersectTile.cu")

    # 1a. Add the accutile_tile_may_contribute predicate after namespace cg
    modify_file(
        isect_cu,
        "namespace cg = cooperative_groups;\n",
        """namespace cg = cooperative_groups;

// AccuTile's conservative predicate (ported from gsplat-1.4.0-accutile.patch).
// It uses the same opacity-thresholded conic as upstream gsplat, but evaluates
// the exact continuous minimum of q over each candidate tile.  The small
// positive margin only retains boundary tiles, so it cannot remove a tile
// containing a rasterizer pixel center.
inline __device__ bool accutile_tile_may_contribute(
    const float A,
    const float B,
    const float C,
    const float mean_x,
    const float mean_y,
    const float threshold,
    const float x0,
    const float y0,
    const float x1,
    const float y1
) {
    const float dx0 = x0 - mean_x;
    const float dx1 = x1 - mean_x;
    const float dy0 = y0 - mean_y;
    const float dy1 = y1 - mean_y;
    const auto q = [=] __device__ (const float dx, const float dy) {
        return A * dx * dx + 2.f * B * dx * dy + C * dy * dy;
    };
    float qmin = fminf(fminf(q(dx0, dy0), q(dx0, dy1)),
                        fminf(q(dx1, dy0), q(dx1, dy1)));
    const float dy_at_x0 = fminf(dy1, fmaxf(dy0, -B * dx0 / C));
    const float dy_at_x1 = fminf(dy1, fmaxf(dy0, -B * dx1 / C));
    const float dx_at_y0 = fminf(dx1, fmaxf(dx0, -B * dy0 / A));
    const float dx_at_y1 = fminf(dx1, fmaxf(dx0, -B * dy1 / A));
    qmin = fminf(qmin, fminf(q(dx0, dy_at_x0), q(dx1, dy_at_x1)));
    qmin = fminf(qmin, fminf(q(dx_at_y0, dy0), q(dx_at_y1, dy1)));
    if (dx0 <= 0.f && dx1 >= 0.f && dy0 <= 0.f && dy1 >= 0.f) {
        qmin = 0.f;
    }
    return qmin <= threshold + 1e-5f * fmaxf(1.f, threshold);
}
""",
    )

    # 1b. Add conics/opacities parameters to intersect_tile_kernel signature
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

    # 1c. Add AccuTile setup after AABB computation, before first_pass check
    modify_file(
        isect_cu,
        """    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    if (first_pass) {""",
        """    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    // radii still defines the candidate AABB exactly as the frozen path.
    // Passing both tensors enables conservative conic-vs-tile rejection.
    bool use_accutile = conics != nullptr && opacities != nullptr;
    float A = 0.f, B = 0.f, Cc = 0.f, threshold = 0.f;
    if (use_accutile) {
        A = conics[idx * 3];
        B = conics[idx * 3 + 1];
        Cc = conics[idx * 3 + 2];
        const float opacity = opacities[idx];
        threshold = fminf(3.33f * 3.33f, 2.f * logf(opacity * 255.f));
        // Invalid or sub-threshold inputs retain the original AABB path.
        use_accutile = isfinite(A) && isfinite(B) && isfinite(Cc) &&
                       isfinite(threshold) && A > 0.f && Cc > 0.f &&
                       A * Cc - B * B > 0.f && threshold > 0.f;
    }

    if (first_pass) {""",
    )

    # 1d. Add AccuTile counting in first_pass
    modify_file(
        isect_cu,
        """    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }""",
        """    if (first_pass) {
        if (use_accutile) {
            int32_t count = 0;
            for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
                for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                    if (accutile_tile_may_contribute(
                            A, B, Cc, mean2d.x, mean2d.y, threshold,
                            j * tile_size, i * tile_size,
                            (j + 1) * tile_size, (i + 1) * tile_size)) {
                        ++count;
                    }
                }
            }
            tiles_per_gauss[idx] = count;
            return;
        }
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }""",
    )

    # 1e. Add AccuTile tile skipping in second_pass
    modify_file(
        isect_cu,
        """    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;""",
        """    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            if (use_accutile && !accutile_tile_may_contribute(
                    A, B, Cc, mean2d.x, mean2d.y, threshold,
                    j * tile_size, i * tile_size,
                    (j + 1) * tile_size, (i + 1) * tile_size)) {
                continue;
            }
            int64_t tile_id = i * tile_width + j;""",
    )

    # 1f. Add conics/opacities to launch_intersect_tile_kernel signature
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

    # 1g. Add conics/opacities pointer extraction and pass to kernel
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

    # 1h. Add conics_ptr/opacities_ptr to both kernel launch calls
    # First launch (first pass) - has "cum_tiles_per_gauss.has_value()" in args
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

    # ── 2. Intersect.h: add conics/opacities to declaration ──

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

    # ── 3. Intersect.cpp: add conics/opacities to intersect_tile function ──

    intersect_cpp = os.path.join(cuda_csrc, "Intersect.cpp")

    # 3a. Add conics/opacities to function signature
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

    # 3b. Add input validation after CHECK_INPUT(depths)
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

    # 3c. Add conics/opacities to first launch_intersect_tile_kernel call
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

    # 3d. Add conics/opacities to second launch_intersect_tile_kernel call
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

    # ── 4. _wrapper.py: add conics/opacities to isect_tiles ──

    wrapper_py = os.path.join(cuda_dir, "_wrapper.py")

    # 4a. Add conics/opacities to function signature
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

    # 4b. Add conics/opacities to _make_lazy_cuda_func call
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

    # ── 5. rendering.py: add accutile parameter ──

    rendering_py = os.path.join(root, "gsplat", "rendering.py")

    # 5a. Add accutile to rasterization signature
    modify_file(
        rendering_py,
        """    covars: Optional[Tensor] = None,
    with_ut: bool = False,""",
        """    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    accutile: bool = False,""",
    )

    # 5b. Add conics/opacities to isect_tiles call
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

    print("\nAll files modified successfully.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153"
    port(root)
