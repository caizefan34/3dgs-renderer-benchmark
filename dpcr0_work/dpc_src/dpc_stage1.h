#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>

namespace gsplat
{
namespace gaussian_render_inference_scene
{
    // DPC0 Stage 1: deterministic tile-primitive partial accumulation
    // followed by a deterministic Gaussian-major segmented reduction.
    //
    // Replaces the float-atomicAdd scatter of the C0 blend backward with:
    //   (1) one partial-gradient record per (tile, Gaussian) isect, written
    //       once by a single warp (no atomicAdd), and
    //   (2) a stable radix sort by (image*N + gaussian) plus a segmented
    //       reduce that sums each Gaussian's partials in a fixed order into
    //       the flat [I*N] raster-gradient buffers.
    //
    // No floating-point atomicAdd touches any final gradient. The flat
    // buffers (v_means2d/v_conics/v_colors/v_opacities) are written in place;
    // they must be pre-zeroed by the caller. The background gradient is NOT
    // computed here (handled by the caller with the C0 background kernel).
    void dpc_stage1(
        at::Tensor means2d,        // [I*N, 2] f32
        at::Tensor conics,         // [I*N, 3] f32
        at::Tensor colors_eval,    // [I*N, CDIM] f32
        at::Tensor opacities,      // [I*N] f32
        const c10::optional<at::Tensor> &backgrounds,    // [I, CDIM] f32 (may be undefined)
        at::Tensor tile_offsets,   // [I, tile_h, tile_w] int32
        at::Tensor flatten_ids,    // [n_isects] int32
        const c10::optional<at::Tensor> &active_tiles,   // [AT] int32 or undefined
        at::Tensor render_alphas,  // [I, H, W] f32
        at::Tensor last_ids,       // [I, H, W] int32
        at::Tensor v_render_colors,// [I, H, W, CDIM] f32
        at::Tensor v_render_alphas,// [I, H, W] f32
        int64_t I,
        int64_t N,
        int64_t n_isects,
        int64_t width,
        int64_t height,
        int64_t tile_size,
        int64_t tile_h,
        int64_t tile_w,
        int64_t color_dim,
        bool scalar_adjoint,
        bool h8_mr,
        at::Tensor v_means2d,      // [I*N, 2] out (pre-zeroed)
        at::Tensor v_conics,       // [I*N, 3] out (pre-zeroed)
        at::Tensor v_colors,       // [I*N, CDIM] out (pre-zeroed)
        at::Tensor v_opacities     // [I*N] out (pre-zeroed)
    );
} // namespace gaussian_render_inference_scene
} // namespace gsplat