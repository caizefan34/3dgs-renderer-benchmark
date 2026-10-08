#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <vector>

namespace gsplat
{
namespace gaussian_render_inference_scene
{
    // DPGR0 Stage 1: Gaussian-major DIRECT primitive gather.
    //
    // Replaces DPC0's tile-major global per-(tile,Gaussian) partial buffer +
    // stable radix sort + segmented reduce with a Gaussian-major organization:
    //   (1) range-build: a stable radix sort by key = image_id*N + gaussian of the
    //       isects yields contiguous Gaussian-major segments. Only int32 keys and
    //       int32 original-isect indices are sorted (~8 bytes/isect), NOT the
    //       9-float partial records DPC0 materializes.
    //   (2) direct-gather: one warp (32 threads, 16x2, 8 px/thread) owns one
    //       segment (one output slot). For each isect of the segment, in ascending
    //       tileID order (== DPC0's ascending-isect reduce order), it REPLAYS that
    //       tile's reverse blend scan from the tile tail down to the isect's
    //       position, carrying per-pixel T/buffer exactly as DPC0, and captures the
    //       warp-reduced 9-float raster gradient at the isect. For m<=32 (DIRECT)
    //       the partials accumulate on-chip and ONE final record is written; for
    //       m>32 (CHUNKED) the tiles are split into fixed contiguous chunks of 32,
    //       each emitting one 9-float chunk partial to a bounded buffer, reduced
    //       later in ascending chunk order.
    //
    // No float atomicAdd touches any final gradient. Every per-isect partial is
    // recomputed with DPC0's exact blend_bwd + warpSum + ascending summation, so
    // DIRECT paths are bit-identical to DPC0 on the same GPU and the fixed
    // reduction order makes the whole kernel deterministic and cross-A100
    // bitwise-equal. The tile-replay costs ~sum(D_t^2/2) pixel-ops (compute
    // bound); this is the architecture DPGR0 is designed to test.

    // Timed-build introspection (last dpgr0_stage1 call on this thread's stream).
    std::vector<double> dpgr0_get_stage_times(); // [range_build_ms, gather_ms, reduce_ms]
    std::vector<int64_t> dpgr0_get_stats();      // [n_isects, n_seg, n_chunked, total_chunks, pstride, N, records, record_bytes]

    void dpgr0_stage1(
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