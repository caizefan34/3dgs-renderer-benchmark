// SPDX-License-Identifier: Apache-2.0
//
// DPGR0 Stage 1: Gaussian-major DIRECT primitive gather.
//
// Replaces DPC0's tile-major global per-(tile,Gaussian) partial buffer +
// stable radix sort + segmented reduce with a Gaussian-major organization:
//
//   (1) range-build: a stable radix sort by key = image_id*N + gaussian of the
//       isects yields contiguous Gaussian-major segments. Only int32 keys and
//       int32 original-isect indices are sorted (~8 bytes/isect), NOT the
//       9-float partial records DPC0 materializes.
//
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

#include "Config.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>

#include <cstdint>
#include <cstdlib>

#include "Common.h"
#include "RasterizeToPixels3DGSDevice.cuh"
#include "Utils.cuh"
#include "dpgr0_stage1.h"

namespace gsplat
{
namespace gaussian_render_inference_scene
{
namespace cg = cooperative_groups;

namespace
{

constexpr uint32_t DPG_PX = 8;   // pixels per thread -> 32 threads (one warp) per tile
constexpr int32_t DPG_CHUNK = 32; // FROZEN chunk size (tile isects per chunk partial)

__device__ __forceinline__ int32_t dpgr0_chunk_count(int32_t count)
{
    return (count + DPG_CHUNK - 1) / DPG_CHUNK;
}

// ---- Range-build: stamp key = image_id*N + gaussian and tile_id per isect ----
__global__ void dpgr0_build_keys_kernel(
    const int32_t *__restrict__ flatten_ids,
    const int32_t *__restrict__ tile_offsets, // [I*th*tw] tile starts
    const int32_t I,
    const int32_t N,
    const int32_t n_tiles,
    const int32_t tile_h,
    const int32_t tile_w,
    const int32_t n_isects,
    int32_t *__restrict__ key,          // [n_isects]
    int32_t *__restrict__ tile_id_of)   // [n_isects]
{
    const int32_t ft = blockIdx.x * blockDim.x + threadIdx.x;
    if(ft >= n_tiles)
    {
        return;
    }
    const int32_t tiles_per_image = tile_h * tile_w;
    const int32_t i = ft / tiles_per_image;
    const int32_t t = ft % tiles_per_image;
    const int32_t range_start = tile_offsets[ft];
    const int32_t range_end   = (ft == n_tiles - 1)
                                    ? n_isects
                                    : tile_offsets[ft + 1];
    const int32_t base = i * N;
    for(int32_t oi = range_start; oi < range_end; ++oi)
    {
        key[oi]        = base + flatten_ids[oi];
        tile_id_of[oi] = t;
    }
}

// Segment start flags: f=1 where a new key begins.
__global__ void dpgr0_mark_segments_kernel(
    const int32_t *__restrict__ sorted_key, // [n_isects]
    const int32_t n_isects,
    int32_t *__restrict__ flag)             // [n_isects] out
{
    const int32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if(m >= n_isects)
    {
        return;
    }
    flag[m] = (m == 0 || sorted_key[m] != sorted_key[m - 1]) ? 1 : 0;
}

// Scatter seg_start/seg_key from flags using the exclusive-scan seg_id.
__global__ void dpgr0_fill_segments_kernel(
    const int32_t *__restrict__ sorted_key,
    const int32_t *__restrict__ flag,
    const int32_t *__restrict__ seg_id,   // [n_isects] exclusive scan of flag
    const int32_t n_isects,
    int32_t *__restrict__ seg_start,
    int32_t *__restrict__ seg_key)
{
    const int32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if(m >= n_isects)
    {
        return;
    }
    if(flag[m])
    {
        const int32_t s = seg_id[m];
        seg_start[s]    = m;
        seg_key[s]      = sorted_key[m];
    }
}

// Per-segment chunk counts (CHUNKED only) for the bounded chunk-partial buffer.
__global__ void dpgr0_chunk_counts_kernel(
    const int32_t *__restrict__ seg_start, // [n_seg+1]
    const int32_t n_seg,
    int32_t *__restrict__ chunk_count,     // [n_seg] out: #chunk partials (CHUNKED only)
    int32_t *__restrict__ chunked_flag)    // [n_seg] out: 1 if CHUNKED (count>32)
{
    const int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n_seg)
    {
        return;
    }
    const int32_t count = seg_start[s + 1] - seg_start[s];
    const bool chunked  = (count > DPG_CHUNK);
    chunk_count[s]      = chunked ? dpgr0_chunk_count(count) : 0;
    chunked_flag[s]     = chunked ? 1 : 0;
}

// ---- Direct gather: one warp per segment; tile-replay; DIRECT/CHUNKED emit ----
template<uint32_t CDIM, bool SCALAR_ADJOINT = false, bool H8_MR = false>
__global__ void __launch_bounds__(32, 8) dpgr0_gather_kernel(
    const int32_t I,
    const int32_t N,
    const int32_t n_isects,
    const int32_t tile_h,
    const int32_t tile_w,
    const int32_t tile_size,
    const int32_t width,
    const int32_t height,
    const vec2 *__restrict__ means2d,
    const vec3 *__restrict__ conics,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ backgrounds,   // [I*CDIM] or nullptr
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const int32_t *__restrict__ tile_id_of,
    const int32_t *__restrict__ sorted_vals, // [n_isects] original isect idx
    const int32_t *__restrict__ seg_start,   // [n_seg+1]
    const int32_t *__restrict__ seg_key,     // [n_seg]
    const int32_t *__restrict__ chunk_offset,// [n_seg]
    float *__restrict__ chunk_buffer,        // [total_chunks * pstride] or null
    const float *__restrict__ render_alphas, // [I*H*W]
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas,
    float *__restrict__ v_means2d,
    float *__restrict__ v_conics,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities,
    const int pstride)
{
    const int32_t seg = blockIdx.x;
    const int32_t start = seg_start[seg];
    const int32_t count = seg_start[seg + 1] - start;
    const int32_t key   = seg_key[seg];
    const int32_t image_id = key / N;

    float *out_means2d = v_means2d   + (int64_t)key * 2;
    float *out_conics  = v_conics    + (int64_t)key * 3;
    float *out_colors  = v_colors    + (int64_t)key * CDIM;
    float *out_opac    = v_opacities + key;

    const uint32_t ty = threadIdx.y;
    const uint32_t tx = threadIdx.x;
    constexpr uint32_t PX_ROWS = 16 / DPG_PX;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t tr = block.thread_rank();

    const float *bg_img = (backgrounds != nullptr) ? backgrounds + image_id * CDIM : nullptr;
    const float *ra_img  = render_alphas   + (int64_t)image_id * height * width;
    const int32_t *li_img = last_ids       + (int64_t)image_id * height * width;
    const float *vrc_img = v_render_colors + (int64_t)image_id * height * width * CDIM;
    const float *vra_img = v_render_alphas + (int64_t)image_id * height * width;
    const int32_t tiles_per_image = tile_h * tile_w;
    const int32_t n_tiles = I * tiles_per_image;

    const bool is_chunked = (count > DPG_CHUNK);

    // Per-isect tile-replay: capture the warp-reduced 9-float raster gradient
    // at isect oi (the isect of segment gaussian g within its tile).
    auto replay_capture = [&](int32_t oi, float *partial)
    {
        const int32_t tile_id = tile_id_of[oi];
        const int32_t ft      = image_id * tiles_per_image + tile_id;
        const int32_t tstart  = tile_offsets[ft];
        const int32_t tend    = (ft == n_tiles - 1) ? n_isects : tile_offsets[ft + 1];

        const uint32_t i0 = (tile_id / tile_w) * tile_size;
        const uint32_t j0 = (tile_id % tile_w) * tile_size;

        float px[DPG_PX], py[DPG_PX];
        int32_t pix_id[DPG_PX];
        bool inside[DPG_PX];
        float T_final[DPG_PX], T[DPG_PX];
        float buffer[DPG_PX][CDIM];
        float v_render_c[DPG_PX][CDIM];
        float v_render_a[DPG_PX];
        int32_t bin_final[DPG_PX];

#pragma unroll
        for(uint32_t q = 0; q < DPG_PX; ++q)
        {
            const uint32_t i = i0 + ty + q * PX_ROWS;
            const uint32_t j = j0 + tx;
            px[q]     = (float)j + 0.5f;
            py[q]     = (float)i + 0.5f;
            pix_id[q] = min(i * width + j, width * height - 1);
            inside[q] = (i < (uint32_t)height && j < (uint32_t)width);
            T_final[q] = 1.0f - ra_img[pix_id[q]];
            T[q]       = T_final[q];
            bin_final[q] = inside[q] ? li_img[pix_id[q]] : 0;
#pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                v_render_c[q][k] = vrc_img[pix_id[q] * CDIM + k];
                buffer[q][k]     = 0.f;
            }
            v_render_a[q] = vra_img[pix_id[q]];
        }

        // Reverse scan from tile tail down to this isect (global isect idx oi).
        for(int32_t gidx = tend - 1; gidx >= oi; --gidx)
        {
            // Fresh per-isect gradient accumulators. warpSum() broadcasts the
            // warp reduction to ALL lanes, so these must NOT carry across the
            // reverse scan (only T/buffer do). The captured partial is exactly
            // isect oi's own gradient contribution.
            float v_rgb_local[CDIM]   = {0.f};
            vec3 v_conic_local        = {0.f, 0.f, 0.f};
            vec2 v_xy_local           = {0.f, 0.f};
            vec2 v_xy_abs_local       = {0.f, 0.f};
            float v_opacity_local     = 0.f;
            const int32_t gpos  = flatten_ids[gidx];
            const vec2 xy       = means2d[gpos];
            const float opac    = opacities[gpos];
            const vec3 conic    = conics[gpos];
            const float *rgbs   = colors + (int64_t)gpos * CDIM;

            float v_rgb_q[CDIM]    = {0.f};
            vec3 v_conic_q         = {0.f, 0.f, 0.f};
            vec2 v_xy_q            = {0.f, 0.f};
            vec2 v_xy_abs_q        = {0.f, 0.f};
            float v_opacity_q      = 0.f;

#pragma unroll
            for(uint32_t q = 0; q < DPG_PX; ++q)
            {
                bool valid_q = inside[q];
                if(gidx > bin_final[q])
                {
                    valid_q = false;
                }
                vec3 conic_q;
                vec2 delta_q;
                float alpha_q = 0.f, opac_q = 0.f, vis_q = 0.f;
                if(valid_q)
                {
                    conic_q     = conic;
                    vec3 xy_o    = {xy.x, xy.y, opac};
                    opac_q       = xy_o.z;
                    delta_q      = {xy_o.x - px[q], xy_o.y - py[q]};
                    const GaussianWeight gw = eval_gaussian_weight(conic_q, delta_q.x, delta_q.y, opac_q);
                    vis_q        = gw.vis;
                    alpha_q      = gw.alpha;
                    if(!gw.valid)
                    {
                        valid_q = false;
                    }
                }
                if(valid_q)
                {
                    float v_rgb_qq[CDIM]    = {0.f};
                    vec3 v_conic_qq         = {0.f, 0.f, 0.f};
                    vec2 v_xy_qq            = {0.f, 0.f};
                    vec2 v_xy_abs_qq        = {0.f, 0.f};
                    float v_opacity_qq      = 0.f;
                    rasterize_to_pixels_3dgs_blend_bwd<CDIM, SCALAR_ADJOINT, H8_MR>(
                        conic_q, delta_q, opac_q, vis_q, alpha_q, rgbs,
                        v_render_c[q], v_render_a[q], T_final[q], bg_img, false,
                        T[q], buffer[q], v_rgb_qq, v_conic_qq, v_xy_qq, v_xy_abs_qq, v_opacity_qq);
#pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        v_rgb_local[k] += v_rgb_qq[k];
                    }
                    v_conic_local.x += v_conic_qq.x;
                    v_conic_local.y += v_conic_qq.y;
                    v_conic_local.z += v_conic_qq.z;
                    v_xy_local.x    += v_xy_qq.x;
                    v_xy_local.y    += v_xy_qq.y;
                    v_opacity_local += v_opacity_qq;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_opacity_local, warp);

            if(gidx == oi)
            {
                partial[0] = v_xy_local.x;
                partial[1] = v_xy_local.y;
                partial[2] = v_conic_local.x;
                partial[3] = v_conic_local.y;
                partial[4] = v_conic_local.z;
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    partial[5 + k] = v_rgb_local[k];
                }
                partial[5 + CDIM] = v_opacity_local;
            }
        }
    };

    if(!is_chunked)
    {
        float acc[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for(int32_t j = 0; j < count; ++j)
        {
            const int32_t oi = sorted_vals[start + j];
            float partial[9];
            replay_capture(oi, partial);
#pragma unroll
            for(int k = 0; k < 9; ++k)
            {
                acc[k] += partial[k];
            }
        }
        if(tr == 0)
        {
            out_means2d[0] = acc[0];
            out_means2d[1] = acc[1];
            out_conics[0]  = acc[2];
            out_conics[1]  = acc[3];
            out_conics[2]  = acc[4];
#pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                out_colors[k] = acc[5 + k];
            }
            *out_opac = acc[5 + CDIM];
        }
    }
    else
    {
        const int32_t n_chunks = dpgr0_chunk_count(count);
        for(int32_t c = 0; c < n_chunks; ++c)
        {
            const int32_t jbegin = c * DPG_CHUNK;
            const int32_t jend   = min(jbegin + DPG_CHUNK, count);
            float cacc[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for(int32_t j = jbegin; j < jend; ++j)
            {
                const int32_t oi = sorted_vals[start + j];
                float partial[9];
                replay_capture(oi, partial);
#pragma unroll
                for(int k = 0; k < 9; ++k)
                {
                    cacc[k] += partial[k];
                }
            }
            if(tr == 0)
            {
                float *cb = chunk_buffer + (int64_t)(chunk_offset[seg] + c) * pstride;
                cb[0] = cacc[0];
                cb[1] = cacc[1];
                cb[2] = cacc[2];
                cb[3] = cacc[3];
                cb[4] = cacc[4];
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    cb[5 + k] = cacc[5 + k];
                }
                cb[5 + CDIM] = cacc[5 + CDIM];
            }
        }
    }
}

// ---- Chunk reducer: sum each CHUNKED segment's chunk partials -> final ----
template<uint32_t CDIM>
__global__ void dpgr0_chunk_reduce_kernel(
    const int32_t n_seg,
    const int32_t *__restrict__ seg_start,
    const int32_t *__restrict__ seg_key,
    const int32_t *__restrict__ chunk_count,
    const int32_t *__restrict__ chunk_offset,
    const float *__restrict__ chunk_buffer,
    const int pstride,
    float *__restrict__ v_means2d,
    float *__restrict__ v_conics,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities)
{
    const int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= n_seg)
    {
        return;
    }
    const int32_t cc = chunk_count[s];
    if(cc == 0)
    {
        return; // DIRECT segment already written by the gather kernel
    }
    const int32_t key = seg_key[s];
    float *out_means2d = v_means2d   + (int64_t)key * 2;
    float *out_conics  = v_conics    + (int64_t)key * 3;
    float *out_colors  = v_colors    + (int64_t)key * CDIM;
    float *out_opac    = v_opacities + key;

    const float *base = chunk_buffer + (int64_t)chunk_offset[s] * pstride;
    float sx = 0.f, sy = 0.f, cx = 0.f, cy = 0.f, cz = 0.f, o = 0.f;
    float col[4] = {0.f, 0.f, 0.f, 0.f};
    for(int32_t c = 0; c < cc; ++c)
    {
        const float *rec = base + (int64_t)c * pstride;
        sx += rec[0];
        sy += rec[1];
        cx += rec[2];
        cy += rec[3];
        cz += rec[4];
        for(int k = 0; k < CDIM; ++k)
        {
            col[k] += rec[5 + k];
        }
        o += rec[5 + CDIM];
    }
    out_means2d[0] = sx;
    out_means2d[1] = sy;
    out_conics[0]  = cx;
    out_conics[1]  = cy;
    out_conics[2]  = cz;
    for(int k = 0; k < CDIM; ++k)
    {
        out_colors[k] = col[k];
    }
    *out_opac = o;
}

} // namespace

// ---- Timed-build introspection state (last dpgr0_stage1 call) ----
static float    g_dpgr0_range_ms   = 0.f;
static float    g_dpgr0_gather_ms  = 0.f;
static float    g_dpgr0_reduce_ms  = 0.f;
static int64_t  g_dpgr0_n_isects   = 0;
static int64_t  g_dpgr0_n_seg      = 0;
static int64_t  g_dpgr0_n_chunked  = 0;
static int64_t  g_dpgr0_total_chunks = 0;
static int      g_dpgr0_pstride   = 0;
static int64_t  g_dpgr0_N          = 0;
static int64_t  g_dpgr0_records    = 0;
static int64_t  g_dpgr0_record_bytes = 0;

void dpgr0_stage1(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor colors_eval,
    at::Tensor opacities,
    const c10::optional<at::Tensor> &backgrounds,
    at::Tensor tile_offsets,
    at::Tensor flatten_ids,
    const c10::optional<at::Tensor> &active_tiles,
    at::Tensor render_alphas,
    at::Tensor last_ids,
    at::Tensor v_render_colors,
    at::Tensor v_render_alphas,
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
    at::Tensor v_means2d,
    at::Tensor v_conics,
    at::Tensor v_colors,
    at::Tensor v_opacities
)
{
    if(n_isects <= 0)
    {
        return;
    }
    TORCH_CHECK(I * N <= 0x7fffffffLL, "DPGR0: I*N exceeds int32 key range");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts   = means2d.options();
    const int pstride = 6 + static_cast<int>(color_dim);

    // ---- timed-build events (frozen 3-bucket breakdown: range / gather / reduce) ----
    cudaEvent_t dpg_e0, dpg_e1, dpg_e2, dpg_e3;
    cudaEventCreate(&dpg_e0); cudaEventCreate(&dpg_e1);
    cudaEventCreate(&dpg_e2); cudaEventCreate(&dpg_e3);
    cudaEventRecord(dpg_e0, stream);

    const int32_t n_tiles = static_cast<int32_t>(I * tile_h * tile_w);

    at::Tensor key          = at::empty({n_isects}, opts.dtype(at::kInt));
    at::Tensor tile_id_of   = at::empty({n_isects}, opts.dtype(at::kInt));
    at::Tensor sorted_key   = at::empty({n_isects}, opts.dtype(at::kInt));
    at::Tensor sorted_vals  = at::empty({n_isects}, opts.dtype(at::kInt));
    at::Tensor flag         = at::empty({n_isects}, opts.dtype(at::kInt));
    at::Tensor seg_id       = at::empty({n_isects}, opts.dtype(at::kInt));

    // (1) range-build keys + tile ids
    {
        const int32_t threads = 256;
        const int32_t grid    = (n_tiles + threads - 1) / threads;
        dpgr0_build_keys_kernel<<<grid, threads, 0, stream>>>(
            flatten_ids.const_data_ptr<int32_t>(),
            tile_offsets.const_data_ptr<int32_t>(),
            static_cast<int32_t>(I), static_cast<int32_t>(N), n_tiles,
            static_cast<int32_t>(tile_h), static_cast<int32_t>(tile_w),
            static_cast<int32_t>(n_isects),
            key.data_ptr<int32_t>(), tile_id_of.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // (2) stable radix sort by key (32-bit), carrying original isect index
    {
        at::Tensor vals_in = at::arange(n_isects, opts.dtype(at::kInt));
        void *d_temp      = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            d_temp, temp_bytes,
            key.data_ptr<int32_t>(), sorted_key.data_ptr<int32_t>(),
            vals_in.data_ptr<int32_t>(), sorted_vals.data_ptr<int32_t>(),
            static_cast<int>(n_isects), 0, 32);
        at::Tensor temp = at::empty({static_cast<int64_t>(temp_bytes)}, opts);
        cub::DeviceRadixSort::SortPairs(
            temp.data_ptr(), temp_bytes,
            key.data_ptr<int32_t>(), sorted_key.data_ptr<int32_t>(),
            vals_in.data_ptr<int32_t>(), sorted_vals.data_ptr<int32_t>(),
            static_cast<int>(n_isects), 0, 32);
    }

    // (3) segment table: flag -> exclusive scan -> seg_id -> scatter seg_start
    int32_t n_seg = 0;
    at::Tensor seg_start;
    at::Tensor seg_key;
    {
        const int32_t threads = 256;
        const int32_t grid    = (static_cast<int32_t>(n_isects) + threads - 1) / threads;
        dpgr0_mark_segments_kernel<<<grid, threads, 0, stream>>>(
            sorted_key.const_data_ptr<int32_t>(), static_cast<int32_t>(n_isects),
            flag.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        void *d_temp      = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            d_temp, temp_bytes, flag.data_ptr<int32_t>(), seg_id.data_ptr<int32_t>(),
            static_cast<int>(n_isects));
        at::Tensor temp = at::empty({static_cast<int64_t>(temp_bytes)}, opts);
        cub::DeviceScan::ExclusiveSum(
            temp.data_ptr(), temp_bytes, flag.data_ptr<int32_t>(), seg_id.data_ptr<int32_t>(),
            static_cast<int>(n_isects));

        // n_seg = seg_id[n_isects-1] + flag[n_isects-1]
        int32_t h_sidlast = 0, h_last = 0;
        cudaMemcpyAsync(&h_sidlast, seg_id.data_ptr<int32_t>() + n_isects - 1,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_last, flag.data_ptr<int32_t>() + n_isects - 1,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaStreamSynchronize(stream);
        n_seg = h_sidlast + h_last;

        seg_start = at::empty({static_cast<int64_t>(n_seg) + 1}, opts.dtype(at::kInt));
        seg_key   = at::empty({static_cast<int64_t>(n_seg)}, opts.dtype(at::kInt));
        dpgr0_fill_segments_kernel<<<grid, threads, 0, stream>>>(
            sorted_key.const_data_ptr<int32_t>(), flag.data_ptr<int32_t>(),
            seg_id.const_data_ptr<int32_t>(), static_cast<int32_t>(n_isects),
            seg_start.data_ptr<int32_t>(), seg_key.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // trailing sentinel seg_start[n_seg] = n_isects
        int32_t h_n = static_cast<int32_t>(n_isects);
        cudaMemcpyAsync(seg_start.data_ptr<int32_t>() + n_seg, &h_n,
                        sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    }

    // (4) chunk counts + exclusive prefix sum -> chunk_offset, total_chunks
    at::Tensor chunk_count;
    at::Tensor chunk_offset;
    int32_t total_chunks = 0;
    {
        chunk_count  = at::zeros({static_cast<int64_t>(n_seg)}, opts.dtype(at::kInt));
        chunk_offset = at::empty({static_cast<int64_t>(n_seg)}, opts.dtype(at::kInt));
        at::Tensor chunked_flag = at::zeros({static_cast<int64_t>(n_seg)}, opts.dtype(at::kInt));
        const int32_t threads = 256;
        const int32_t grid    = (n_seg + threads - 1) / threads;
        dpgr0_chunk_counts_kernel<<<grid, threads, 0, stream>>>(
            seg_start.const_data_ptr<int32_t>(), n_seg, chunk_count.data_ptr<int32_t>(),
            chunked_flag.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        void *d_temp      = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            d_temp, temp_bytes, chunk_count.data_ptr<int32_t>(), chunk_offset.data_ptr<int32_t>(),
            n_seg);
        at::Tensor temp = at::empty({static_cast<int64_t>(temp_bytes)}, opts);
        cub::DeviceScan::ExclusiveSum(
            temp.data_ptr(), temp_bytes, chunk_count.data_ptr<int32_t>(), chunk_offset.data_ptr<int32_t>(),
            n_seg);
        int32_t h_co = 0, h_cc = 0;
        cudaMemcpyAsync(&h_co, chunk_offset.data_ptr<int32_t>() + n_seg - 1,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_cc, chunk_count.data_ptr<int32_t>() + n_seg - 1,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaStreamSynchronize(stream);
        total_chunks = (n_seg > 0) ? (h_co + h_cc) : 0;

        // n_chunked_seg = sum(chunked_flag); records = n_direct_seg + total_chunks.
        int64_t n_chunked = 0;
        if(n_seg > 0)
        {
            at::Tensor d_nchunked = at::zeros({1}, opts.dtype(at::kInt));
            void *d_t2 = nullptr; size_t tb2 = 0;
            cub::DeviceReduce::Sum(d_t2, tb2, chunked_flag.data_ptr<int32_t>(),
                                   d_nchunked.data_ptr<int32_t>(), n_seg);
            at::Tensor t2 = at::empty({static_cast<int64_t>(tb2)}, opts);
            cub::DeviceReduce::Sum(t2.data_ptr(), tb2, chunked_flag.data_ptr<int32_t>(),
                                   d_nchunked.data_ptr<int32_t>(), n_seg);
            int32_t h_nc = 0;
            cudaMemcpyAsync(&h_nc, d_nchunked.data_ptr<int32_t>(), sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            n_chunked = (int64_t)h_nc;
        }
        g_dpgr0_n_chunked     = n_chunked;
        g_dpgr0_records       = (int64_t)(n_seg - n_chunked) + (int64_t)total_chunks;
        g_dpgr0_record_bytes  = g_dpgr0_records * (int64_t)pstride * (int64_t)sizeof(float);
        cudaEventRecord(dpg_e1, stream);
    }

    at::Tensor chunk_buffer;
    float *chunk_ptr = nullptr;
    if(total_chunks > 0)
    {
        chunk_buffer = at::empty({static_cast<int64_t>(total_chunks) * pstride}, opts);
        chunk_ptr    = chunk_buffer.data_ptr<float>();
    }

    const float *bg_ptr =
        backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr;
    (void)active_tiles; // DPGR0 ranges are built from tile_offsets/flatten_ids directly

#define LAUNCH_GATHER(CDIM, SCALAR_VAL, H8_MR_VAL)                                                \
    {                                                                                          \
        dpgr0_gather_kernel<CDIM, (SCALAR_VAL), (H8_MR_VAL)>                                    \
            <<<n_seg, dim3(16, 2, 1), 0, stream>>>(                                            \
                static_cast<int32_t>(I), static_cast<int32_t>(N), static_cast<int32_t>(n_isects),\
                static_cast<int32_t>(tile_h), static_cast<int32_t>(tile_w),                    \
                static_cast<int32_t>(tile_size), static_cast<int32_t>(width),                  \
                static_cast<int32_t>(height),                                                  \
                reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),       reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()), \
                colors_eval.const_data_ptr<float>(), opacities.const_data_ptr<float>(),        \
                bg_ptr, tile_offsets.const_data_ptr<int32_t>(),                                \
                flatten_ids.const_data_ptr<int32_t>(), tile_id_of.const_data_ptr<int32_t>(),   \
                sorted_vals.const_data_ptr<int32_t>(),                                         \
                seg_start.const_data_ptr<int32_t>(), seg_key.data_ptr<int32_t>(),              \
                chunk_offset.const_data_ptr<int32_t>(), chunk_ptr,                             \
                render_alphas.const_data_ptr<float>(), last_ids.const_data_ptr<int32_t>(),     \
                v_render_colors.const_data_ptr<float>(), v_render_alphas.const_data_ptr<float>(),\
                v_means2d.data_ptr<float>(), v_conics.data_ptr<float>(),                       \
                v_colors.data_ptr<float>(), v_opacities.data_ptr<float>(), pstride);           \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                        \
    }

    switch(color_dim)
    {
    case 1:
        if(scalar_adjoint && h8_mr) LAUNCH_GATHER(1, true, true)
        else if(scalar_adjoint) LAUNCH_GATHER(1, true, false)
        else LAUNCH_GATHER(1, false, false)
        break;
    case 3:
        if(scalar_adjoint && h8_mr) LAUNCH_GATHER(3, true, true)
        else if(scalar_adjoint) LAUNCH_GATHER(3, true, false)
        else LAUNCH_GATHER(3, false, false)
        break;
    default:
        if(scalar_adjoint && h8_mr) LAUNCH_GATHER(4, true, true)
        else if(scalar_adjoint) LAUNCH_GATHER(4, true, false)
        else LAUNCH_GATHER(4, false, false)
        break;
    }
#undef LAUNCH_GATHER
    cudaEventRecord(dpg_e2, stream);

    if(total_chunks > 0)
    {
#define LAUNCH_REDUCE(CDIM)                                                                    \
        {                                                                                     \
            const int32_t rthreads = 256;                                                     \
            const int32_t rgrid   = (n_seg + rthreads - 1) / rthreads;                        \
            dpgr0_chunk_reduce_kernel<CDIM><<<rgrid, rthreads, 0, stream>>>(                 \
                n_seg, seg_start.const_data_ptr<int32_t>(), seg_key.const_data_ptr<int32_t>(), \
                chunk_count.const_data_ptr<int32_t>(), chunk_offset.const_data_ptr<int32_t>(),  \
                chunk_buffer.const_data_ptr<float>(), pstride,                                \
                v_means2d.data_ptr<float>(), v_conics.data_ptr<float>(),                      \
                v_colors.data_ptr<float>(), v_opacities.data_ptr<float>());                   \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                   \
        }
        switch(color_dim)
        {
        case 1: LAUNCH_REDUCE(1) break;
        case 3: LAUNCH_REDUCE(3) break;
        default: LAUNCH_REDUCE(4) break;
        }
#undef LAUNCH_REDUCE
    }

    cudaEventRecord(dpg_e3, stream);
    cudaEventSynchronize(dpg_e3);
    cudaEventElapsedTime(&g_dpgr0_range_ms, dpg_e0, dpg_e1);
    cudaEventElapsedTime(&g_dpgr0_gather_ms, dpg_e1, dpg_e2);
    cudaEventElapsedTime(&g_dpgr0_reduce_ms, dpg_e2, dpg_e3);
    g_dpgr0_n_isects      = (int64_t)n_isects;
    g_dpgr0_n_seg         = (int64_t)n_seg;
    g_dpgr0_total_chunks  = (int64_t)total_chunks;
    g_dpgr0_pstride       = pstride;
    g_dpgr0_N             = (int64_t)N;
    cudaEventDestroy(dpg_e0); cudaEventDestroy(dpg_e1);
    cudaEventDestroy(dpg_e2); cudaEventDestroy(dpg_e3);
}

std::vector<double> dpgr0_get_stage_times()
{
    return {(double)g_dpgr0_range_ms, (double)g_dpgr0_gather_ms, (double)g_dpgr0_reduce_ms};
}

std::vector<int64_t> dpgr0_get_stats()
{
    return {g_dpgr0_n_isects, g_dpgr0_n_seg, g_dpgr0_n_chunked, g_dpgr0_total_chunks,
            (int64_t)g_dpgr0_pstride, g_dpgr0_N, g_dpgr0_records, g_dpgr0_record_bytes};
}

} // namespace gaussian_render_inference_scene
} // namespace gsplat
