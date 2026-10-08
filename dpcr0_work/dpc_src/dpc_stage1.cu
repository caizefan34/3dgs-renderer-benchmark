// SPDX-License-Identifier: Apache-2.0
//
// DPC0 Stage 1: deterministic tile-primitive partial accumulation + Gaussian-major
// segmented reduction. See dpc_stage1.h. No floating-point atomicAdd touches any
// final gradient.

#include "Config.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/device/device_radix_sort.cuh>

#include <cstdint>
#include <cstdlib>

#include "Common.h"
#include "RasterizeToPixels3DGSDevice.cuh"
#include "Utils.cuh"
#include "dpc_stage1.h"

namespace gsplat
{
namespace gaussian_render_inference_scene
{
namespace cg = cooperative_groups;

namespace
{

constexpr uint32_t DPC_PX = 8; // pixels per thread -> 32 threads (one warp) per tile block

// One block per (image, tile); the block is a single warp (32 threads), each
// thread owning DPC_PX=8 pixels (strided by PX_ROWS=2 rows). For every
// (tile, Gaussian) isect the warp reduces the covered pixels in a fixed
// reduction tree and lane 0 writes EXACTLY ONE partial record at the isect's
// global slot. No atomicAdd: each isect slot is written once. Occluded
// isects (beyond every pixel's last contributing id) write a zero record.
template<uint32_t CDIM, bool SCALAR_ADJOINT = false, bool H8_MR = false>
__global__ void __launch_bounds__(32, 8) dpc_blend_partial_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2 *__restrict__ means2d,
    const vec3 *__restrict__ conics,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ backgrounds,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ active_tiles,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas,
    float *__restrict__ partial,        // [n_isects, pstride]
    int32_t *__restrict__ partial_key,  // [n_isects] = image*N + gaussian
    const int pstride
)
{
    auto block = cg::this_thread_block();

    uint32_t image_id, tile_id, i0, j0;
    if(active_tiles != nullptr)
    {
        const int32_t global_tile = active_tiles[blockIdx.x];
        image_id = static_cast<uint32_t>(global_tile / (int32_t)(tile_width * tile_height));
        tile_id  = static_cast<uint32_t>(global_tile % (int32_t)(tile_width * tile_height));
        j0       = (tile_id % tile_width) * tile_size;
        i0       = (tile_id / tile_width) * tile_size;
    }
    else
    {
        image_id = block.group_index().x;
        tile_id  = block.group_index().y * tile_width + block.group_index().z;
        i0       = block.group_index().y * tile_size;
        j0       = block.group_index().z * tile_size;
    }

    const uint32_t ty = block.thread_index().y;
    const uint32_t tx = block.thread_index().x;
    constexpr uint32_t PX_ROWS = 16 / DPC_PX; // 2

    tile_offsets    += (int64_t)image_id * tile_height * tile_width;
    render_alphas   += (int64_t)image_id * image_height * image_width;
    last_ids        += (int64_t)image_id * image_height * image_width;
    v_render_colors += (int64_t)image_id * image_height * image_width * CDIM;
    v_render_alphas += (int64_t)image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }

    float px[DPC_PX], py[DPC_PX];
    int32_t pix_id[DPC_PX];
    bool inside[DPC_PX];
    float T_final[DPC_PX], T[DPC_PX];
    float buffer[DPC_PX][CDIM];
    float v_render_c[DPC_PX][CDIM];
    float v_render_a[DPC_PX];
    int32_t bin_final[DPC_PX];

#pragma unroll
    for(uint32_t q = 0; q < DPC_PX; ++q)
    {
        const uint32_t i = i0 + ty + q * PX_ROWS;
        const uint32_t j = j0 + tx;
        px[q]            = (float)j + 0.5f;
        py[q]            = (float)i + 0.5f;
        pix_id[q]        = min(i * image_width + j, image_width * image_height - 1);
        inside[q]        = (i < image_height && j < image_width);
        T_final[q]       = 1.0f - render_alphas[pix_id[q]];
        T[q]             = T_final[q];
        bin_final[q]     = inside[q] ? last_ids[pix_id[q]] : 0;
#pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            v_render_c[q][k] = v_render_colors[pix_id[q] * CDIM + k];
            buffer[q][k]     = 0.f;
        }
        v_render_a[q] = v_render_alphas[pix_id[q]];
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end   = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
                              ? (int32_t)n_isects
                              : tile_offsets[tile_id + 1];
    const uint32_t block_size  = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);
    float *rgbs_batch      = (float *)&conic_batch[block_size];

    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    for(uint32_t b = 0; b < num_batches; ++b)
    {
        block.sync();
        const int32_t batch_end  = range_end - 1 - (int32_t)(block_size * b);
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx        = batch_end - (int32_t)tr;
        if(idx >= range_start)
        {
            int32_t g            = flatten_ids[idx];
            id_batch[tr]         = g;
            const vec2 xy        = means2d[g];
            const float opac     = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr]      = conics[g];
#pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[(int64_t)g * CDIM + k];
            }
        }
        block.sync();
        // Process every isect in the batch (occluded isects write a zero
        // record) so each isect slot is written exactly once and the partial
        // buffer needs no zero-initialization.
        for(uint32_t t = 0; t < (uint32_t)batch_size; ++t)
        {
            float v_rgb_local[CDIM]   = {0.f};
            vec3 v_conic_local        = {0.f, 0.f, 0.f};
            vec2 v_xy_local           = {0.f, 0.f};
            vec2 v_xy_abs_local       = {0.f, 0.f};
            float v_opacity_local     = 0.f;
#pragma unroll
            for(uint32_t q = 0; q < DPC_PX; ++q)
            {
                bool valid_q = inside[q];
                if(batch_end - (int32_t)t > bin_final[q])
                {
                    valid_q = false;
                }
                vec3 conic_q;
                vec2 delta_q;
                float alpha_q = 0.f, opac_q = 0.f, vis_q = 0.f;
                if(valid_q)
                {
                    conic_q                   = conic_batch[t];
                    vec3 xy_opac_q            = xy_opacity_batch[t];
                    opac_q                    = xy_opac_q.z;
                    delta_q                   = {xy_opac_q.x - px[q], xy_opac_q.y - py[q]};
                    const GaussianWeight gw   = eval_gaussian_weight(conic_q, delta_q.x, delta_q.y, opac_q);
                    vis_q                     = gw.vis;
                    alpha_q                   = gw.alpha;
                    if(!gw.valid)
                    {
                        valid_q = false;
                    }
                }
                if(valid_q)
                {
                    float v_rgb_q[CDIM]    = {0.f};
                    vec3 v_conic_q         = {0.f, 0.f, 0.f};
                    vec2 v_xy_q            = {0.f, 0.f};
                    vec2 v_xy_abs_q        = {0.f, 0.f};
                    float v_opacity_q      = 0.f;
                    rasterize_to_pixels_3dgs_blend_bwd<CDIM, SCALAR_ADJOINT, H8_MR>(
                        conic_q,
                        delta_q,
                        opac_q,
                        vis_q,
                        alpha_q,
                        rgbs_batch + t * CDIM,
                        v_render_c[q],
                        v_render_a[q],
                        T_final[q],
                        backgrounds,
                        false,
                        T[q],
                        buffer[q],
                        v_rgb_q,
                        v_conic_q,
                        v_xy_q,
                        v_xy_abs_q,
                        v_opacity_q
                    );
#pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        v_rgb_local[k] += v_rgb_q[k];
                    }
                    v_conic_local.x += v_conic_q.x;
                    v_conic_local.y += v_conic_q.y;
                    v_conic_local.z += v_conic_q.z;
                    v_xy_local.x += v_xy_q.x;
                    v_xy_local.y += v_xy_q.y;
                    v_xy_abs_local.x += v_xy_abs_q.x;
                    v_xy_abs_local.y += v_xy_abs_q.y;
                    v_opacity_local += v_opacity_q;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g     = id_batch[t];
                int64_t k_iso = (int64_t)batch_end - (int64_t)t;
                float *rec    = partial + k_iso * (int64_t)pstride;
                rec[0]        = v_xy_local.x;
                rec[1]        = v_xy_local.y;
                rec[2]        = v_conic_local.x;
                rec[3]        = v_conic_local.y;
                rec[4]        = v_conic_local.z;
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    rec[5 + k] = v_rgb_local[k];
                }
                rec[5 + CDIM]      = v_opacity_local;
                partial_key[k_iso] = (int32_t)image_id * (int32_t)N + (int32_t)g;
            }
        }
    }
}

// Deterministic segmented reduce: isects are already sorted by key (stable
// radix sort). Each segment start reduces its run in sorted order (== isect
// order == tile order) into the flat [I*N] raster-gradient buffers. No atomic.
__global__ void dpc_reduce_segments_kernel(
    const int32_t *__restrict__ sorted_keys, // [n_isects]
    const int32_t *__restrict__ sorted_vals, // [n_isects] original isect index
    const float *__restrict__ partial,       // [n_isects, pstride]
    const int32_t n_isects,
    const int pstride,
    const int color_dim,
    float *__restrict__ v_means2d,   // [I*N, 2]
    float *__restrict__ v_conics,    // [I*N, 3]
    float *__restrict__ v_colors,    // [I*N, color_dim]
    float *__restrict__ v_opacities  // [I*N]
)
{
    const int64_t m = cg::this_grid().thread_rank();
    if(m >= n_isects)
    {
        return;
    }
    const int32_t key = sorted_keys[m];
    if(m > 0 && sorted_keys[m - 1] == key)
    {
        return; // not a segment start
    }
    int64_t end = m + 1;
    while(end < n_isects && sorted_keys[end] == key)
    {
        ++end;
    }
    float sx = 0.f, sy = 0.f, cx = 0.f, cy = 0.f, cz = 0.f, o = 0.f;
    float col[4] = {0.f, 0.f, 0.f, 0.f};
    for(int64_t e = m; e < end; ++e)
    {
        const float *rec = partial + (int64_t)sorted_vals[e] * (int64_t)pstride;
        sx += rec[0];
        sy += rec[1];
        cx += rec[2];
        cy += rec[3];
        cz += rec[4];
        for(int k = 0; k < color_dim; ++k)
        {
            col[k] += rec[5 + k];
        }
        o += rec[5 + color_dim];
    }
    const int64_t base = (int64_t)key; // flat index = image*N + gaussian
    v_means2d[base * 2 + 0]      = sx;
    v_means2d[base * 2 + 1]      = sy;
    v_conics[base * 3 + 0]       = cx;
    v_conics[base * 3 + 1]       = cy;
    v_conics[base * 3 + 2]       = cz;
    for(int k = 0; k < color_dim; ++k)
    {
        v_colors[base * color_dim + k] = col[k];
    }
    v_opacities[base] = o;
}

} // namespace

void dpc_stage1(
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
    TORCH_CHECK(I * N <= 0x7fffffffLL, "DPC0: I*N exceeds int32 key range");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts   = means2d.options();
    const int pstride = 6 + static_cast<int>(color_dim);

    at::Tensor partial      = at::empty({n_isects * pstride}, opts);
    at::Tensor partial_key  = at::empty({n_isects}, opts.dtype(at::kInt));
    at::Tensor vals_in      = at::arange(n_isects, opts.dtype(at::kInt));
    at::Tensor sorted_keys  = at::empty({n_isects}, opts.dtype(at::kInt));
    at::Tensor sorted_vals  = at::empty({n_isects}, opts.dtype(at::kInt));

    const int32_t *active_ptr =
        active_tiles.has_value() ? active_tiles.value().const_data_ptr<int32_t>() : nullptr;
    const float *bg_ptr =
        backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr;

    // PX=8 -> 32 threads (one warp) per tile block.
    dim3 threads(16, 2, 1);
    dim3 grid;
    if(active_ptr != nullptr)
    {
        grid = dim3(static_cast<uint32_t>(active_tiles.value().numel()), 1, 1);
    }
    else
    {
        grid = dim3(static_cast<uint32_t>(I), static_cast<uint32_t>(tile_h), static_cast<uint32_t>(tile_w));
    }
    const uint32_t block_size = threads.x * threads.y * threads.z;
    const uint32_t shmem_size = block_size
        * (sizeof(int32_t) + 2 * sizeof(vec3) + static_cast<size_t>(color_dim) * sizeof(float));

#define LAUNCH_DPC_PARTIAL(CDIM, SCALAR_VAL, H8_MR_VAL)                                                       \
    {                                                                                                        \
        C10_CUDA_CHECK(cudaFuncSetAttribute(                                                                \
            dpc_blend_partial_kernel<(CDIM), (SCALAR_VAL), (H8_MR_VAL)>,                                     \
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));                     \
        dpc_blend_partial_kernel<(CDIM), (SCALAR_VAL), (H8_MR_VAL)>                                          \
            <<<grid, threads, static_cast<size_t>(shmem_size), stream>>>(                                   \
                static_cast<uint32_t>(I), static_cast<uint32_t>(N), static_cast<uint32_t>(n_isects),         \
                means2d.const_data_ptr<float>(), conics.const_data_ptr<float>(),                             \
                colors_eval.const_data_ptr<float>(), opacities.const_data_ptr<float>(), bg_ptr,              \
                static_cast<uint32_t>(tile_w), static_cast<uint32_t>(tile_h), active_ptr,                    \
                static_cast<uint32_t>(width), static_cast<uint32_t>(height),                                 \
                static_cast<uint32_t>(tile_size), tile_offsets.const_data_ptr<int32_t>(),                   \
                flatten_ids.const_data_ptr<int32_t>(), render_alphas.const_data_ptr<float>(),               \
                last_ids.const_data_ptr<int32_t>(), v_render_colors.const_data_ptr<float>(),                \
                v_render_alphas.const_data_ptr<float>(), partial.data_ptr<float>(),                          \
                partial_key.data_ptr<int32_t>(), pstride);                                                  \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                     \
    }

    switch(color_dim)
    {
    case 1:
        if(scalar_adjoint && h8_mr) LAUNCH_DPC_PARTIAL(1, true, true)
        else if(scalar_adjoint) LAUNCH_DPC_PARTIAL(1, true, false)
        else LAUNCH_DPC_PARTIAL(1, false, false)
        break;
    case 3:
        if(scalar_adjoint && h8_mr) LAUNCH_DPC_PARTIAL(3, true, true)
        else if(scalar_adjoint) LAUNCH_DPC_PARTIAL(3, true, false)
        else LAUNCH_DPC_PARTIAL(3, false, false)
        break;
    default:
        if(scalar_adjoint && h8_mr) LAUNCH_DPC_PARTIAL(4, true, true)
        else if(scalar_adjoint) LAUNCH_DPC_PARTIAL(4, true, false)
        else LAUNCH_DPC_PARTIAL(4, false, false)
        break;
    }
#undef LAUNCH_DPC_PARTIAL

    // Deterministic Gaussian-major segmented reduction (stable radix sort).
    void *d_temp      = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        partial_key.data_ptr<int32_t>(), sorted_keys.data_ptr<int32_t>(),
        vals_in.data_ptr<int32_t>(), sorted_vals.data_ptr<int32_t>(),
        static_cast<int>(n_isects), 0, 32);
    at::Tensor temp = at::empty({static_cast<int64_t>(temp_bytes)}, opts);
    cub::DeviceRadixSort::SortPairs(
        temp.data_ptr(), temp_bytes,
        partial_key.data_ptr<int32_t>(), sorted_keys.data_ptr<int32_t>(),
        vals_in.data_ptr<int32_t>(), sorted_vals.data_ptr<int32_t>(),
        static_cast<int>(n_isects), 0, 32);

    const uint32_t rthreads = 256;
    const uint32_t rgrid   = static_cast<uint32_t>((n_isects + rthreads - 1) / rthreads);
    dpc_reduce_segments_kernel<<<rgrid, rthreads, 0, stream>>>(
        sorted_keys.data_ptr<int32_t>(), sorted_vals.data_ptr<int32_t>(),
        partial.data_ptr<float>(), static_cast<int>(n_isects), pstride,
        static_cast<int>(color_dim), v_means2d.data_ptr<float>(),
        v_conics.data_ptr<float>(), v_colors.data_ptr<float>(),
        v_opacities.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gaussian_render_inference_scene
} // namespace gsplat