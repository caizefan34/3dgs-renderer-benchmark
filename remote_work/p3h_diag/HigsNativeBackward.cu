/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Native CUDA backward for the differentiable HiGS training path.
//
// The backward consumes *forward-captured* state from the differentiable
// forward (means2d/conics/evaluated-colors/opacities, per-tile sorted
// intersection ids, last contributing ids and render alphas) so it uses the
// exact same scene/order/visibility version as the forward. It never
// re-runs the rasterization pipeline.

#include "Config.h"

#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdint>
#include <cstdlib>

#include "Common.h"
#include "HigsNativeBackward.h"
#include "RasterizeToPixels3DGSDevice.cuh"
#include "Utils.cuh"

namespace gsplat
{
namespace gaussian_render_inference_scene
{
namespace cg = cooperative_groups;

// DSH-H diagnostic sink mode (compile-time, via -DP3_SINK_MODE=n).
//   0 = FULL (frozen C0 V3)
//   1 = ACCUM_SINK_ALL     (B color + C opacity + D H8 geom -> sink)
//   2 = ACCUM_SINK_GEOM    (D H8 geom only     -> sink; B,C real)
//   3 = ACCUM_SINK_APPEAR  (B color + C opacity -> sink; D real)
//   4 = WRITE_ONLY / NO_ATOMIC (full values, plain non-atomic uncontended stores)
// Sink = a per-warp uncontended slot in a diagnostic scratch buffer. All local
// gradient arithmetic (q eval, alpha derivative, T recurrence, color/opacity/H8
// moment accumulation, warp reduction, index mapping, branch decisions) is
// preserved; only the global atomic scatter destination is replaced.
#ifndef P3_SINK_MODE
#define P3_SINK_MODE 0
#endif

namespace
{
constexpr uint32_t kHigsColorDim = 3; // RGB channels (depth modes use 1 or 4)

// ============================================================================
// Stage 1: pixel-blend backward (per-tile, warp-reduced, atomicAdd scatter)
// ============================================================================
template<uint32_t CDIM>
__global__ void __launch_bounds__(256, 5) higs_blend_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    // forward-captured rasterize inputs (flat [I * N, ...] layout)
    const vec2 *__restrict__ means2d,
    const vec3 *__restrict__ conics,
    const float *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ backgrounds, // [I, CDIM] or nullptr
    // image geometry
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ active_tiles, // [AT] selected tile ids or nullptr (dense grid)
    const bool skip_background_atomic,        // background grads handled by a separate kernel
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // forward outputs
    const float *__restrict__ render_alphas, // [I, H, W]
    const int32_t *__restrict__ last_ids,    // [I, H, W]
    // loss gradients
    const float *__restrict__ v_render_colors, // [I, H, W, CDIM]
    const float *__restrict__ v_render_alphas, // [I, H, W]
    // gradient outputs (flat [I * N, ...])
    vec2 *__restrict__ v_means2d,
    vec3 *__restrict__ v_conics,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities,
    float *__restrict__ v_backgrounds // [CDIM] (summed over all images)
)
{
    auto block = cg::this_thread_block();

    uint32_t image_id, tile_id;
    if(active_tiles != nullptr)
    {
        // compacted grid: one block per selected tile, decoded from the list
        const int32_t global_tile = active_tiles[blockIdx.x];
        image_id = static_cast<uint32_t>(global_tile / (int32_t)(tile_width * tile_height));
        tile_id  = static_cast<uint32_t>(global_tile % (int32_t)(tile_width * tile_height));
    }
    else
    {
        image_id = block.group_index().x;
        tile_id  = block.group_index().y * tile_width + block.group_index().z;
    }
    const uint32_t tile_x = tile_id % tile_width;
    const uint32_t tile_y = tile_id / tile_width;
    uint32_t i            = tile_y * tile_size + block.thread_index().y;
    uint32_t j            = tile_x * tile_size + block.thread_index().x;

    tile_offsets    += (int64_t)image_id * tile_height * tile_width;
    render_alphas   += (int64_t)image_id * image_height * image_width;
    last_ids        += (int64_t)image_id * image_height * image_width;
    v_render_colors += (int64_t)image_id * image_height * image_width * CDIM;
    v_render_alphas += (int64_t)image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end   = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
                              ? (int32_t)n_isects
                              : tile_offsets[tile_id + 1];
    const uint32_t block_size  = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;                                            // [block_size]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);         // [block_size]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *rgbs_batch      = (float *)&conic_batch[block_size];                       // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final      = 1.0f - render_alphas[pix_id];
    float T            = T_final;
    float buffer[CDIM] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // background gradient: d(render)/d(background) = T_final per pixel. Skipped
    // when the compacted grid runs a separate all-pixel background kernel (so
    // active pixels are not double-counted).
    if(backgrounds != nullptr && !skip_background_atomic)
    {
#pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            atomicAdd(v_backgrounds + k, T_final * v_render_c[k]);
        }
    }

    // collect and process batches of gaussians
    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final   = cg::reduce(warp, bin_final, cg::greater<int>());
    for(uint32_t b = 0; b < num_batches; ++b)
    {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        const int32_t batch_end  = range_end - 1 - (int32_t)(block_size * b);
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx        = batch_end - (int32_t)tr;
        if(idx >= range_start)
        {
            int32_t g            = flatten_ids[idx]; // flatten index in [I * N]
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
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end - (int32_t)t > bin_final)
            {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;

            if(valid)
            {
                conic                   = conic_batch[t];
                vec3 xy_opac            = xy_opacity_batch[t];
                opac                    = xy_opac.z;
                delta                   = {xy_opac.x - px, xy_opac.y - py};
                const GaussianWeight gw = eval_gaussian_weight(conic, delta.x, delta.y, opac);
                vis                     = gw.vis;
                alpha                   = gw.alpha;
                if(!gw.valid)
                {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid))
            {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if(valid)
            {
                rasterize_to_pixels_3dgs_blend_bwd<CDIM>(
                    conic,
                    delta,
                    opac,
                    vis,
                    alpha,
                    rgbs_batch + t * CDIM,
                    v_render_c,
                    v_render_a,
                    T_final,
                    backgrounds,
                    false, // compute_abs
                    T,
                    buffer,
                    v_rgb_local,
                    v_conic_local,
                    v_xy_local,
                    v_xy_local, // v_xy_abs_local (unused when compute_abs=false)
                    v_opacity_local
                );
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t]; // flatten index in [I * N]
                float *v_rgb_ptr = v_colors + (int64_t)CDIM * g;
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    atomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * (int64_t)g;
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * (int64_t)g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                atomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}


// ============================================================================
// PX-pixels-per-thread variant of the pixel-blend backward.  Each thread owns
// PX pixels (strided by 16/PX rows inside the 16x16 render tile), so a tile's
// 256 pixels are covered by 256/PX threads.  The per-isect warp reductions
// and atomic scatters scale as 1/PX (same total per-pixel math), trading
// register pressure (PX pixel states) for fewer reduction/atomic operations.
// ============================================================================
template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false, bool H8_MR = false>
__global__ void __launch_bounds__(256 / PX, PX == 1 ? 5 : PX == 2 ? 5 : PX == 4 ? 8 : 10)
higs_blend_bwd_px_kernel(
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
    const int32_t *__restrict__ active_tiles, // [AT] selected tile ids or nullptr (dense grid)
    const bool skip_background_atomic,        // background grads handled by a separate kernel
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas,
    vec2 *__restrict__ v_means2d,
    vec3 *__restrict__ v_conics,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities,
    float *__restrict__ v_backgrounds
#if P3_SINK_MODE != 0
    ,
    float *__restrict__ v_p3h_scratch // diagnostic uncontended sink slot
#endif
)
{
    static_assert(PX == 1 || PX == 2 || PX == 4 || PX == 8, "PX must be 1/2/4/8");
    auto block = cg::this_thread_block();

    uint32_t image_id, tile_id, i0, j0;
    if(active_tiles != nullptr)
    {
        // compacted grid: one block per selected tile, decoded from the list
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
    constexpr uint32_t PX_ROWS = 16 / PX; // tile rows assigned per q (tile_size == 16)

    tile_offsets    += (int64_t)image_id * tile_height * tile_width;
    render_alphas   += (int64_t)image_id * image_height * image_width;
    last_ids        += (int64_t)image_id * image_height * image_width;
    v_render_colors += (int64_t)image_id * image_height * image_width * CDIM;
    v_render_alphas += (int64_t)image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }

    // per-pixel state for each of the PX pixels
    float px[PX], py[PX];
    int32_t pix_id[PX];
    bool inside[PX];
    float T_final[PX], T[PX];
    float buffer[PX][CDIM];
    float buffer_dot[PX];
    float tail_const[PX];
    float v_render_c[PX][CDIM];
    float v_render_a[PX];
    int32_t bin_final[PX];
    bool valid[PX];

#pragma unroll
    for(uint32_t q = 0; q < PX; ++q)
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
        float bg_dot = 0.f;
#pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            v_render_c[q][k] = v_render_colors[pix_id[q] * CDIM + k];
            if constexpr(!SCALAR_ADJOINT)
            {
                buffer[q][k] = 0.f;
            }
            else if(backgrounds != nullptr)
            {
                bg_dot += backgrounds[k] * v_render_c[q][k];
            }
        }
        v_render_a[q] = v_render_alphas[pix_id[q]];
        if constexpr(SCALAR_ADJOINT)
        {
            buffer_dot[q] = 0.f;
            tail_const[q] = T_final[q] * (v_render_a[q] - bg_dot);
        }
        if(backgrounds != nullptr && !skip_background_atomic)
        {
#pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                atomicAdd(v_backgrounds + k, T_final[q] * v_render_c[q][k]);
            }
        }
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
    int32_t warp_bin_final         = 0;
#pragma unroll
    for(uint32_t q = 0; q < PX; ++q)
    {
        warp_bin_final = max(warp_bin_final, bin_final[q]);
    }
    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

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

        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t)
        {
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            bool any_valid          = false;
            for(uint32_t q = 0; q < PX; ++q)
            {
                valid[q] = inside[q];
                if(batch_end - (int32_t)t > bin_final[q])
                {
                    valid[q] = false;
                }
                if(valid[q])
                {
                    const vec3 conic          = conic_batch[t];
                    const vec3 xy_opac        = xy_opacity_batch[t];
                    const float opac          = xy_opac.z;
                    const vec2 delta          = {xy_opac.x - px[q], xy_opac.y - py[q]};
                    const GaussianWeight gw   = eval_gaussian_weight(conic, delta.x, delta.y, opac);
                    const float vis           = gw.vis;
                    const float alpha         = gw.alpha;
                    if(!gw.valid)
                    {
                        valid[q] = false;
                    }
                    else
                    {
                        any_valid = true;
                        // rasterize_to_pixels_3dgs_blend_bwd ASSIGNS its
                        // v_*_local outputs (it is written for one call per
                        // (thread, isect)), so each q-pixel must use private
                        // scratch and the results are accumulated into the
                        // shared per-isect totals before the single warp
                        // reduction and atomic scatter below.
                        float v_rgb_q[CDIM] = {0.f};
                        vec3 v_conic_q      = {0.f, 0.f, 0.f};
                        vec2 v_xy_q         = {0.f, 0.f};
                        float v_opacity_q   = 0.f;
                        if constexpr(SCALAR_ADJOINT)
                        {
                            const float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                            T[q] *= ra;
                            const float fac = alpha * T[q];
                            float rgb_dot = 0.f;
#pragma unroll
                            for(uint32_t k = 0; k < CDIM; ++k)
                            {
                                v_rgb_q[k] = fac * v_render_c[q][k];
                                rgb_dot += rgbs_batch[t * CDIM + k] * v_render_c[q][k];
                            }
                            const float v_alpha = T[q] * rgb_dot
                                + ra * (tail_const[q] - buffer_dot[q]);
                            if(opac * vis <= MAX_ALPHA)
                            {
                                const float r = vis * v_alpha;
                                const float v_sigma = -opac * r;
                                if constexpr(H8_MR)
                                {
                                    // H8-0R: opacity-absorbed moments
                                    // Sx=v_sigma*dx, Sy=v_sigma*dy
                                    // Sxx=v_sigma*dx^2, Sxy=v_sigma*dx*dy, Syy=v_sigma*dy^2
                                    // (NO 0.5 — moved to reconstruction)
                                    v_xy_q = {v_sigma * delta.x,
                                              v_sigma * delta.y};
                                    v_conic_q = {v_sigma * delta.x * delta.x,
                                                 v_sigma * delta.x * delta.y,
                                                 v_sigma * delta.y * delta.y};
                                    v_opacity_q = r;
                                }
                                else
                                {
                                    v_conic_q = {0.5f * v_sigma * delta.x * delta.x,
                                                 v_sigma * delta.x * delta.y,
                                                 0.5f * v_sigma * delta.y * delta.y};
                                    v_xy_q = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                              v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                                    v_opacity_q = vis * v_alpha;
                                }
                            }
                            buffer_dot[q] += rgb_dot * fac;
                        }
                        else
                        {
                            rasterize_to_pixels_3dgs_blend_bwd<CDIM>(
                                conic,
                                delta,
                                opac,
                                vis,
                                alpha,
                                rgbs_batch + t * CDIM,
                                v_render_c[q],
                                v_render_a[q],
                                T_final[q],
                                backgrounds,
                                false, // compute_abs
                                T[q],
                                buffer[q],
                                v_rgb_q,
                                v_conic_q,
                                v_xy_q,
                                v_xy_q,
                                v_opacity_q
                            );
                        }
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
                        v_opacity_local += v_opacity_q;
                    }
                }
            }
            if(!warp.any(any_valid))
            {
                continue;
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t];
                float *v_rgb_ptr = v_colors + (int64_t)CDIM * g;
                float *v_conic_ptr = (float *)(v_conics) + 3 * (int64_t)g;
                float *v_xy_ptr = (float *)(v_means2d) + 2 * (int64_t)g;
#if P3_SINK_MODE != 0
                // Diagnostic sink: a per-warp UNCONTENDED slot in the scratch
                // buffer. Preserves all local arithmetic via the store dependency
                // while removing the global atomic accumulation / output traffic.
                const int64_t warp_slot = (int64_t)(cg::this_grid().thread_rank() / 32);
                float *sink = (float *)(v_p3h_scratch) + 16 * warp_slot;
                sink[11] = (float)g; // keep index-mapping alive
#endif
#if P3_SINK_MODE == 4
                // V4 WRITE_ONLY: full values, plain (non-atomic) stores.
                // Isolates atomic RMW/contention from plain output-store cost.
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k) sink[3 + k] = v_rgb_local[k];
                sink[0] = v_conic_local.x; sink[1] = v_conic_local.y; sink[2] = v_conic_local.z;
                sink[6] = v_xy_local.x; sink[7] = v_xy_local.y; sink[8] = v_opacity_local;
#else
#if (P3_SINK_MODE == 1) || (P3_SINK_MODE == 3)
                // appearance sink (B color + C opacity): fuse to one slot
                {
                    float fB = v_opacity_local;
#pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k) fB += v_rgb_local[k];
                    sink[9] = fB;
                }
#else
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    atomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }
                atomicAdd(v_opacities + g, v_opacity_local);
#endif
#if (P3_SINK_MODE == 1) || (P3_SINK_MODE == 2)
                // geometry sink (D H8 moments): fuse to one slot
                {
                    sink[10] = v_conic_local.x + v_conic_local.y + v_conic_local.z
                             + v_xy_local.x + v_xy_local.y;
                }
#else
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);
#endif
#endif
            }
        }
    }
}



// Background gradient (per-pixel reduction; only used when there are no
// intersections -- the blend kernel covers it otherwise).
// ============================================================================
__global__ void higs_background_bwd_kernel(
    const uint32_t n_pixels,
    const uint32_t color_dim,
    const float *__restrict__ render_alphas,   // [I, H, W]
    const float *__restrict__ v_render_colors, // [I, H, W, color_dim]
    float *__restrict__ v_backgrounds          // [color_dim]
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pixels)
    {
        return;
    }
    const float T_final = 1.0f - render_alphas[idx];
#pragma unroll
    for(uint32_t k = 0; k < color_dim; ++k)
    {
        atomicAdd(v_backgrounds + k, T_final * v_render_colors[idx * color_dim + k]);
    }
}

// ============================================================================
// Stage 2: projection VJP (classic pinhole, single-scene batch B == 1)
// ============================================================================
template<bool H8_MR = false>
__global__ void higs_projection_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t C,
    // forward inputs (FP32 master)
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ quats,    // [N, 4]
    const float *__restrict__ scales,   // [N, 3]
    const float *__restrict__ viewmats, // [1, C, 4, 4]
    const float *__restrict__ Ks,       // [1, C, 3, 3]
    const int64_t camera_model,
    const uint32_t image_width,
    const uint32_t image_height,
    // forward outputs
    const int32_t *__restrict__ radii, // [I, N, 2]
    const vec3 *__restrict__ conics,   // [I * N, 3] (forward captured)
    // gradient outputs (per camera)
    vec2 *__restrict__ v_means2d, // [I * N, 2] (non-const for H8_MR write-back)
    const vec3 *__restrict__ v_conics,  // [I * N, 3]
    // depth-channel gradient (optional): d(depth)/d(means) chain
    const float *__restrict__ v_colors_flat, // [I * N, color_dim]
    const int32_t color_dim,
    const int32_t depth_channel,             // -1 when no depth channel
    const int64_t *__restrict__ visible_ids, // [N] master row per visible gaussian
    // gradient inputs (FP32 master accumulators, written via visible_ids)
    float *__restrict__ v_means,  // [M, 3]
    float *__restrict__ v_quats,  // [M, 4]
    float *__restrict__ v_scales  // [M, 3]
)
{
    const int64_t idx   = cg::this_grid().thread_rank();
    const int64_t count = (int64_t)I * N;
    if(idx >= count)
    {
        return;
    }
    const uint32_t i   = idx / N;
    const uint32_t g   = idx % N;
    const uint32_t cid = i % C;
    const int32_t *r   = radii + idx * 2;
    if(r[0] <= 0 || r[1] <= 0)
    {
        return;
    }

    const float *view_ptr = viewmats + cid * 16;
    const float *K_ptr    = Ks + cid * 9;

    // vjp: compute the inverse of the 2d covariance
    const vec3 *conic_ptr = conics + idx;
    mat2 covar2d_inv      = mat2(conic_ptr->x, conic_ptr->y, conic_ptr->y, conic_ptr->z);
    const vec3 *v_conic_ptr = v_conics + idx;
    // H8-MR: reconstruct real gradients from opacity-absorbed moments
    float v_conic_x, v_conic_y, v_conic_z;
    vec2 v_means2d_val;
    if constexpr(H8_MR)
    {
        // Read raw moments from gradient buffers (already loaded)
        const float Sx  = v_means2d[idx].x;   // sum(v_sigma * dx)
        const float Sy  = v_means2d[idx].y;   // sum(v_sigma * dy)
        const float Sxx = v_conic_ptr->x;      // sum(v_sigma * dx^2)
        const float Sxy = v_conic_ptr->y;      // sum(v_sigma * dx*dy)
        const float Syy = v_conic_ptr->z;      // sum(v_sigma * dy^2)
        // Reconstruct using forward conic (A, B, C already loaded)
        v_means2d_val = {conic_ptr->x * Sx + conic_ptr->y * Sy,
                         conic_ptr->y * Sx + conic_ptr->z * Sy};
        v_conic_x = 0.5f * Sxx;
        v_conic_y = Sxy;
        v_conic_z = 0.5f * Syy;
        // Write back reconstructed v_means2d for Python/densification
        v_means2d[idx] = v_means2d_val;
    }
    else
    {
        v_conic_x = v_conic_ptr->x;
        v_conic_y = v_conic_ptr->y;
        v_conic_z = v_conic_ptr->z;
        v_means2d_val = v_means2d[idx];
    }
    mat2 v_covar2d_inv
        = mat2(v_conic_x, v_conic_y * 0.5f, v_conic_y * 0.5f, v_conic_z);
    mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    // transform Gaussian to camera space
    mat3 R = mat3(
        view_ptr[0], view_ptr[4], view_ptr[8], // 1st column
        view_ptr[1], view_ptr[5], view_ptr[9], // 2nd column
        view_ptr[2], view_ptr[6], view_ptr[10] // 3rd column
    );
    vec3 t = vec3(view_ptr[3], view_ptr[7], view_ptr[11]);

    // compute covariance from quaternions and scales
    const vec4 quat  = glm::make_vec4(quats + g * 4);
    const vec3 scale = glm::make_vec3(scales + g * 3);
    mat3 covar;
    quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    const vec3 mean_w = glm::make_vec3(means + g * 3);
    vec3 mean_c;
    posW2C(R, t, mean_w, mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // vjp: camera projection (pinhole / ortho / fisheye, matching the forward)
    const float fx = K_ptr[0], cx = K_ptr[2], fy = K_ptr[4], cy = K_ptr[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);
    if(camera_model == static_cast<int64_t>(CameraModelType::ORTHO))
    {
        ortho_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            v_means2d_val,
            v_mean_c,
            v_covar_c
        );
    }
    else if(camera_model == static_cast<int64_t>(CameraModelType::FISHEYE))
    {
        fisheye_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            v_means2d_val,
            v_mean_c,
            v_covar_c
        );
    }
    else
    {
        persp_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            v_means2d_val,
            v_mean_c,
            v_covar_c
        );
    }

    // depth channel VJP: depth = camera-space z of the Gaussian center
    // (projection depth from fully_fused_projection), so the blend-scaled
    // depth gradient chains into v_mean_c.z before posW2C_VJP.
    if(depth_channel >= 0)
    {
        v_mean_c.z += v_colors_flat[(int64_t)idx * color_dim + depth_channel];
    }

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(R, t, mean_w, v_mean_c, v_R, v_t, v_mean);
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    // write out results with warp-level reduction
    auto warp         = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, g);

    mat3 rotmat = quat_to_rotmat(quat);
    vec4 v_quat(0.f);
    vec3 v_scale(0.f);
    quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
    warpSum(v_mean, warp_group_g);
    warpSum(v_quat, warp_group_g);
    warpSum(v_scale, warp_group_g);
    if(warp_group_g.thread_rank() == 0)
    {
        const int64_t m_id = visible_ids[g];
        float *v_m = v_means + m_id * 3;
        gpuAtomicAdd(v_m, v_mean.x);
        gpuAtomicAdd(v_m + 1, v_mean.y);
        gpuAtomicAdd(v_m + 2, v_mean.z);
        float *v_q = v_quats + m_id * 4;
        gpuAtomicAdd(v_q, v_quat[0]);
        gpuAtomicAdd(v_q + 1, v_quat[1]);
        gpuAtomicAdd(v_q + 2, v_quat[2]);
        gpuAtomicAdd(v_q + 3, v_quat[3]);
        float *v_s = v_scales + m_id * 3;
        gpuAtomicAdd(v_s, v_scale[0]);
        gpuAtomicAdd(v_s + 1, v_scale[1]);
        gpuAtomicAdd(v_s + 2, v_scale[2]);
    }
}

// ============================================================================
// Stage 3: SH coefficient + direction VJP (degree 0..3), float specialization
// of gsplat's sh_coeffs_to_color_fast_vjp. Chained through the activation
// clamp_min(sph + 0.5, 0): v_sph = v_colors_eval * (colors_eval > 0).
// ============================================================================
__device__ void higs_sh_coeffs_to_color_fast_vjp(
    const uint32_t degree,
    const uint32_t D,
    const uint32_t c,
    const vec3 &dir,
    const float *coeffs, // [K, D]
    const float v_colors_local,
    float *v_coeffs, // [K, D]
    vec3 *v_dir      // [3] optional
)
{
    gpuAtomicAdd(&v_coeffs[c], 0.2820947917738781f * v_colors_local);
    if(degree < 1)
    {
        return;
    }
    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x     = dir.x * inorm;
    float y     = dir.y * inorm;
    float z     = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    gpuAtomicAdd(&v_coeffs[1 * D + c], -0.48860251190292f * y * v_colors_local);
    gpuAtomicAdd(&v_coeffs[2 * D + c], 0.48860251190292f * z * v_colors_local);
    gpuAtomicAdd(&v_coeffs[3 * D + c], -0.48860251190292f * x * v_colors_local);

    if(v_dir != nullptr)
    {
        v_x += -0.48860251190292f * coeffs[3 * D + c] * v_colors_local;
        v_y += -0.48860251190292f * coeffs[1 * D + c] * v_colors_local;
        v_z += 0.48860251190292f * coeffs[2 * D + c] * v_colors_local;
    }
    if(degree < 2)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float z2     = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1    = x * x - y * y;
    float fS1    = 2.f * x * y;
    float pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7   = fTmp0B * x;
    float pSH5   = fTmp0B * y;
    float pSH8   = 0.5462742152960395f * fC1;
    float pSH4   = 0.5462742152960395f * fS1;
    gpuAtomicAdd(&v_coeffs[4 * D + c], pSH4 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[5 * D + c], pSH5 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[6 * D + c], pSH6 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[7 * D + c], pSH7 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[8 * D + c], pSH8 * v_colors_local);

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y, pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if(v_dir != nullptr)
    {
        fTmp0B_z = -1.092548430592079f;
        fC1_x    = 2.f * x;
        fC1_y    = -2.f * y;
        fS1_x    = 2.f * y;
        fS1_y    = 2.f * x;
        pSH6_z   = 2.f * 0.9461746957575601f * z;
        pSH7_x   = fTmp0B;
        pSH7_z   = fTmp0B_z * x;
        pSH5_y   = fTmp0B;
        pSH5_z   = fTmp0B_z * y;
        pSH8_x   = 0.5462742152960395f * fC1_x;
        pSH8_y   = 0.5462742152960395f * fC1_y;
        pSH4_x   = 0.5462742152960395f * fS1_x;
        pSH4_y   = 0.5462742152960395f * fS1_y;

        v_x += v_colors_local
             * (pSH4_x * coeffs[4 * D + c]
                + pSH8_x * coeffs[8 * D + c]
                + pSH7_x * coeffs[7 * D + c]);
        v_y += v_colors_local
             * (pSH4_y * coeffs[4 * D + c]
                + pSH8_y * coeffs[8 * D + c]
                + pSH5_y * coeffs[5 * D + c]);
        v_z += v_colors_local
             * (pSH6_z * coeffs[6 * D + c]
                + pSH7_z * coeffs[7 * D + c]
                + pSH5_z * coeffs[5 * D + c]);
    }

    if(degree < 3)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2    = x * fC1 - y * fS1;
    float fS2    = x * fS1 + y * fC1;
    float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13  = fTmp0C * x;
    float pSH11  = fTmp0C * y;
    float pSH14  = fTmp1B * fC1;
    float pSH10  = fTmp1B * fS1;
    float pSH15  = -0.5900435899266435f * fC2;
    float pSH9   = -0.5900435899266435f * fS2;
    gpuAtomicAdd(&v_coeffs[9 * D + c], pSH9 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[10 * D + c], pSH10 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[11 * D + c], pSH11 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[12 * D + c], pSH12 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[13 * D + c], pSH13 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[14 * D + c], pSH14 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[15 * D + c], pSH15 * v_colors_local);

    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x, pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y,
        pSH14_z, pSH10_x, pSH10_y, pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if(v_dir != nullptr)
    {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x    = fC1 + x * fC1_x - y * fS1_x;
        fC2_y    = x * fC1_y - fS1 - y * fS1_y;
        fS2_x    = fS1 + x * fS1_x + y * fC1_x;
        fS2_y    = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z  = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x  = fTmp0C;
        pSH13_z  = fTmp0C_z * x;
        pSH11_y  = fTmp0C;
        pSH11_z  = fTmp0C_z * y;
        pSH14_x  = fTmp1B * fC1_x;
        pSH14_y  = fTmp1B * fC1_y;
        pSH14_z  = fTmp1B_z * fC1;
        pSH10_x  = fTmp1B * fS1_x;
        pSH10_y  = fTmp1B * fS1_y;
        pSH10_z  = fTmp1B_z * fS1;
        pSH15_x  = -0.5900435899266435f * fC2_x;
        pSH15_y  = -0.5900435899266435f * fC2_y;
        pSH9_x   = -0.5900435899266435f * fS2_x;
        pSH9_y   = -0.5900435899266435f * fS2_y;

        v_x += v_colors_local
             * (pSH9_x * coeffs[9 * D + c]
                + pSH15_x * coeffs[15 * D + c]
                + pSH10_x * coeffs[10 * D + c]
                + pSH14_x * coeffs[14 * D + c]
                + pSH13_x * coeffs[13 * D + c]);

        v_y += v_colors_local
             * (pSH9_y * coeffs[9 * D + c]
                + pSH15_y * coeffs[15 * D + c]
                + pSH10_y * coeffs[10 * D + c]
                + pSH14_y * coeffs[14 * D + c]
                + pSH11_y * coeffs[11 * D + c]);

        v_z += v_colors_local
             * (pSH12_z * coeffs[12 * D + c]
                + pSH13_z * coeffs[13 * D + c]
                + pSH11_z * coeffs[11 * D + c]
                + pSH14_z * coeffs[14 * D + c]
                + pSH10_z * coeffs[10 * D + c]);
    }

    if(degree < 4)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }
}

// ============================================================================
// Stage 3: SH coefficient + direction VJP over a FIXED grid of I * N * D
// threads with a per-pair radii mask (std-shaped like gsplat's
// spherical_harmonics_bwd). A fixed grid avoids the visible-pair compaction
// (per-block counts + exclusive scan + packed pair list) and, more
// importantly, the device->host readback that used to size the dynamic grid:
// the VJP now launches back-to-back with the projection VJP. Masked
// (invisible) threads exit before any memory reads, exactly like std's
// masked SH backward.
// ============================================================================
// Precompute camera world positions cam_pos = -R^t t once per camera so the
// SH VJP grid kernel does not re-derive the mat3 transpose per thread.
__global__ void higs_camera_positions_kernel(
    const float *__restrict__ viewmats, // [1, C, 4, 4] row-major
    float *__restrict__ cam_positions,  // [C, 3]
    const uint32_t C
)
{
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= C)
    {
        return;
    }
    const float *v = viewmats + c * 16;
    const float tx = v[3];
    const float ty = v[7];
    const float tz = v[11];
    float *p       = cam_positions + c * 3;
    p[0]           = -(v[0] * tx + v[4] * ty + v[8] * tz);
    p[1]           = -(v[1] * tx + v[5] * ty + v[9] * tz);
    p[2]           = -(v[2] * tx + v[6] * ty + v[10] * tz);
}

__global__ void higs_sh_vjp_grid_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t C,
    const uint32_t degree,
    const uint32_t K,
    const uint32_t D,
    const uint32_t stride,                    // colors_eval row stride (== D for RGB)
    const int32_t *__restrict__ radii,        // [I, N, 2]
    const float *__restrict__ means,          // [N, 3]
    const float *__restrict__ cam_positions,  // [C, 3]
    const float *__restrict__ coeffs,         // [N, K, D]
    const float *__restrict__ colors_eval,    // [I * N, D] forward-captured
    const float *__restrict__ v_colors_eval,  // [I * N, D]
    const int64_t *__restrict__ visible_ids,  // [N] master row per visible gaussian
    float *__restrict__ v_coeffs,             // [M, K, D] master
    float *__restrict__ v_means               // [M, 3] master
)
{
    const int64_t t   = cg::this_grid().thread_rank();
    const int64_t num = static_cast<int64_t>(I) * N * D;
    if(t >= num)
    {
        return;
    }
    // One thread per (camera, gaussian, channel), gaussian fastest within a
    // camera and the D channels of a gaussian adjacent (matching gsplat's SH
    // bwd) so the atomic coefficient updates of one Gaussian stay in the
    // same warp/cache lines.
    const uint32_t elem = static_cast<uint32_t>(t / D);
    const uint32_t g    = elem % N;
    const uint32_t i    = elem / N;
    const uint32_t c    = static_cast<uint32_t>(t % D);
    const int32_t *r    = radii + ((int64_t)i * N + g) * 2;
    const bool masked   = !(r[0] > 0 && r[1] > 0); // invisible in this camera
    const uint32_t cid  = i % C;
    const int64_t idx   = (int64_t)i * N + g;
    const int64_t m_id  = visible_ids[g];

    vec3 v_dir(0.f);
    if(!masked)
    {
        const float *cp = cam_positions + cid * 3;
        const vec3 cam_pos(cp[0], cp[1], cp[2]);
        const vec3 dir = glm::make_vec3(means + g * 3) - cam_pos;

        const float *coeff_ptr    = coeffs + (int64_t)g * K * D;
        const float *col_eval_ptr = colors_eval + idx * stride;
        const float *v_col_ptr    = v_colors_eval + idx * stride;
        float *v_coeff_ptr        = v_coeffs + m_id * K * D;
        // activation chain: colors_eval = clamp_min(sph + 0.5, 0), so
        // d(colors_eval)/d(sph) = 1{colors_eval > 0} (mask from the FORWARD
        // evaluated colors, never from the gradient sign).
        const float v_sph = col_eval_ptr[c] > 0.f ? v_col_ptr[c] : 0.f;
        higs_sh_coeffs_to_color_fast_vjp(degree, D, c, dir, coeff_ptr, v_sph, v_coeff_ptr, &v_dir);
    }

    // dir = means - cam_pos -> d(loss)/d(means) += v_dir. The D channel lanes
    // of one (camera, gaussian) are consecutive (c == t % D), so the three
    // partials reduce with two shuffle-downs and one atomicAdd per output
    // coordinate instead of three serialized same-address atomicAdds. Partial
    // warps (grid tail) and groups that straddle a warp boundary fall back to
    // the per-lane atomics; masked lanes contribute v_dir = 0 to the shuffle.
    const uint32_t lane       = static_cast<uint32_t>(t) & 31u;
    const uint32_t warp_start = static_cast<uint32_t>(t) & ~31u;
    const bool full_warp      = (static_cast<int64_t>(warp_start) + 32 <= num);
    const int32_t group_lane  = static_cast<int32_t>(lane) - static_cast<int32_t>(c);
    const bool complete_group = full_warp && group_lane >= 0 && (group_lane + 2 <= 31);

    vec3 v_dir_sum = v_dir;
    if(full_warp)
    {
        v_dir_sum.x += __shfl_down_sync(0xffffffffu, v_dir.x, 1);
        v_dir_sum.y += __shfl_down_sync(0xffffffffu, v_dir.y, 1);
        v_dir_sum.z += __shfl_down_sync(0xffffffffu, v_dir.z, 1);
        v_dir_sum.x += __shfl_down_sync(0xffffffffu, v_dir.x, 2);
        v_dir_sum.y += __shfl_down_sync(0xffffffffu, v_dir.y, 2);
        v_dir_sum.z += __shfl_down_sync(0xffffffffu, v_dir.z, 2);
    }

    if(!masked)
    {
        if(c == 0 && complete_group)
        {
            atomicAdd(v_means + m_id * 3 + 0, v_dir_sum.x);
            atomicAdd(v_means + m_id * 3 + 1, v_dir_sum.y);
            atomicAdd(v_means + m_id * 3 + 2, v_dir_sum.z);
        }
        else if(!complete_group)
        {
            atomicAdd(v_means + m_id * 3 + 0, v_dir.x);
            atomicAdd(v_means + m_id * 3 + 1, v_dir.y);
            atomicAdd(v_means + m_id * 3 + 2, v_dir.z);
        }
        // else: c != 0 lane of a complete group; its v_dir is already in the
        // leader's shuffled sum.
    }
}
// Per-gaussian reduction across camera views (RGB-mode colors + opacities).
// ============================================================================
__global__ void higs_reduce_master_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t D,
    const uint32_t stride,            // v_flat row stride (== color_dim)
    const float *__restrict__ v_flat, // [I * N, stride]
    const int64_t *__restrict__ visible_ids, // [N] master row per visible gaussian
    float *__restrict__ v_master      // [M, D]
)
{
    const int64_t idx   = cg::this_grid().thread_rank();
    const int64_t total = (int64_t)N * D;
    if(idx >= total)
    {
        return;
    }
    const uint32_t g = idx / D;
    const uint32_t d = idx % D;
    float acc        = 0.f;
    for(uint32_t i = 0; i < I; ++i)
    {
        acc += v_flat[((int64_t)i * N + g) * stride + d];
    }
    v_master[(int64_t)visible_ids[g] * D + d] = acc;
}

// ============================================================================
// Host launcher
// ============================================================================
void check_float_cuda(const at::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_int_cuda(const at::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_long_cuda(const at::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
} // namespace

std::tuple<
    at::Tensor, // v_means [N, 3]
    at::Tensor, // v_quats [N, 4]
    at::Tensor, // v_scales [N, 3]
    at::Tensor, // v_opacities [N]
    at::Tensor, // v_colors_master
    at::Tensor, // v_backgrounds [3]
    at::Tensor  // v_means2d [I*N, 2]
    >
higs_rasterize_backward(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors_eval,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    const at::optional<at::Tensor> &active_tiles, // [AT] int32 selected tile ids or nullopt
    const at::Tensor &render_alphas,
    const at::Tensor &last_ids,
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &radii,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t width,
    int64_t height,
    int64_t tile_size,
    double eps2d,
    int64_t camera_model,
    const at::Tensor &v_render_colors,
    const at::Tensor &v_render_alphas,
    const at::optional<at::Tensor> &sh_coeffs,
    int64_t sh_degree,
    const at::Tensor &visible_ids,  // [N] int64 ascending master indices
    const at::Tensor &grad_means,   // [M, 3] pre-zeroed master gradients
    const at::Tensor &grad_quats,   // [M, 4]
    const at::Tensor &grad_scales,  // [M, 3]
    const at::Tensor &grad_opacities, // [M]
    const at::Tensor &grad_colors   // [M, K, D] (SH) or [M, D] (RGB)
)
{
    check_float_cuda(means2d, "means2d");
    check_float_cuda(conics, "conics");
    check_float_cuda(colors_eval, "colors_eval");
    check_float_cuda(opacities, "opacities");
    check_int_cuda(tile_offsets, "tile_offsets");
    check_int_cuda(flatten_ids, "flatten_ids");
    if(active_tiles.has_value())
    {
        check_int_cuda(active_tiles.value(), "active_tiles");
        TORCH_CHECK(active_tiles.value().dim() == 1, "active_tiles must be 1-D");
    }
    check_float_cuda(render_alphas, "render_alphas");
    check_int_cuda(last_ids, "last_ids");
    check_float_cuda(means, "means");
    check_float_cuda(quats, "quats");
    check_float_cuda(scales, "scales");
    check_int_cuda(radii, "radii");
    check_float_cuda(viewmats, "viewmats");
    check_float_cuda(Ks, "Ks");
    check_float_cuda(v_render_colors, "v_render_colors");
    check_float_cuda(v_render_alphas, "v_render_alphas");
    if(backgrounds.has_value())
    {
        check_float_cuda(backgrounds.value(), "backgrounds");
    }
    if(sh_coeffs.has_value())
    {
        check_float_cuda(sh_coeffs.value(), "sh_coeffs");
    }
    check_long_cuda(visible_ids, "visible_ids");
    check_float_cuda(grad_means, "grad_means");
    check_float_cuda(grad_quats, "grad_quats");
    check_float_cuda(grad_scales, "grad_scales");
    check_float_cuda(grad_opacities, "grad_opacities");
    check_float_cuda(grad_colors, "grad_colors");

    const int64_t B = viewmats.size(0);
    const int64_t C = viewmats.size(1);
    const int64_t I = render_alphas.size(0);
    const int64_t N = means.size(0);
    TORCH_CHECK(visible_ids.size(0) == N, "visible_ids size ", visible_ids.size(0),
        " does not match visible-subset N ", N);
    TORCH_CHECK(B == 1, "higs_native backward requires a single scene batch (B == 1), got ", B);
    TORCH_CHECK(I == B * C, "render_alphas batch size does not match viewmats");
    TORCH_CHECK(
        radii.dim() == 3 && radii.size(0) == I && radii.size(1) == N && radii.size(2) == 2,
        "bad radii shape"
    );
    const int64_t color_dim = colors_eval.size(-1);
    TORCH_CHECK(
        color_dim == 1 || color_dim == 3 || color_dim == 4,
        "higs_native backward supports 1 (depth-only), 3 (RGB) or 4 (RGB+depth) "
        "channels, got ",
        color_dim
    );
    TORCH_CHECK(
        camera_model == static_cast<int64_t>(CameraModelType::PINHOLE)
            || camera_model == static_cast<int64_t>(CameraModelType::ORTHO)
            || camera_model == static_cast<int64_t>(CameraModelType::FISHEYE),
        "higs_native backward supports pinhole/ortho/fisheye camera models, got ",
        camera_model
    );
    TORCH_CHECK(
        v_render_colors.size(-1) == color_dim,
        "bad v_render_colors channel count: ",
        v_render_colors.size(-1),
        " != colors_eval ",
        color_dim
    );
    if(sh_coeffs.has_value())
    {
        TORCH_CHECK(
            sh_degree >= 0 && sh_degree <= 3,
            "higs_native SH backward supports sh_degree 0..3, got ",
            sh_degree
        );
    }

    const at::Device device = means.device();
    const c10::cuda::CUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    auto opts   = at::TensorOptions().device(device).dtype(at::kFloat);

    at::Tensor v_backgrounds = at::zeros({color_dim}, opts);
    at::Tensor v_means2d     = at::zeros({I * N, 2}, opts);
    at::Tensor v_conics_out  = at::zeros({I * N, 3}, opts);
    at::Tensor v_colors_flat = at::zeros({I * N, color_dim}, opts);
    at::Tensor v_opacities_flat = at::zeros({I * N}, opts);

    const int64_t tile_h   = tile_offsets.size(-2);
    const int64_t tile_w   = tile_offsets.size(-1);
    const int64_t n_isects = flatten_ids.size(0);

    // Round 39: optional active-tile compaction. When the forward ran with a
    // tile mask, the blend grid is launched only over the selected tiles (and
    // the per-pixel background gradient is moved to a separate all-pixel
    // kernel so active pixels are not double-counted). Dense callers pass
    // nullopt and keep the original full-grid path byte-for-byte unchanged.
    const int32_t *active_tiles_ptr = nullptr;
    uint32_t n_active_tiles = 0;
    if(active_tiles.has_value())
    {
        active_tiles_ptr = active_tiles.value().const_data_ptr<int32_t>();
        n_active_tiles   = static_cast<uint32_t>(active_tiles.value().numel());
    }
    const bool compacted = (active_tiles_ptr != nullptr);
    const char *scalar_adjoint_env = std::getenv("HIGS_BWD_SCALAR_ADJOINT");
    const char *scalar_adjoint_name =
        (scalar_adjoint_env == nullptr) ? "baseline" : scalar_adjoint_env;
    const bool scalar_adjoint = strcmp(scalar_adjoint_name, "scalar_adjoint") == 0;
    TORCH_CHECK(
        scalar_adjoint || strcmp(scalar_adjoint_name, "baseline") == 0,
        "unsupported HIGS_BWD_SCALAR_ADJOINT: ",
        scalar_adjoint_name
    );
    // H8-MR: opacity-absorbed moment-space geometry adjoint (requires SCALAR_ADJOINT)
    const char *h8_mr_env = std::getenv("HIGS_BWD_H8_MR");
    const bool h8_mr = (h8_mr_env != nullptr) && (strcmp(h8_mr_env, "1") == 0);
    TORCH_CHECK(
        !h8_mr || scalar_adjoint,
        "HIGS_BWD_H8_MR=1 requires HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint"
    );
    // Channel layout of the captured colors_eval: 1 = depth-only, 3 = RGB,
    // 4 = RGB + depth (depth is the LAST channel). -1 means no depth channel.
    const int32_t depth_channel = (color_dim == 4) ? 3 : (color_dim == 1) ? 0 : -1;

    // ---- Stage 1: pixel-blend backward ----
    if(n_isects > 0)
    {
        const uint32_t threads_dim = static_cast<uint32_t>(tile_size);
        dim3 threads(threads_dim, threads_dim, 1);
        dim3 grid;
        if(active_tiles_ptr != nullptr)
        {
            // one block per selected tile
            grid = dim3(n_active_tiles, 1, 1);
        }
        else
        {
            grid = dim3(
                static_cast<uint32_t>(I),
                static_cast<uint32_t>(tile_h),
                static_cast<uint32_t>(tile_w)
            );
        }
#if P3_SINK_MODE != 0
        // Diagnostic scratch: 16 floats per warp, capped at 8 warps/block.
        const int64_t p3h_n_blk = (int64_t)grid.x * (int64_t)grid.y * (int64_t)grid.z;
        at::Tensor v_p3h_scratch = at::zeros({p3h_n_blk * 128}, opts);
#endif
#define HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE, H8_MR_VALUE)                                    \
    do                                                                                             \
    {                                                                                              \
        C10_CUDA_CHECK(cudaFuncSetAttribute(higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE),          \
                                                (SCALAR_VALUE), (H8_MR_VALUE)>,                                   \
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));          \
        higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE), (SCALAR_VALUE), (H8_MR_VALUE)><<<grid, threads_px,          \
            static_cast<size_t>(shmem_size), stream>>>(                                           \
            static_cast<uint32_t>(I), static_cast<uint32_t>(N), static_cast<uint32_t>(n_isects),  \
            reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),                      \
            reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),                       \
            colors_eval.const_data_ptr<float>(), opacities.const_data_ptr<float>(),               \
            backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,      \
            static_cast<uint32_t>(tile_w), static_cast<uint32_t>(tile_h),                         \
            active_tiles_ptr, compacted,                                                          \
            static_cast<uint32_t>(width), static_cast<uint32_t>(height),                          \
            static_cast<uint32_t>(tile_size), tile_offsets.const_data_ptr<int32_t>(),             \
            flatten_ids.const_data_ptr<int32_t>(), render_alphas.const_data_ptr<float>(),         \
            last_ids.const_data_ptr<int32_t>(), v_render_colors.const_data_ptr<float>(),          \
            v_render_alphas.const_data_ptr<float>(),                                              \
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),                               \
            reinterpret_cast<vec3 *>(v_conics_out.data_ptr<float>()),                            \
            v_colors_flat.data_ptr<float>(), v_opacities_flat.data_ptr<float>(),                  \
            v_backgrounds.data_ptr<float>()
#if P3_SINK_MODE != 0
            , v_p3h_scratch.data_ptr<float>()
#endif
            );                                                     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                            \
    } while(false)

#define LAUNCH_BLEND_BWD(CDIM)                                                                     \
    {                                                                                              \
        const char *px_env = std::getenv("HIGS_PX_RUNTIME");                                       \
        const int64_t px_use = (px_env != nullptr) ? atoi(px_env) : 2;                             \
        const int64_t px_safe = (px_use == 0) ? 0 : (px_use == 1) ? 1 : (px_use == 2) ? 2 :         \
                                (px_use == 4) ? 4 : (px_use == 8) ? 8 : 2;                          \
        const int64_t thrs = (tile_size * tile_size) / (px_safe == 0 ? 1 : px_safe);               \
        const int64_t shmem_size = thrs                                                             \
            * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * (CDIM));             \
        dim3 threads_px(static_cast<uint32_t>(tile_size),                                           \
                        static_cast<uint32_t>(thrs / tile_size), 1);                                \
        if(px_safe == 0)                                                                            \
        {                                                                                           \
            C10_CUDA_CHECK(cudaFuncSetAttribute(higs_blend_bwd_kernel<(CDIM)>,                      \
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));        \
            higs_blend_bwd_kernel<(CDIM)><<<grid, threads_px, static_cast<size_t>(shmem_size), stream>>>( \
                static_cast<uint32_t>(I), static_cast<uint32_t>(N), static_cast<uint32_t>(n_isects),\
                reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),                    \
                reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),                     \
                colors_eval.const_data_ptr<float>(), opacities.const_data_ptr<float>(),             \
                backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,    \
                static_cast<uint32_t>(tile_w), static_cast<uint32_t>(tile_h),                       \
                active_tiles_ptr, compacted,                                                        \
                static_cast<uint32_t>(width), static_cast<uint32_t>(height),                        \
                static_cast<uint32_t>(tile_size), tile_offsets.const_data_ptr<int32_t>(),           \
                flatten_ids.const_data_ptr<int32_t>(), render_alphas.const_data_ptr<float>(),       \
                last_ids.const_data_ptr<int32_t>(), v_render_colors.const_data_ptr<float>(),        \
                v_render_alphas.const_data_ptr<float>(),                                            \
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),                              \
                reinterpret_cast<vec3 *>(v_conics_out.data_ptr<float>()),                           \
                v_colors_flat.data_ptr<float>(), v_opacities_flat.data_ptr<float>(),                \
                v_backgrounds.data_ptr<float>());                                                   \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                         \
        }                                                                                           \
        else if(px_safe == 1)                                                                       \
        {                                                                                           \
            if(scalar_adjoint && h8_mr) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, true, true);                            \
            else if(scalar_adjoint) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, true, false);                            \
            else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 1, false, false);                                         \
        }                                                                                           \
        else if(px_safe == 2)                                                                       \
        {                                                                                           \
            if(scalar_adjoint && h8_mr) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, true, true);                            \
            else if(scalar_adjoint) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, true, false);                            \
            else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 2, false, false);                                         \
        }                                                                                           \
        else if(px_safe == 4)                                                                       \
        {                                                                                           \
            if(scalar_adjoint && h8_mr) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, true, true);                            \
            else if(scalar_adjoint) HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, true, false);                            \
            else HIGS_LAUNCH_BLEND_BWD_PX(CDIM, 4, false, false);                                         \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            TORCH_CHECK(false, "unsupported HIGS_PX_RUNTIME");                                      \
        }                                                                                           \
    }

        switch(color_dim)
        {
        case 1:
            LAUNCH_BLEND_BWD(1);
            break;
        case 3:
            LAUNCH_BLEND_BWD(3);
            break;
        default: // 4
            LAUNCH_BLEND_BWD(4);
            break;
        }
#undef LAUNCH_BLEND_BWD
#undef HIGS_LAUNCH_BLEND_BWD_PX

        if(compacted && backgrounds.has_value())
        {
            // The compacted blend grid skips the masked tiles' per-pixel
            // background gradient (skip_background_atomic), so every pixel's
            // d(render)/d(background) = T_final must be accumulated here. For
            // masked tiles render == background (T_final == 1), which also
            // covers LPIPS-style losses that backprop through all pixels.
            const int64_t n_pixels = I * height * width;
            const uint32_t threads = 256;
            const uint32_t bg_grid = static_cast<uint32_t>((n_pixels + threads - 1) / threads);
            higs_background_bwd_kernel<<<bg_grid, threads, 0, stream>>>(static_cast<uint32_t>(n_pixels), static_cast<uint32_t>(color_dim), render_alphas.const_data_ptr<float>(), v_render_colors.const_data_ptr<float>(), v_backgrounds.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    else if(backgrounds.has_value())
    {
        // No intersections: render == background everywhere.
        const int64_t n_pixels = I * height * width;
        const uint32_t threads = 256;
        const uint32_t grid    = static_cast<uint32_t>((n_pixels + threads - 1) / threads);
        higs_background_bwd_kernel<<<grid, threads, 0, stream>>>(
            static_cast<uint32_t>(n_pixels),
            static_cast<uint32_t>(color_dim),
            render_alphas.const_data_ptr<float>(),
            v_render_colors.const_data_ptr<float>(),
            v_backgrounds.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- Stage 2: projection VJP ----
    {
        const int64_t total    = I * N;
        const uint32_t threads = 256;
        const uint32_t grid    = static_cast<uint32_t>((total + threads - 1) / threads);
        if(h8_mr)
        {
            higs_projection_bwd_kernel<true><<<grid, threads, 0, stream>>>(
                static_cast<uint32_t>(I),
                static_cast<uint32_t>(N),
                static_cast<uint32_t>(C),
                means.const_data_ptr<float>(),
                quats.const_data_ptr<float>(),
                scales.const_data_ptr<float>(),
                viewmats.const_data_ptr<float>(),
                Ks.const_data_ptr<float>(),
                camera_model,
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height),
                radii.const_data_ptr<int32_t>(),
                reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<const vec3 *>(v_conics_out.const_data_ptr<float>()),
                v_colors_flat.const_data_ptr<float>(),
                static_cast<int32_t>(color_dim),
                depth_channel,
                visible_ids.const_data_ptr<int64_t>(),
                grad_means.data_ptr<float>(),
                grad_quats.data_ptr<float>(),
                grad_scales.data_ptr<float>()
            );
        }
        else
        {
            higs_projection_bwd_kernel<false><<<grid, threads, 0, stream>>>(
                static_cast<uint32_t>(I),
                static_cast<uint32_t>(N),
                static_cast<uint32_t>(C),
                means.const_data_ptr<float>(),
                quats.const_data_ptr<float>(),
                scales.const_data_ptr<float>(),
                viewmats.const_data_ptr<float>(),
                Ks.const_data_ptr<float>(),
                camera_model,
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height),
                radii.const_data_ptr<int32_t>(),
                reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<const vec3 *>(v_conics_out.const_data_ptr<float>()),
                v_colors_flat.const_data_ptr<float>(),
                static_cast<int32_t>(color_dim),
                depth_channel,
                visible_ids.const_data_ptr<int64_t>(),
                grad_means.data_ptr<float>(),
                grad_quats.data_ptr<float>(),
                grad_scales.data_ptr<float>()
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

        if(sh_coeffs.has_value() && color_dim >= 3)
    {
        // ---- Stage 3a: SH VJP (depth-only modes have no color channels;
        // their SH/color input gradients are identically zero) ----
        // Fixed grid over I * N * D with a per-pair radii mask: no visible-
        // pair compaction and no device->host readback to size the grid, so
        // this launches back-to-back with the projection VJP.
        const int64_t K = sh_coeffs.value().size(-2);
        const int64_t D = sh_coeffs.value().size(-1);
        const int64_t total = I * N * D;
        const uint32_t threads = 256;
        const uint32_t grid = static_cast<uint32_t>((total + threads - 1) / threads);
        at::Tensor cam_positions = at::empty({static_cast<int64_t>(C), 3}, viewmats.options());
        {
            const uint32_t threads = 64;
            const uint32_t grid    = (C + threads - 1) / threads;
            higs_camera_positions_kernel<<<grid, threads, 0, stream>>>(
                viewmats.const_data_ptr<float>(),
                cam_positions.data_ptr<float>(),
                C
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        higs_sh_vjp_grid_kernel<<<grid, threads, 0, stream>>>(

            static_cast<uint32_t>(I),
            static_cast<uint32_t>(N),
            static_cast<uint32_t>(C),
            static_cast<uint32_t>(sh_degree),
            static_cast<uint32_t>(K),
            static_cast<uint32_t>(D),
            static_cast<uint32_t>(color_dim),
            radii.const_data_ptr<int32_t>(),
            means.const_data_ptr<float>(),
            cam_positions.data_ptr<float>(),
            sh_coeffs.value().const_data_ptr<float>(),
            colors_eval.const_data_ptr<float>(),
            v_colors_flat.const_data_ptr<float>(),
            visible_ids.const_data_ptr<int64_t>(),
            grad_colors.data_ptr<float>(),
            grad_means.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
    {
        // ---- Stage 3b: RGB mode - reduce evaluated-color grads over views ----
        if(color_dim >= 3)
        {
            const int64_t total    = N * kHigsColorDim;
            const uint32_t threads = 256;
            const uint32_t grid    = static_cast<uint32_t>((total + threads - 1) / threads);
            higs_reduce_master_kernel<<<grid, threads, 0, stream>>>(
                static_cast<uint32_t>(I),
                static_cast<uint32_t>(N),
                kHigsColorDim,
                static_cast<uint32_t>(color_dim),
                v_colors_flat.const_data_ptr<float>(),
                visible_ids.const_data_ptr<int64_t>(),
                grad_colors.data_ptr<float>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // ---- reduce opacity grads over views ----
    {
        const int64_t total    = N;
        const uint32_t threads = 256;
        const uint32_t grid    = static_cast<uint32_t>((total + threads - 1) / threads);
        higs_reduce_master_kernel<<<grid, threads, 0, stream>>>(
            static_cast<uint32_t>(I),
            static_cast<uint32_t>(N),
            1,
            1, // stride
            v_opacities_flat.const_data_ptr<float>(),
            visible_ids.const_data_ptr<int64_t>(),
            grad_opacities.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    (void)eps2d;
    return {
        grad_means,
        grad_quats,
        grad_scales,
        grad_opacities,
        grad_colors,
        v_backgrounds,
        v_means2d
    };
}
} // namespace gaussian_render_inference_scene
} // namespace gsplat
