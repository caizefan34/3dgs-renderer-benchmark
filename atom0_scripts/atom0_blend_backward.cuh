#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "sh_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cstdint>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::backward {

    __global__ void blend_backward_atom0_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        const float* __restrict__ grad_image,
        const float* __restrict__ image,
        const float* __restrict__ tile_final_transmittances,
        const uint* __restrict__ tile_max_n_processed,
        const uint* __restrict__ tile_n_processed,
        const uint* __restrict__ bucket_tile_index,
        const float4* __restrict__ bucket_color_transmittance,
        float2* __restrict__ grad_mean2d,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_opacity,
        float3* __restrict__ grad_colors,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width,
        const uint variant,
        const uint chunk_first_bucket,
        const int chunk_base,
        float* __restrict__ private_grad,
        float* __restrict__ partial_out)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        const uint bucket_idx = chunk_first_bucket + block.group_index().x;
        const uint lane_idx = warp.thread_rank();

        // chunk_base_dev = instance offset of the first bucket's tile within this chunk.
        // Computed on-device: the host launcher cannot dereference device pointers.
        // (chunk_base param is unused/placeholder; recomputed here from device pointers.)
        const int chunk_base_dev = static_cast<int>(tile_instance_ranges[bucket_tile_index[chunk_first_bucket]].x);

        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        if (tile_bucket_idx * 32 >= tile_max_n_processed[tile_idx]) return;

        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;
        const int slot = instance_idx - chunk_base_dev;

        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};
        float3 conic = {0.0f, 0.0f, 0.0f};
        float opacity = 0.0f;
        float3 color = {0.0f, 0.0f, 0.0f};
        float3 color_grad_factor = {0.0f, 0.0f, 0.0f};
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
            conic = make_float3(conic_opacity);
            opacity = conic_opacity.w;
            const float3 color_unclamped = primitive_color[primitive_idx];
            color = fmaxf(color_unclamped, 0.0f);
            if (color_unclamped.x >= 0.0f) color_grad_factor.x = 1.0f;
            if (color_unclamped.y >= 0.0f) color_grad_factor.y = 1.0f;
            if (color_unclamped.z >= 0.0f) color_grad_factor.z = 1.0f;
        }

        const float3 background = bg_color[0];
        const uint n_pixels = width * height;

        float2 dL_dmean2d_accum = {0.0f, 0.0f};
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
        float dL_dopacity_accum = 0.0f;
        float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        uint last_contributor;
        float3 color_pixel_after;
        float transmittance;
        float3 grad_color_pixel;
        float grad_alpha_common;

        bucket_color_transmittance += bucket_idx * config::block_size_blend;
        __shared__ uint collected_last_contributor[32];
        __shared__ float4 collected_color_pixel_after_transmittance[32];
        __shared__ float4 collected_grad_info_pixel[32];

        #pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                if (local_idx < config::block_size_blend) {
                    const float4 color_transmittance = bucket_color_transmittance[local_idx];
                    const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                    const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                    float3 color_pixel_w_bg, grad_color_pixel;
                    if (pixel_coords.x < width && pixel_coords.y < height) {
                        color_pixel_w_bg = make_float3(
                            image[pixel_idx],
                            image[n_pixels + pixel_idx],
                            image[2 * n_pixels + pixel_idx]
                        );
                        grad_color_pixel = make_float3(
                            grad_image[pixel_idx],
                            grad_image[n_pixels + pixel_idx],
                            grad_image[2 * n_pixels + pixel_idx]
                        );
                    }
                    const float final_transmittance = tile_final_transmittances[pixel_idx];
                    collected_color_pixel_after_transmittance[lane_idx] = make_float4(
                        color_pixel_w_bg - final_transmittance * background - make_float3(color_transmittance),
                        color_transmittance.w
                    );
                    collected_grad_info_pixel[lane_idx] = make_float4(
                        grad_color_pixel,
                        final_transmittance * -dot(grad_color_pixel, background)
                    );
                    collected_last_contributor[lane_idx] = tile_n_processed[pixel_idx];
                }
                warp.sync();
            }

            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
                color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
                color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
                transmittance = warp.shfl_up(transmittance, 1);
                grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
                grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
                grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
                grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
            }

            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int current_shmem_index = i % 32;
                last_contributor = collected_last_contributor[current_shmem_index];
                const float4 color_pixel_after_transmittance = collected_color_pixel_after_transmittance[current_shmem_index];
                color_pixel_after = make_float3(color_pixel_after_transmittance);
                transmittance = color_pixel_after_transmittance.w;
                const float4 grad_info_pixel = collected_grad_info_pixel[current_shmem_index];
                grad_color_pixel = make_float3(grad_info_pixel);
                grad_alpha_common = grad_info_pixel.w;
            }

            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip) continue;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
            const float gaussian = expf(fminf(exponent, 0.0f));
            const float alpha = opacity * gaussian;
            if (alpha < config::min_alpha_threshold) continue;

            const float blending_weight = transmittance * alpha;

            const float3 dL_dcolor = blending_weight * grad_color_pixel * color_grad_factor;
            dL_dcolor_accum += dL_dcolor;

            color_pixel_after -= blending_weight * color;

            const float one_minus_alpha = 1.0f - alpha;
            const float one_minus_alpha_rcp = 1.0f / fmaxf(one_minus_alpha, config::one_minus_alpha_eps);
            const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_after * one_minus_alpha_rcp, grad_color_pixel);
            const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
            const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha;
            const float dL_dopacity = gaussian * dL_dalpha;
            dL_dopacity_accum += dL_dopacity;

            const float gaussian_grad_helper = -alpha * dL_dalpha;
            const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(
                delta.x * delta.x,
                delta.x * delta.y,
                delta.y * delta.y
            );
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                conic.x * delta.x + conic.y * delta.y,
                conic.y * delta.x + conic.z * delta.y
            );
            dL_dmean2d_accum += dL_dmean2d;

            transmittance *= one_minus_alpha;
        }

        if (valid_primitive) {
            const float p0 = dL_dmean2d_accum.x;
            const float p1 = dL_dmean2d_accum.y;
            const float p2 = dL_dconic_accum.x;
            const float p3 = dL_dconic_accum.y;
            const float p4 = dL_dconic_accum.z;
            const float p5 = opacity * (1.0f - opacity) * dL_dopacity_accum;
            const float p6 = dL_dcolor_accum.x;
            const float p7 = dL_dcolor_accum.y;
            const float p8 = dL_dcolor_accum.z;

            if (partial_out != nullptr) {
                float* po = partial_out + static_cast<size_t>(slot) * 9;
                po[0] = p0; po[1] = p1; po[2] = p2; po[3] = p3; po[4] = p4;
                po[5] = p5; po[6] = p6; po[7] = p7; po[8] = p8;
            }

            if (variant == 0) {
                atomicAdd(&grad_mean2d[primitive_idx].x, p0);
                atomicAdd(&grad_mean2d[primitive_idx].y, p1);
                atomicAdd(&grad_conic[primitive_idx], p2);
                atomicAdd(&grad_conic[n_primitives + primitive_idx], p3);
                atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], p4);
                atomicAdd(&grad_opacity[primitive_idx], p5);
                atomicAdd(&grad_colors[primitive_idx].x, p6);
                atomicAdd(&grad_colors[primitive_idx].y, p7);
                atomicAdd(&grad_colors[primitive_idx].z, p8);
            } else if (variant == 1) {
                float* pg = private_grad + static_cast<size_t>(slot) * 9;
                atomicAdd(&pg[0], p0);
                atomicAdd(&pg[1], p1);
                atomicAdd(&pg[2], p2);
                atomicAdd(&pg[3], p3);
                atomicAdd(&pg[4], p4);
                atomicAdd(&pg[5], p5);
                atomicAdd(&pg[6], p6);
                atomicAdd(&pg[7], p7);
                atomicAdd(&pg[8], p8);
            } else {
                float* pg = private_grad + static_cast<size_t>(slot) * 9;
                pg[0] = p0; pg[1] = p1; pg[2] = p2; pg[3] = p3; pg[4] = p4;
                pg[5] = p5; pg[6] = p6; pg[7] = p7; pg[8] = p8;
            }
        }
    }

}
