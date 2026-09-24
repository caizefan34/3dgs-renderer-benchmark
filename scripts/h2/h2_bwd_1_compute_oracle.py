#!/usr/bin/env python3
"""H2-BWD-1: Compute-Factored HiGS Backward Oracle.

Implements microbenchmarks for 7 variants of higs_blend_bwd_px_kernel<3,2>:
  0: BASELINE      — exact current kernel
  1: SIGMA_GATE    — skip expf for drop/clamp regions (Candidate A)
  2: SCALAR_ADJOINT — scalar buffer_dot instead of buffer[CDIM] (Candidate B)
  3: UV_REUSE      — factor ux/uy from sigma and v_xy (Candidate C)
  4: COMBINED      — A + B + C
  5: NO_EXP_ORACLE — replace expf with constant (diagnostic)
  6: NO_VJP_ORACLE — keep weight eval, replace VJP with sink (diagnostic)

All variants preserve: same traversal, same last_ids, same warpSum, same atomics.
"""
import argparse, json, math, os, time, sys
import numpy as np
import torch

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>

namespace cg = cooperative_groups;

// Constants from gsplat Common.h
#define ALPHA_THRESHOLD  (1.0f / 255.0f)
#define MAX_ALPHA        0.99f
#define MIN_ONE_MINUS_ALPHA 1e-6f

// Variant selectors
#define VAR_BASELINE      0
#define VAR_SIGMA_GATE    1
#define VAR_SCALAR_ADJOINT 2
#define VAR_UV_REUSE      3
#define VAR_COMBINED      4
#define VAR_NO_EXP        5
#define VAR_NO_VJP        6

// Counters passed as kernel parameter to avoid static variable issues

template<int CDIM, int PX, int VARIANT>
__global__ void compute_oracle_kernel(
    const uint32_t n_isects,
    const float *__restrict__ means2d_flat,
    const float *__restrict__ conics_flat,
    const float *__restrict__ colors_flat,
    const float *__restrict__ opacities,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas,
    float *__restrict__ v_colors_out,
    float *__restrict__ v_conics_out,
    float *__restrict__ v_means2d_out,
    float *__restrict__ v_opacities_out,
    int64_t *__restrict__ counters  // [4] for SIGMA_GATE counting, nullptr otherwise
)
{
    constexpr uint32_t PX_ROWS = 16 / PX;
    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i0 = block.group_index().y * tile_size;
    const uint32_t j0 = block.group_index().z * tile_size;
    const uint32_t ty = block.thread_index().y;
    const uint32_t tx = block.thread_index().x;

    float px[PX], py[PX];
    int32_t pix_id[PX];
    bool inside[PX];
    float T_final[PX], T[PX];
    float buffer[PX][CDIM];        // used by BASELINE, SIGMA_GATE, UV_REUSE, NO_EXP, NO_VJP
    float buffer_dot[PX];          // used by SCALAR_ADJOINT, COMBINED
    float v_render_c[PX][CDIM];
    float v_render_a[PX];
    int32_t bin_final[PX];

    // Precompute bg_dot for SCALAR_ADJOINT (no backgrounds in this benchmark, so bg_dot = 0)
    float bg_dot[PX];
    float tail_const[PX];

    #pragma unroll
    for(uint32_t q = 0; q < PX; ++q) {
        const uint32_t i = i0 + ty + q * PX_ROWS;
        const uint32_t j = j0 + tx;
        px[q] = (float)j + 0.5f;
        py[q] = (float)i + 0.5f;
        pix_id[q] = min(i * image_width + j, image_width * image_height - 1);
        inside[q] = (i < image_height && j < image_width);
        T_final[q] = 1.0f - render_alphas[pix_id[q]];
        T[q] = T_final[q];
        bin_final[q] = inside[q] ? last_ids[pix_id[q]] : 0;
        #pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k) {
            v_render_c[q][k] = v_render_colors[pix_id[q] * CDIM + k];
            buffer[q][k] = 0.f;
        }
        v_render_a[q] = v_render_alphas[pix_id[q]];
        buffer_dot[q] = 0.f;
        bg_dot[q] = 0.f;  // no backgrounds in benchmark
        tail_const[q] = T_final[q] * (v_render_a[q] - bg_dot[q]);
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (tile_id == tile_width * tile_height - 1)
                            ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = 128;
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ char s[];
    int32_t *id_batch = (int32_t *)s;
    float *xy_opac_batch = (float *)&id_batch[block_size];
    float *conic_batch = (float *)&xy_opac_batch[block_size * 3];
    float *rgbs_batch = (float *)&conic_batch[block_size * 3];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int32_t warp_bin_final = 0;
    #pragma unroll
    for(uint32_t q = 0; q < PX; ++q)
        warp_bin_final = max(warp_bin_final, bin_final[q]);
    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    for(uint32_t b = 0; b < num_batches; ++b) {
        block.sync();
        const int32_t batch_end = range_end - 1 - (int32_t)(block_size * b);
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - (int32_t)tr;
        if(idx >= range_start) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xy_opac_batch[tr * 3] = means2d_flat[g * 2];
            xy_opac_batch[tr * 3 + 1] = means2d_flat[g * 2 + 1];
            xy_opac_batch[tr * 3 + 2] = opacities[g];
            conic_batch[tr * 3] = conics_flat[g * 3];
            conic_batch[tr * 3 + 1] = conics_flat[g * 3 + 1];
            conic_batch[tr * 3 + 2] = conics_flat[g * 3 + 2];
            #pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
                rgbs_batch[tr * CDIM + k] = colors_flat[(int64_t)g * CDIM + k];
        }
        block.sync();

        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t) {
            float v_rgb_local[CDIM] = {0.f};
            float v_conic_local[3] = {0.f, 0.f, 0.f};
            float v_xy_local[2] = {0.f, 0.f};
            float v_opacity_local = 0.f;
            bool any_valid = false;

            for(uint32_t q = 0; q < PX; ++q) {
                bool valid_q = inside[q];
                if(batch_end - (int32_t)t > bin_final[q]) valid_q = false;
                if(!valid_q) continue;

                const float cx = conic_batch[t * 3];
                const float cy = conic_batch[t * 3 + 1];
                const float cz = conic_batch[t * 3 + 2];
                const float ox = xy_opac_batch[t * 3];
                const float oy = xy_opac_batch[t * 3 + 1];
                const float opac = xy_opac_batch[t * 3 + 2];
                const float dx = ox - px[q];
                const float dy = oy - py[q];

                // ===== SIGMA COMPUTATION =====
                float sigma, vis, alpha;
                bool is_valid, is_clamped;

            #if VARIANT == VAR_BASELINE || VARIANT == VAR_NO_VJP
                // Original: compute sigma, exp, alpha, check validity
                sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                vis = __expf(-sigma);
                alpha = min(MAX_ALPHA, opac * vis);
                is_valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
                is_clamped = (opac * vis > MAX_ALPHA);
            #elif VARIANT == VAR_SIGMA_GATE || VARIANT == VAR_COMBINED
                // SIGMA_GATE: classify before exp
                sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                if (sigma < 0.f || opac < ALPHA_THRESHOLD) {
                    is_valid = false;
                    vis = 0.f; alpha = 0.f; is_clamped = false;
                    if (VARIANT == VAR_SIGMA_GATE) {
                        // Count: only thread 0 per warp counts to reduce atomics
                        if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[1], 1ULL);
                        if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[0], 1ULL);
                    }
                } else {
                    const float sigma_drop = __logf(opac / ALPHA_THRESHOLD);
                    if (sigma > sigma_drop) {
                        is_valid = false;
                        vis = 0.f; alpha = 0.f; is_clamped = false;
                        if (VARIANT == VAR_SIGMA_GATE) {
                            if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[1], 1ULL);
                            if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[0], 1ULL);
                        }
                    } else if (opac >= MAX_ALPHA) {
                        const float sigma_clamp = __logf(opac / MAX_ALPHA);
                        if (sigma < sigma_clamp) {
                            // Clamp path: alpha = MAX_ALPHA, no exp needed
                            is_valid = true;
                            vis = 0.f; // not needed
                            alpha = MAX_ALPHA;
                            is_clamped = true;
                            if (VARIANT == VAR_SIGMA_GATE) {
                                if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[2], 1ULL);
                                if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[0], 1ULL);
                            }
                        } else {
                            vis = __expf(-sigma);
                            alpha = opac * vis;
                            is_valid = true;
                            is_clamped = false;
                            if (VARIANT == VAR_SIGMA_GATE) {
                                if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[3], 1ULL);
                                if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[0], 1ULL);
                            }
                        }
                    } else {
                        vis = __expf(-sigma);
                        alpha = opac * vis;
                        is_valid = true;
                        is_clamped = false;
                        if (VARIANT == VAR_SIGMA_GATE) {
                            if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[3], 1ULL);
                            if (warp.thread_rank() == 0) atomicAdd((unsigned long long*)&counters[0], 1ULL);
                        }
                    }
                }
            #elif VARIANT == VAR_UV_REUSE
                // UV_REUSE: factor ux, uy
                float ux = cx * dx + cy * dy;
                float uy = cy * dx + cz * dy;
                sigma = 0.5f * (dx * ux + dy * uy);
                vis = __expf(-sigma);
                alpha = min(MAX_ALPHA, opac * vis);
                is_valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
                is_clamped = (opac * vis > MAX_ALPHA);
            #elif VARIANT == VAR_SCALAR_ADJOINT
                // SCALAR_ADJOINT: same sigma/exp as baseline
                sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                vis = __expf(-sigma);
                alpha = min(MAX_ALPHA, opac * vis);
                is_valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
                is_clamped = (opac * vis > MAX_ALPHA);
            #elif VARIANT == VAR_NO_EXP
                // NO_EXP: replace expf with constant
                sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                vis = 1.0f; // cheapest possible replacement
                alpha = min(MAX_ALPHA, opac * vis);
                is_valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
                is_clamped = (opac * vis > MAX_ALPHA);
            #endif

                if(!is_valid) continue;

                any_valid = true;

            #if VARIANT == VAR_NO_VJP
                // NO_VJP: do weight eval (done above) but skip derivative arithmetic
                // Just update T and buffer to maintain traversal state
                {
                    const float next_T = T[q] * (1.0f - alpha);
                    // Sink: consume fac into a volatile to prevent optimization
                    volatile float sink = alpha * T[q];
                    (void)sink;
                    T[q] = next_T;
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        buffer[q][k] += rgbs_batch[t * CDIM + k] * alpha * T[q];
                }
            #else
                // ===== VJP COMPUTATION =====
                const float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T[q] *= ra;
                const float fac = alpha * T[q];

                // Color gradient (all variants)
                #pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                    v_rgb_local[k] += fac * v_render_c[q][k];

                // v_alpha computation
                float v_alpha;
            #if VARIANT == VAR_SCALAR_ADJOINT || VARIANT == VAR_COMBINED
                // SCALAR_ADJOINT: use scalar buffer_dot
                {
                    float rgb_dot = 0.f;
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        rgb_dot += rgbs_batch[t * CDIM + k] * v_render_c[q][k];
                    v_alpha = T[q] * rgb_dot + ra * (tail_const[q] - buffer_dot[q]);
                    buffer_dot[q] += rgb_dot * fac;
                }
            #else
                // BASELINE/SIGMA_GATE/UV_REUSE/NO_EXP: use vector buffer
                {
                    v_alpha = 0.f;
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        v_alpha += (rgbs_batch[t * CDIM + k] * T[q] - buffer[q][k] * ra) * v_render_c[q][k];
                    v_alpha += T_final[q] * ra * v_render_a[q];
                    // no backgrounds in benchmark, so no bg term
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        buffer[q][k] += rgbs_batch[t * CDIM + k] * fac;
                }
            #endif

                // Conic/xy/opacity gradients (skip if clamped)
                if(!is_clamped) {
            #if VARIANT == VAR_UV_REUSE || VARIANT == VAR_COMBINED
                    // UV_REUSE: reuse ux, uy from sigma computation
                    // Need to recompute ux, uy since they were computed above
                    // (or store them — but registers are tight, let's recompute)
                    // Actually, for UV_REUSE we need to store ux/uy from the sigma step
                    // Let's recompute (the compiler may CSE them)
                    const float v_sigma = -opac * vis * v_alpha;
                    float ux2 = cx * dx + cy * dy;
                    float uy2 = cy * dx + cz * dy;
                    v_conic_local[0] += 0.5f * v_sigma * dx * dx;
                    v_conic_local[1] += v_sigma * dx * dy;
                    v_conic_local[2] += 0.5f * v_sigma * dy * dy;
                    v_xy_local[0] += v_sigma * ux2;
                    v_xy_local[1] += v_sigma * uy2;
                    v_opacity_local += vis * v_alpha;
            #else
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local[0] += 0.5f * v_sigma * dx * dx;
                    v_conic_local[1] += v_sigma * dx * dy;
                    v_conic_local[2] += 0.5f * v_sigma * dy * dy;
                    v_xy_local[0] += v_sigma * (cx * dx + cy * dy);
                    v_xy_local[1] += v_sigma * (cy * dx + cz * dy);
                    v_opacity_local += vis * v_alpha;
            #endif
                }
            #endif // NO_VJP
            }

            if(!warp.any(any_valid)) continue;

            // Warp reduction (shuffle-based, same for all variants)
            for(int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                    v_rgb_local[k] += __shfl_xor_sync(0xffffffff, v_rgb_local[k], offset);
                v_conic_local[0] += __shfl_xor_sync(0xffffffff, v_conic_local[0], offset);
                v_conic_local[1] += __shfl_xor_sync(0xffffffff, v_conic_local[1], offset);
                v_conic_local[2] += __shfl_xor_sync(0xffffffff, v_conic_local[2], offset);
                v_xy_local[0] += __shfl_xor_sync(0xffffffff, v_xy_local[0], offset);
                v_xy_local[1] += __shfl_xor_sync(0xffffffff, v_xy_local[1], offset);
                v_opacity_local += __shfl_xor_sync(0xffffffff, v_opacity_local, offset);
            }

            if(warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                #pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                    atomicAdd(v_colors_out + (int64_t)CDIM * g + k, v_rgb_local[k]);
                atomicAdd(v_conics_out + 3 * (int64_t)g, v_conic_local[0]);
                atomicAdd(v_conics_out + 3 * (int64_t)g + 1, v_conic_local[1]);
                atomicAdd(v_conics_out + 3 * (int64_t)g + 2, v_conic_local[2]);
                atomicAdd(v_means2d_out + 2 * (int64_t)g, v_xy_local[0]);
                atomicAdd(v_means2d_out + 2 * (int64_t)g + 1, v_xy_local[1]);
                atomicAdd(v_opacities_out + g, v_opacity_local);
            }
        }
    }
}

// Launch wrapper
void run_variant(
    int64_t variant,
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor tile_offsets, torch::Tensor flatten_ids,
    torch::Tensor render_alphas, torch::Tensor last_ids,
    torch::Tensor v_render_colors, torch::Tensor v_render_alphas,
    int64_t width, int64_t height, int64_t tile_size,
    torch::Tensor v_colors_out, torch::Tensor v_conics_out,
    torch::Tensor v_means2d_out, torch::Tensor v_opacities_out,
    torch::Tensor counters  // [4] int64 for SIGMA_GATE counting, empty otherwise
)
{
    constexpr int CDIM = 3;
    const uint32_t N = means2d.size(0);
    const uint32_t n_isects = flatten_ids.size(0);
    const uint32_t tile_w = tile_offsets.size(1);
    const uint32_t tile_h = tile_offsets.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(1, tile_h, tile_w);
    dim3 threads(16, 8, 1);
    const int64_t shmem = 128 * (4 + 12 + 12 + 12);

    auto m2d = means2d.const_data_ptr<float>();
    auto con = conics.const_data_ptr<float>();
    auto col = colors.const_data_ptr<float>();
    auto opa = opacities.const_data_ptr<float>();
    auto toff = tile_offsets.const_data_ptr<int32_t>();
    auto fid = flatten_ids.const_data_ptr<int32_t>();
    auto ra = render_alphas.const_data_ptr<float>();
    auto li = last_ids.const_data_ptr<int32_t>();
    auto vrc = v_render_colors.const_data_ptr<float>();
    auto vra = v_render_alphas.const_data_ptr<float>();
    auto vco = v_colors_out.data_ptr<float>();
    auto vcon = v_conics_out.data_ptr<float>();
    auto vm2 = v_means2d_out.data_ptr<float>();
    auto vop = v_opacities_out.data_ptr<float>();

    // Set up counters pointer
    int64_t* ctr_ptr = nullptr;
    if (variant == VAR_SIGMA_GATE && counters.numel() >= 4) {
        ctr_ptr = (int64_t*)counters.data_ptr<int64_t>();
    }

    #define LAUNCH(V) do { \
        cudaFuncSetAttribute(compute_oracle_kernel<3,2,V>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem); \
        compute_oracle_kernel<3,2,V><<<grid, threads, (size_t)shmem, stream>>>( \
            n_isects, m2d, con, col, opa, tile_w, tile_h, \
            (uint32_t)width, (uint32_t)height, (uint32_t)tile_size, \
            toff, fid, ra, li, vrc, vra, vco, vcon, vm2, vop, ctr_ptr); \
    } while(0)

    switch(variant) {
        case 0: LAUNCH(VAR_BASELINE); break;
        case 1: LAUNCH(VAR_SIGMA_GATE); break;
        case 2: LAUNCH(VAR_SCALAR_ADJOINT); break;
        case 3: LAUNCH(VAR_UV_REUSE); break;
        case 4: LAUNCH(VAR_COMBINED); break;
        case 5: LAUNCH(VAR_NO_EXP); break;
        case 6: LAUNCH(VAR_NO_VJP); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    #undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_variant", &run_variant, "Compute oracle variant");
}
"""

def load_ply_scene(ply_path, device):
    from plyfile import PlyData
    SH_DEGREE = 3; K_SH = (SH_DEGREE + 1) ** 2
    ply = PlyData.read(ply_path); v = ply["vertex"]; N = len(v)
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), dtype=torch.float32, device=device)
    quats = torch.tensor(np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]), dtype=torch.float32, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]), dtype=torch.float32, device=device))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], dtype=torch.float32, device=device))
    f_dc = torch.tensor(np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]), dtype=torch.float32, device=device)
    n_rest = 3 * (K_SH - 1)
    f_rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], dtype=torch.float32, device=device) for i in range(n_rest)], dim=1)
    f_rest = f_rest.reshape(N, 3, K_SH - 1).permute(0, 2, 1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = f_dc; sh[:, 1:] = f_rest
    return means, quats, scales, opacities, sh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-measure", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = f"cuda:{args.gpu}"

    SH_DEGREE = 3; K_SH = (SH_DEGREE + 1) ** 2
    scene_configs = {
        "room": {
            "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
            "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
            "native_w": 3114, "native_h": 2075,
        },
        "bicycle": {
            "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
            "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
            "native_w": 4946, "native_h": 3286,
        },
    }
    cfg = scene_configs[args.scene]
    nw, nh = cfg["native_w"], cfg["native_h"]
    if args.max_long_side > 0 and max(nw, nh) > args.max_long_side:
        scale = args.max_long_side / max(nw, nh)
        width, height = max(1, int(round(nw * scale))), max(1, int(round(nh * scale)))
    else:
        width, height = nw, nh

    print(f"[H2-BWD-1] {args.scene}/cam{args.cam_idx} {width}x{height}")
    means, quats, scales, opacities, sh = load_ply_scene(cfg["ply"], device)
    N_total = len(means)

    with open(cfg["cams"]) as f:
        cams = json.load(f)
    c = cams[args.cam_idx]
    R = np.asarray(c["rotation"], dtype=np.float64); p = np.asarray(c["position"], dtype=np.float64)
    Rw2c = R.T; vm = np.eye(4); vm[:3, :3] = Rw2c; vm[:3, 3] = -Rw2c @ p
    scale_f = width / float(c["width"])
    K = np.array([[float(c["fx"]) * scale_f, 0.0, (width - 1) / 2.0],
                  [0.0, float(c["fy"]) * scale_f, (height - 1) / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    vm_t = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K_t = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    tile_size = 16
    tile_width = math.ceil(width / tile_size); tile_height = math.ceil(height / tile_size)

    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode, _make_lazy_cuda_func
    from gsplat.rendering import _maybe_evaluate_sh
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native

    with torch.no_grad():
        visible_ids, _, _ = _cull_gaussians_batched(
            means, quats, scales, vm_t, K_t, width, height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, camera_model="pinhole")
        v_means, v_quats, v_scales, v_opacities, v_colors = _gather_visible_native(
            means, quats, scales, opacities, sh, visible_ids)
        N_visible = visible_ids.numel()

        v_means_b = v_means.unsqueeze(0).contiguous(); v_quats_b = v_quats.unsqueeze(0).contiguous()
        v_scales_b = v_scales.unsqueeze(0).contiguous(); v_opacities_b = v_opacities.unsqueeze(0).contiguous()
        v_colors_input = v_colors.unsqueeze(0) if v_colors.dim() == 2 else v_colors
        opacities_bc = torch.broadcast_to(v_opacities_b[..., None, :], (1, 1, N_visible)).contiguous()

        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=v_means_b, covars=None, quats=v_quats_b, scales=v_scales_b,
            viewmats=vm_t, Ks=K_t, width=width, height=height, eps2d=0.3,
            near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
            calc_compensations=False, camera_model="pinhole")
        colors_eval = _maybe_evaluate_sh(
            SH_DEGREE, v_colors_input, v_means_b, radii, vm_t, (1,), 1, N_visible, True).contiguous()

        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=False, n_images=1, image_ids=None, gaussian_ids=None,
            conics=conics, opacities=opacities_bc)
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height).reshape((1, 1, tile_height, tile_width))

        render_out = _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
            means2d.contiguous(), conics.contiguous(), colors_eval.contiguous(), opacities_bc.contiguous(),
            None, None, width, height, tile_size, isect_offsets.contiguous(), flatten_ids.contiguous(),
            False, False)
        render_colors, render_alphas, _, last_ids = render_out

    n_isects = isect_ids.numel()
    print(f"[data] N_visible={N_visible}, n_isects={n_isects}, tiles={tile_width}x{tile_height}")

    # Prepare flat inputs
    means2d_flat = means2d[0].contiguous()
    conics_flat = conics[0].contiguous()
    colors_flat = colors_eval[0].contiguous()
    opacities_flat = v_opacities.contiguous()
    tile_offsets_flat = isect_offsets[0, 0].contiguous()
    flatten_ids_flat = flatten_ids.contiguous().to(torch.int32)
    render_alphas_flat = render_alphas[0].squeeze(-1).contiguous() if render_alphas[0].dim() == 3 else render_alphas[0].contiguous()
    last_ids_flat = last_ids[0].contiguous().to(torch.int32)
    v_render_colors = torch.rand(1, height, width, 3, dtype=torch.float32, device=device) * 0.01
    v_render_alphas = torch.rand(1, height, width, dtype=torch.float32, device=device) * 0.01
    v_render_colors_flat = v_render_colors[0].contiguous()
    v_render_alphas_flat = v_render_alphas[0].contiguous()

    # Build extension
    print("[build] Compiling CUDA extension...")
    from torch.utils.cpp_extension import load_inline
    ext = load_inline(
        name="h2_bwd_1_oracle",
        cpp_sources=[], cuda_sources=[CUDA_SOURCE],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-Xptxas=-v"],
        verbose=True)
    print("[build] Done.")

    # Gradient output buffers
    def make_grads():
        return (torch.zeros(N_visible, 3, device=device), torch.zeros(N_visible, 3, device=device),
                torch.zeros(N_visible, 2, device=device), torch.zeros(N_visible, device=device))

    # ---- Step 1: Run BASELINE and capture reference gradients ----
    print("\n===== BASELINE =====")
    ref_grads = make_grads()
    ext.run_variant(0, means2d_flat, conics_flat, colors_flat, opacities_flat,
                    tile_offsets_flat, flatten_ids_flat, render_alphas_flat, last_ids_flat,
                    v_render_colors_flat, v_render_alphas_flat, width, height, tile_size,
                    *ref_grads, torch.zeros(4, dtype=torch.int64, device=device))
    torch.cuda.synchronize()

    # ---- Step 2: SIGMA_GATE counting ----
    print("\n===== SIGMA_GATE counting =====")
    counters = torch.zeros(4, dtype=torch.int64, device=device)
    sg_grads = make_grads()
    ext.run_variant(1, means2d_flat, conics_flat, colors_flat, opacities_flat,
                    tile_offsets_flat, flatten_ids_flat, render_alphas_flat, last_ids_flat,
                    v_render_colors_flat, v_render_alphas_flat, width, height, tile_size,
                    *sg_grads, counters)
    torch.cuda.synchronize()
    counts = counters.cpu().numpy()
    N_total_samples = int(counts[0]); N_drop = int(counts[1])
    N_clamp = int(counts[2]); N_exp = int(counts[3])
    print(f"  N_sample_candidates = {N_total_samples}")
    print(f"  N_drop_before_exp = {N_drop} ({N_drop/max(N_total_samples,1)*100:.1f}%)")
    print(f"  N_clamp_without_exp = {N_clamp} ({N_clamp/max(N_total_samples,1)*100:.1f}%)")
    print(f"  N_exp_required = {N_exp} ({N_exp/max(N_total_samples,1)*100:.1f}%)")

    # ---- Step 3: Correctness check for each variant ----
    print("\n===== Correctness =====")
    correctness = {}
    for vid, vname in [(1,"SIGMA_GATE"),(2,"SCALAR_ADJOINT"),(3,"UV_REUSE"),(4,"COMBINED"),(5,"NO_EXP"),(6,"NO_VJP")]:
        g = make_grads()
        ext.run_variant(vid, means2d_flat, conics_flat, colors_flat, opacities_flat,
                        tile_offsets_flat, flatten_ids_flat, render_alphas_flat, last_ids_flat,
                        v_render_colors_flat, v_render_alphas_flat, width, height, tile_size,
                        *g, torch.zeros(4, dtype=torch.int64, device=device))
        torch.cuda.synchronize()
        # Compare against BASELINE
        diffs = {}
        for name, ref, var in [("v_colors", ref_grads[0], g[0]), ("v_conics", ref_grads[1], g[1]),
                                ("v_means2d", ref_grads[2], g[2]), ("v_opacities", ref_grads[3], g[3])]:
            max_abs = (ref - var).abs().max().item()
            mean_abs = (ref - var).abs().mean().item()
            ref_norm = ref.norm().item()
            rel_l2 = ((ref - var).norm() / max(ref_norm, 1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), var.flatten().unsqueeze(0)).item()
            has_nan = bool(torch.isnan(var).any().item())
            has_inf = bool(torch.isinf(var).any().item())
            nonzero_disagree = int(((ref.abs() > 0) != (var.abs() > 0)).sum().item())
            diffs[name] = {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2,
                           "cosine": cos, "nan": has_nan, "inf": has_inf, "nonzero_disagree": nonzero_disagree}
        # Overall
        all_cos = min(d["cosine"] for d in diffs.values())
        all_rel = max(d["rel_l2"] for d in diffs.values())
        all_nan = any(d["nan"] for d in diffs.values())
        all_disagree = sum(d["nonzero_disagree"] for d in diffs.values())
        pass_gate = (all_cos >= 0.999999 and all_rel <= 1e-4 and all_disagree == 0 and not all_nan)
        correctness[vname] = {"per_metric": diffs, "min_cosine": all_cos, "max_rel_l2": all_rel,
                              "nonzero_disagree": all_disagree, "nan": all_nan, "pass": pass_gate}
        status = "PASS" if pass_gate else "FAIL"
        print(f"  {vname}: {status} cos={all_cos:.8f} rel_l2={all_rel:.2e} disagree={all_disagree} nan={all_nan}")

    # ---- Step 4: Timing all variants ----
    print("\n===== Timing =====")
    timing = {}
    variant_names = {0:"BASELINE",1:"SIGMA_GATE",2:"SCALAR_ADJOINT",3:"UV_REUSE",4:"COMBINED",5:"NO_EXP",6:"NO_VJP"}

    for vid in [0,1,2,3,4,5,6]:
        vname = variant_names[vid]
        def run():
            g = make_grads()
            ext.run_variant(vid, means2d_flat, conics_flat, colors_flat, opacities_flat,
                            tile_offsets_flat, flatten_ids_flat, render_alphas_flat, last_ids_flat,
                            v_render_colors_flat, v_render_alphas_flat, width, height, tile_size,
                            *g, torch.zeros(4, dtype=torch.int64, device=device))
        # Warmup
        for _ in range(args.n_warmup): run()
        torch.cuda.synchronize(device)
        times = []
        for _ in range(args.n_measure):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); run(); end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        times = np.array(times)
        median = float(np.median(times))
        timing[vname] = {"median_ms": median, "mean_ms": float(np.mean(times)),
                         "std_ms": float(np.std(times)), "times": [float(t) for t in times]}
        print(f"  {vname:20s}: median={median:.3f}ms mean={np.mean(times):.3f}ms")

    # ---- Step 5: Measure real F+B for reference ----
    print("\n===== Real F+B reference =====")
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(means.detach(), quats.detach(), scales.detach(),
                                  opacities.detach(), sh.detach(), sh_degree=SH_DEGREE)
    torch.cuda.synchronize()
    with torch.no_grad():
        ref_res = rasterize_gaussian_higs_frozen(
            means.detach(), quats.detach(), scales.detach(), opacities.detach(), sh.detach(),
            backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm_t, Ks=K_t, width=width, height=height,
            sh_degree=SH_DEGREE, use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
    target = ref_res["frame"].detach().clone()

    def fb_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        res = rasterize_gaussian_higs_frozen(
            m, q, s, o, c, backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm_t, Ks=K_t, width=width, height=height,
            sh_degree=SH_DEGREE, use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
        loss = (res["frame"] - target).abs().mean()
        loss.backward()

    real_times = []
    for _ in range(args.n_warmup): fb_fn()
    torch.cuda.synchronize(device)
    for _ in range(args.n_measure):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); fb_fn(); end.record(); torch.cuda.synchronize(device)
        real_times.append(start.elapsed_time(end))
    real_times = np.array(real_times)
    T_fb = float(np.median(real_times))
    print(f"  T_F+B: median={T_fb:.3f}ms")

    # ---- Compile results ----
    T_baseline = timing["BASELINE"]["median_ms"]
    results = {
        "scene": args.scene, "camera_idx": args.cam_idx,
        "width": width, "height": height,
        "N_visible": int(N_visible), "n_isects": int(n_isects),
        "n_warmup": args.n_warmup, "n_measure": args.n_measure,
        "sigma_gate_counts": {
            "N_sample_candidates": N_total_samples,
            "N_drop_before_exp": N_drop,
            "N_clamp_without_exp": N_clamp,
            "N_exp_required": N_exp,
            "fraction_drop": N_drop / max(N_total_samples, 1),
            "fraction_clamp": N_clamp / max(N_total_samples, 1),
            "fraction_exp_required": N_exp / max(N_total_samples, 1),
        },
        "timing": {},
        "correctness": correctness,
        "T_fb_reference_ms": T_fb,
    }

    for vname, t in timing.items():
        speedup = T_baseline / t["median_ms"] if t["median_ms"] > 0 else 0
        saved = T_baseline - t["median_ms"]
        pct_blend = saved / T_baseline * 100 if T_baseline > 0 else 0
        pct_fb = saved / T_fb * 100 if T_fb > 0 else 0
        results["timing"][vname] = {
            "median_ms": t["median_ms"], "mean_ms": t["mean_ms"], "std_ms": t["std_ms"],
            "speedup": speedup, "saved_ms": saved,
            "pct_blend_saved": pct_blend, "pct_fb_saved": pct_fb,
        }
        print(f"\n  {vname:20s}: median={t['median_ms']:.3f}ms speedup={speedup:.3f}x saved={saved:.3f}ms "
              f"blend={pct_blend:.1f}% fb={pct_fb:.1f}%")

    # Gate evaluation
    combined_t = timing["COMBINED"]["median_ms"]
    combined_saved = T_baseline - combined_t
    combined_pct_blend = combined_saved / T_baseline * 100 if T_baseline > 0 else 0
    combined_pct_fb = combined_saved / T_fb * 100 if T_fb > 0 else 0
    combined_correct = correctness["COMBINED"]["pass"]

    gate_blend = combined_pct_blend >= 10
    gate_fb = combined_pct_fb >= 5
    gate_correct = combined_correct
    promote = (gate_blend or gate_fb) and gate_correct

    results["gate"] = {
        "combined_pct_blend_saved": combined_pct_blend,
        "combined_pct_fb_saved": combined_pct_fb,
        "gate_blend_ge_10": gate_blend,
        "gate_fb_ge_5": gate_fb,
        "gate_correctness": gate_correct,
        "promote_to_cuda": promote,
        "verdict": "PROMOTE_TO_CUDA" if promote else ("KEEP" if combined_correct else "DROP"),
    }

    print(f"\n===== GATE =====")
    print(f"  COMBINED: blend={combined_pct_blend:.1f}% fb={combined_pct_fb:.1f}% correct={combined_correct}")
    print(f"  Verdict: {results['gate']['verdict']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
