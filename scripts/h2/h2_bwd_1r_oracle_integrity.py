#!/usr/bin/env python3
"""H2-BWD-1R: Oracle Integrity Repair.

Root cause: #if preprocessor directives don't work with C++ template parameters.
The preprocessor evaluates VARIANT as 0 (undefined macro), so ALL variants compiled
as BASELINE. Fix: use if constexpr (C++17) which evaluates at template instantiation.

This script implements:
1. 9 kernel variants with if constexpr dispatch + variant signatures
2. Destructive canaries (BROKEN_VIS, ZERO_VJP) that MUST fail correctness
3. Buffer identity verification (no pointer reuse)
4. Fixed SIGMA_GATE counters (now actually compiled)
5. SASS instruction counting via cuobjdump
6. Interleaved randomized-block timing (5 reps × 100 measurements)
7. Provenance (run_id, timestamps, hashes, GPU UUID)
"""
import argparse, hashlib, json, math, os, random, time, uuid
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

// Variant selectors (template parameters)
#define VAR_BASELINE       0
#define VAR_SIGMA_GATE     1
#define VAR_SCALAR_ADJOINT 2
#define VAR_UV_REUSE       3
#define VAR_COMBINED       4
#define VAR_NO_EXP         5
#define VAR_NO_VJP         6
#define VAR_BROKEN_VIS     7
#define VAR_ZERO_VJP       8

// Variant signatures (written to counters[4] for verification)
#define SIG_BASELINE       1001
#define SIG_SIGMA_GATE     1002
#define SIG_SCALAR_ADJOINT 1003
#define SIG_UV_REUSE       1004
#define SIG_COMBINED       1005
#define SIG_NO_EXP         1006
#define SIG_NO_VJP         1007
#define SIG_BROKEN_VIS     1008
#define SIG_ZERO_VJP       1009

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
    int64_t *__restrict__ counters  // [6]: total, drop, clamp, exp, signature, do_count_flag
)
{
    constexpr uint32_t PX_ROWS = 16 / PX;
    auto block = cg::this_thread_block();
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i0 = block.group_index().y * tile_size;
    const uint32_t j0 = block.group_index().z * tile_size;
    const uint32_t ty = block.thread_index().y;
    const uint32_t tx = block.thread_index().x;

    // Write variant signature (thread 0 only)
    if (block.thread_rank() == 0) {
        if constexpr (VARIANT == VAR_BASELINE) counters[4] = SIG_BASELINE;
        else if constexpr (VARIANT == VAR_SIGMA_GATE) counters[4] = SIG_SIGMA_GATE;
        else if constexpr (VARIANT == VAR_SCALAR_ADJOINT) counters[4] = SIG_SCALAR_ADJOINT;
        else if constexpr (VARIANT == VAR_UV_REUSE) counters[4] = SIG_UV_REUSE;
        else if constexpr (VARIANT == VAR_COMBINED) counters[4] = SIG_COMBINED;
        else if constexpr (VARIANT == VAR_NO_EXP) counters[4] = SIG_NO_EXP;
        else if constexpr (VARIANT == VAR_NO_VJP) counters[4] = SIG_NO_VJP;
        else if constexpr (VARIANT == VAR_BROKEN_VIS) counters[4] = SIG_BROKEN_VIS;
        else if constexpr (VARIANT == VAR_ZERO_VJP) counters[4] = SIG_ZERO_VJP;
    }

    float px[PX], py[PX];
    int32_t pix_id[PX];
    bool inside[PX];
    float T_final[PX], T[PX];
    float buffer[PX][CDIM];
    float buffer_dot[PX];
    float v_render_c[PX][CDIM];
    float v_render_a[PX];
    int32_t bin_final[PX];
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
        bg_dot[q] = 0.f;
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

                // ===== SIGMA COMPUTATION (if constexpr) =====
                float sigma = 0.f, vis = 0.f, alpha = 0.f;
                bool is_valid = false, is_clamped = false;

                if constexpr (VARIANT == VAR_BASELINE || VARIANT == VAR_SCALAR_ADJOINT ||
                              VARIANT == VAR_NO_VJP || VARIANT == VAR_ZERO_VJP) {
                    sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                    vis = __expf(-sigma);
                    alpha = min(MAX_ALPHA, opac * vis);
                    is_valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
                    is_clamped = (opac * vis > MAX_ALPHA);
                } else if constexpr (VARIANT == VAR_SIGMA_GATE || VARIANT == VAR_COMBINED) {
                    sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                    if (sigma < 0.f || opac < ALPHA_THRESHOLD) {
                        is_valid = false; vis = 0.f; alpha = 0.f; is_clamped = false;
                        if constexpr (VARIANT == VAR_SIGMA_GATE) {
                            if (counters[5] != 0 && warp.thread_rank() == 0) {
                                atomicAdd((unsigned long long*)&counters[1], 1ULL);
                                atomicAdd((unsigned long long*)&counters[0], 1ULL);
                            }
                        }
                    } else {
                        const float sigma_drop = __logf(opac / ALPHA_THRESHOLD);
                        if (sigma > sigma_drop) {
                            is_valid = false; vis = 0.f; alpha = 0.f; is_clamped = false;
                            if constexpr (VARIANT == VAR_SIGMA_GATE) {
                                if (counters[5] != 0 && warp.thread_rank() == 0) {
                                    atomicAdd((unsigned long long*)&counters[1], 1ULL);
                                    atomicAdd((unsigned long long*)&counters[0], 1ULL);
                                }
                            }
                        } else if (opac >= MAX_ALPHA) {
                            const float sigma_clamp = __logf(opac / MAX_ALPHA);
                            if (sigma < sigma_clamp) {
                                is_valid = true; vis = 0.f; alpha = MAX_ALPHA; is_clamped = true;
                                if constexpr (VARIANT == VAR_SIGMA_GATE) {
                                    if (counters[5] != 0 && warp.thread_rank() == 0) {
                                        atomicAdd((unsigned long long*)&counters[2], 1ULL);
                                        atomicAdd((unsigned long long*)&counters[0], 1ULL);
                                    }
                                }
                            } else {
                                vis = __expf(-sigma);
                                alpha = opac * vis;
                                is_valid = true; is_clamped = false;
                                if constexpr (VARIANT == VAR_SIGMA_GATE) {
                                    if (counters[5] != 0 && warp.thread_rank() == 0) {
                                        atomicAdd((unsigned long long*)&counters[3], 1ULL);
                                        atomicAdd((unsigned long long*)&counters[0], 1ULL);
                                    }
                                }
                            }
                        } else {
                            vis = __expf(-sigma);
                            alpha = opac * vis;
                            is_valid = true; is_clamped = false;
                            if constexpr (VARIANT == VAR_SIGMA_GATE) {
                                if (counters[5] != 0 && warp.thread_rank() == 0) {
                                    atomicAdd((unsigned long long*)&counters[3], 1ULL);
                                    atomicAdd((unsigned long long*)&counters[0], 1ULL);
                                }
                            }
                        }
                    }
                } else if constexpr (VARIANT == VAR_UV_REUSE) {
                    float ux = cx * dx + cy * dy;
                    float uy = cy * dx + cz * dy;
                    sigma = 0.5f * (dx * ux + dy * uy);
                    vis = __expf(-sigma);
                    alpha = min(MAX_ALPHA, opac * vis);
                    is_valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
                    is_clamped = (opac * vis > MAX_ALPHA);
                } else if constexpr (VARIANT == VAR_NO_EXP || VARIANT == VAR_BROKEN_VIS) {
                    sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                    vis = 1.0f;
                    alpha = min(MAX_ALPHA, opac * vis);
                    is_valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
                    is_clamped = (opac * vis > MAX_ALPHA);
                }

                if(!is_valid) continue;

                any_valid = true;

                // ===== VJP COMPUTATION (if constexpr) =====
                if constexpr (VARIANT == VAR_NO_VJP) {
                    // Skip ALL VJP, just update T/buffer for traversal state
                    const float next_T = T[q] * (1.0f - alpha);
                    volatile float sink = alpha * T[q];
                    (void)sink;
                    T[q] = next_T;
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        buffer[q][k] += rgbs_batch[t * CDIM + k] * alpha * T[q];
                } else if constexpr (VARIANT == VAR_ZERO_VJP) {
                    // Keep v_rgb, zero v_conic/v_xy/v_opacity (already zero from init)
                    const float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                    T[q] *= ra;
                    const float fac = alpha * T[q];
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        v_rgb_local[k] += fac * v_render_c[q][k];
                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        buffer[q][k] += rgbs_batch[t * CDIM + k] * fac;
                } else {
                    // Full VJP
                    const float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                    T[q] *= ra;
                    const float fac = alpha * T[q];

                    #pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                        v_rgb_local[k] += fac * v_render_c[q][k];

                    float v_alpha;
                    if constexpr (VARIANT == VAR_SCALAR_ADJOINT || VARIANT == VAR_COMBINED) {
                        float rgb_dot = 0.f;
                        #pragma unroll
                        for(uint32_t k = 0; k < CDIM; ++k)
                            rgb_dot += rgbs_batch[t * CDIM + k] * v_render_c[q][k];
                        v_alpha = T[q] * rgb_dot + ra * (tail_const[q] - buffer_dot[q]);
                        buffer_dot[q] += rgb_dot * fac;
                    } else {
                        v_alpha = 0.f;
                        #pragma unroll
                        for(uint32_t k = 0; k < CDIM; ++k)
                            v_alpha += (rgbs_batch[t * CDIM + k] * T[q] - buffer[q][k] * ra) * v_render_c[q][k];
                        v_alpha += T_final[q] * ra * v_render_a[q];
                        #pragma unroll
                        for(uint32_t k = 0; k < CDIM; ++k)
                            buffer[q][k] += rgbs_batch[t * CDIM + k] * fac;
                    }

                    if(!is_clamped) {
                        const float v_sigma = -opac * vis * v_alpha;
                        v_conic_local[0] += 0.5f * v_sigma * dx * dx;
                        v_conic_local[1] += v_sigma * dx * dy;
                        v_conic_local[2] += 0.5f * v_sigma * dy * dy;
                        if constexpr (VARIANT == VAR_UV_REUSE || VARIANT == VAR_COMBINED) {
                            float ux = cx * dx + cy * dy;
                            float uy = cy * dx + cz * dy;
                            v_xy_local[0] += v_sigma * ux;
                            v_xy_local[1] += v_sigma * uy;
                        } else {
                            v_xy_local[0] += v_sigma * (cx * dx + cy * dy);
                            v_xy_local[1] += v_sigma * (cy * dx + cz * dy);
                        }
                        v_opacity_local += vis * v_alpha;
                    }
                }
            }

            if(!warp.any(any_valid)) continue;

            // Warp reduction
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

void run_variant(
    int64_t variant,
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor tile_offsets, torch::Tensor flatten_ids,
    torch::Tensor render_alphas, torch::Tensor last_ids,
    torch::Tensor v_render_colors, torch::Tensor v_render_alphas,
    int64_t width, int64_t height, int64_t tile_size,
    torch::Tensor v_colors_out, torch::Tensor v_conics_out,
    torch::Tensor v_means2d_out, torch::Tensor v_opacities_out,
    torch::Tensor counters  // [5] int64
)
{
    constexpr int CDIM = 3;
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
    auto ra_d = render_alphas.const_data_ptr<float>();
    auto li = last_ids.const_data_ptr<int32_t>();
    auto vrc = v_render_colors.const_data_ptr<float>();
    auto vra = v_render_alphas.const_data_ptr<float>();
    auto vco = v_colors_out.data_ptr<float>();
    auto vcon = v_conics_out.data_ptr<float>();
    auto vm2 = v_means2d_out.data_ptr<float>();
    auto vop = v_opacities_out.data_ptr<float>();
    int64_t* ctr = (int64_t*)counters.data_ptr<int64_t>();

    #define LAUNCH(V) do { \
        cudaFuncSetAttribute(compute_oracle_kernel<3,2,V>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem); \
        compute_oracle_kernel<3,2,V><<<grid, threads, (size_t)shmem, stream>>>( \
            n_isects, m2d, con, col, opa, tile_w, tile_h, \
            (uint32_t)width, (uint32_t)height, (uint32_t)tile_size, \
            toff, fid, ra_d, li, vrc, vra, vco, vcon, vm2, vop, ctr); \
    } while(0)

    switch(variant) {
        case 0: LAUNCH(VAR_BASELINE); break;
        case 1: LAUNCH(VAR_SIGMA_GATE); break;
        case 2: LAUNCH(VAR_SCALAR_ADJOINT); break;
        case 3: LAUNCH(VAR_UV_REUSE); break;
        case 4: LAUNCH(VAR_COMBINED); break;
        case 5: LAUNCH(VAR_NO_EXP); break;
        case 6: LAUNCH(VAR_NO_VJP); break;
        case 7: LAUNCH(VAR_BROKEN_VIS); break;
        case 8: LAUNCH(VAR_ZERO_VJP); break;
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


VARIANT_NAMES = {
    0: "BASELINE", 1: "SIGMA_GATE", 2: "SCALAR_ADJOINT", 3: "UV_REUSE",
    4: "COMBINED", 5: "NO_EXP", 6: "NO_VJP", 7: "BROKEN_VIS", 8: "ZERO_VJP"
}
EXPECTED_SIG = {
    0: 1001, 1: 1002, 2: 1003, 3: 1004, 4: 1005,
    5: 1006, 6: 1007, 7: 1008, 8: 1009
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-measure", type=int, default=100)
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"
    run_id = str(uuid.uuid4())[:12]
    timestamp = time.strftime("%Y%m%dT%H%M%S")

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

    print(f"[H2-BWD-1R] run_id={run_id} scene={args.scene}/cam{args.cam_idx} {width}x{height} GPU={args.gpu}")

    # GPU UUID
    gpu_uuid = "unknown"
    try:
        gpu_uuid = torch.cuda.get_device_properties(device).name
    except:
        pass

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

    # Fixture hash: hash of key input tensors
    fixture_hasher = hashlib.sha256()
    fixture_hasher.update(means2d.cpu().numpy().tobytes()[:1000])
    fixture_hasher.update(conics.cpu().numpy().tobytes()[:1000])
    fixture_hasher.update(flatten_ids.cpu().numpy().tobytes()[:1000])
    fixture_hasher.update(str(N_visible).encode())
    fixture_hasher.update(str(n_isects).encode())
    fixture_hash = fixture_hasher.hexdigest()[:16]

    # Prepare flat inputs
    means2d_flat = means2d[0, 0].contiguous()
    conics_flat = conics[0, 0].contiguous()
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
    print("[build] Compiling CUDA extension with if constexpr...")
    from torch.utils.cpp_extension import load_inline
    ext = load_inline(
        name="h2_bwd_1r_oracle",
        cpp_sources=[], cuda_sources=[CUDA_SOURCE],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-Xptxas=-v"],
        verbose=True)
    print("[build] Done.")

    # Save generated CUDA source as artifact
    cuda_src_path = os.path.join(args.out_dir, "generated_cuda.cu")
    with open(cuda_src_path, "w") as f:
        f.write(CUDA_SOURCE)
    cuda_source_sha256 = hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()

    # Find .so file and compute hash
    so_dir = os.path.expanduser("~/.cache/torch_extensions/py310_cu128/h2_bwd_1r_oracle")
    so_path = os.path.join(so_dir, "h2_bwd_1r_oracle.so")
    binary_sha256 = "unknown"
    if os.path.exists(so_path):
        with open(so_path, "rb") as f:
            binary_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"[provenance] cuda_sha256={cuda_source_sha256[:16]} binary_sha256={binary_sha256[:16]}")

    def make_grads():
        return (torch.zeros(N_visible, 3, device=device), torch.zeros(N_visible, 3, device=device),
                torch.zeros(N_visible, 2, device=device), torch.zeros(N_visible, device=device))

    def make_counters(do_count=False):
        c = torch.zeros(6, dtype=torch.int64, device=device)
        if do_count:
            c[5] = 1  # do_count flag
        return c

    def run_one(vid, grads=None, counters=None):
        if grads is None:
            grads = make_grads()
        if counters is None:
            counters = make_counters()
        ext.run_variant(vid, means2d_flat, conics_flat, colors_flat, opacities_flat,
                        tile_offsets_flat, flatten_ids_flat, render_alphas_flat, last_ids_flat,
                        v_render_colors_flat, v_render_alphas_flat, width, height, tile_size,
                        *grads, counters)
        torch.cuda.synchronize(device)
        return grads, counters

    # ===== STEP 1: Variant signature verification =====
    print("\n===== STEP 1: Variant signature verification =====")
    dispatch_info = {}
    all_sigs_ok = True
    for vid in range(9):
        vname = VARIANT_NAMES[vid]
        _, ctr = run_one(vid)
        sig = int(ctr[4].item())
        expected = EXPECTED_SIG[vid]
        ok = (sig == expected)
        dispatch_info[vname] = {
            "variant_id": vid, "template_int": vid,
            "expected_signature": expected, "actual_signature": sig,
            "signature_ok": ok
        }
        status = "OK" if ok else "MISMATCH!"
        print(f"  {vname:20s}: sig={sig} expected={expected} {status}")
        if not ok:
            all_sigs_ok = False

    if not all_sigs_ok:
        print("FATAL: Signature mismatch! Variants are not distinct.")
        # Save what we have and exit
        with open(os.path.join(args.out_dir, "variant_dispatch.json"), "w") as f:
            json.dump(dispatch_info, f, indent=2)
        return

    print("  All signatures match. Variants are distinct.")

    # ===== STEP 2: Buffer identity verification =====
    print("\n===== STEP 2: Buffer identity verification =====")
    buffer_info = {}
    ref_grads = make_grads()
    ref_ptrs = [t.data_ptr() for t in ref_grads]
    for vid in range(9):
        vname = VARIANT_NAMES[vid]
        g = make_grads()
        g_ptrs = [t.data_ptr() for t in g]
        same = [ref_ptrs[i] == g_ptrs[i] for i in range(4)]
        buffer_info[vname] = {
            "v_colors_ptr": g_ptrs[0], "v_conics_ptr": g_ptrs[1],
            "v_means2d_ptr": g_ptrs[2], "v_opacities_ptr": g_ptrs[3],
            "same_as_baseline": same
        }
        any_same = any(same)
        status = "REUSE!" if any_same else "OK"
        print(f"  {vname:20s}: {status}")
    print("  All buffers independent." if not any(any(v["same_as_baseline"]) for v in buffer_info.values()) else "  FATAL: Buffer reuse detected!")

    # ===== STEP 3: Destructive canary correctness =====
    print("\n===== STEP 3: Destructive canary correctness =====")
    # Run BASELINE for reference
    ref_grads, _ = run_one(0)
    ref_v_colors, ref_v_conics, ref_v_means2d, ref_v_opacities = ref_grads

    canary_results = {}
    for vid, vname in [(7, "BROKEN_VIS"), (8, "ZERO_VJP")]:
        g, _ = run_one(vid)
        diffs = {}
        for name, ref, var in [("v_colors", ref_v_colors, g[0]), ("v_conics", ref_v_conics, g[1]),
                                ("v_means2d", ref_v_means2d, g[2]), ("v_opacities", ref_v_opacities, g[3])]:
            ref_norm = ref.norm().item()
            var_norm = var.norm().item()
            rel_l2 = ((ref - var).norm() / max(ref_norm, 1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), var.flatten().unsqueeze(0)).item() if var_norm > 1e-12 else 0.0
            is_zero = (var_norm < 1e-12)
            diffs[name] = {"rel_l2": rel_l2, "cosine": cos, "ref_norm": ref_norm, "var_norm": var_norm, "is_zero": is_zero}
        canary_results[vname] = diffs
        # Print
        print(f"  {vname}:")
        for name, d in diffs.items():
            print(f"    {name}: rel_l2={d['rel_l2']:.4f} cos={d['cosine']:.6f} ref_norm={d['ref_norm']:.2e} var_norm={d['var_norm']:.2e} zero={d['is_zero']}")

    # Check: BROKEN_VIS must fail substantially
    broken_vis_fail = any(d["rel_l2"] > 0.01 for d in canary_results["BROKEN_VIS"].values())
    # Check: ZERO_VJP must fail for v_conics, v_means2d, v_opacities but may pass v_colors
    zero_vjp_geom_fail = all(canary_results["ZERO_VJP"][n]["rel_l2"] > 0.01 or canary_results["ZERO_VJP"][n]["is_zero"]
                              for n in ["v_conics", "v_means2d", "v_opacities"])
    zero_vjp_color_ok = canary_results["ZERO_VJP"]["v_colors"]["rel_l2"] < 0.01 or canary_results["ZERO_VJP"]["v_colors"]["is_zero"]

    print(f"\n  BROKEN_VIS fails correctness: {broken_vis_fail}")
    print(f"  ZERO_VJP fails geometry VJP: {zero_vjp_geom_fail}")
    print(f"  ZERO_VJP v_colors OK/zero: {zero_vjp_color_ok}")

    if not broken_vis_fail:
        print("  FATAL: BROKEN_VIS did NOT fail correctness! Harness is broken.")
    if not zero_vjp_geom_fail:
        print("  FATAL: ZERO_VJP geometry VJP did NOT fail! Harness is broken.")

    with open(os.path.join(args.out_dir, "destructive_canary_correctness.json"), "w") as f:
        json.dump({
            "BROKEN_VIS": {"correctness": canary_results["BROKEN_VIS"], "fails_as_expected": broken_vis_fail},
            "ZERO_VJP": {"correctness": canary_results["ZERO_VJP"], "fails_as_expected": zero_vjp_geom_fail, "v_colors_ok": zero_vjp_color_ok},
            "harness_valid": broken_vis_fail and zero_vjp_geom_fail
        }, f, indent=2)

    # ===== STEP 4: SIGMA_GATE CUDA counters =====
    print("\n===== STEP 4: SIGMA_GATE CUDA counters =====")
    sg_counters = make_counters(do_count=True)
    sg_grads = make_grads()
    ext.run_variant(1, means2d_flat, conics_flat, colors_flat, opacities_flat,
                    tile_offsets_flat, flatten_ids_flat, render_alphas_flat, last_ids_flat,
                    v_render_colors_flat, v_render_alphas_flat, width, height, tile_size,
                    *sg_grads, sg_counters)
    torch.cuda.synchronize(device)
    counts = sg_counters.cpu().numpy()
    n_total = int(counts[0]); n_drop = int(counts[1])
    n_clamp = int(counts[2]); n_exp = int(counts[3])
    print(f"  N_classified (warp-level) = {n_total}")
    print(f"  N_drop = {n_drop} ({n_drop/max(n_total,1)*100:.1f}%)")
    print(f"  N_clamp = {n_clamp} ({n_clamp/max(n_total,1)*100:.1f}%)")
    print(f"  N_exp = {n_exp} ({n_exp/max(n_total,1)*100:.1f}%)")
    sum_check = (n_drop + n_clamp + n_exp == n_total) if n_total > 0 else False
    print(f"  n_drop + n_clamp + n_exp == n_total: {sum_check}")

    sigma_counts = {
        "N_classified_warp_level": n_total, "N_drop": n_drop, "N_clamp": n_clamp, "N_exp": n_exp,
        "fraction_drop": n_drop / max(n_total, 1), "fraction_clamp": n_clamp / max(n_total, 1),
        "fraction_exp": n_exp / max(n_total, 1), "sum_check": sum_check, "counters_nonzero": n_total > 0
    }
    with open(os.path.join(args.out_dir, "sigma_cuda_counts.json"), "w") as f:
        json.dump(sigma_counts, f, indent=2)

    # ===== STEP 5: Correctness for all variants vs BASELINE =====
    print("\n===== STEP 5: Correctness for all variants =====")
    correctness = {}
    for vid in range(1, 9):
        vname = VARIANT_NAMES[vid]
        g, _ = run_one(vid)
        diffs = {}
        for name, ref, var in [("v_colors", ref_v_colors, g[0]), ("v_conics", ref_v_conics, g[1]),
                                ("v_means2d", ref_v_means2d, g[2]), ("v_opacities", ref_v_opacities, g[3])]:
            max_abs = (ref - var).abs().max().item()
            ref_norm = ref.norm().item()
            rel_l2 = ((ref - var).norm() / max(ref_norm, 1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), var.flatten().unsqueeze(0)).item() if var.norm().item() > 1e-12 else 0.0
            nonzero_disagree = int(((ref.abs() > 0) != (var.abs() > 0)).sum().item())
            diffs[name] = {"max_abs": max_abs, "rel_l2": rel_l2, "cosine": cos, "nonzero_disagree": nonzero_disagree}
        all_cos = min(d["cosine"] for d in diffs.values())
        all_rel = max(d["rel_l2"] for d in diffs.values())
        # For exact variants (1-4): should pass. For diagnostic (5-8): expected to fail
        is_exact = vid in (1, 2, 3, 4)
        pass_gate = (all_cos >= 0.999999 and all_rel <= 1e-4) if is_exact else False
        correctness[vname] = {"per_metric": diffs, "min_cosine": all_cos, "max_rel_l2": all_rel,
                              "is_exact_variant": is_exact, "pass": pass_gate}
        status = "PASS" if pass_gate else ("FAIL (expected)" if not is_exact else "FAIL (UNEXPECTED!)")
        print(f"  {vname:20s}: cos={all_cos:.8f} rel_l2={all_rel:.2e} {status}")

    # ===== STEP 6: SASS instruction counting =====
    print("\n===== STEP 6: SASS instruction counting =====")
    sass_info = {}
    try:
        import subprocess
        cuobjdump_path = None
        for p in [os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "cuobjdump"),
                  "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/cuobjdump",
                  "cuobjdump"]:
            try:
                r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    cuobjdump_path = p
                    break
            except:
                continue

        if cuobjdump_path and os.path.exists(so_path):
            print(f"  Using cuobjdump: {cuobjdump_path}")
            r = subprocess.run([cuobjdump_path, "-sass", so_path], capture_output=True, text=True, timeout=30)
            sass_output = r.stdout

            # Parse SASS: find kernel functions and count instructions
            # SASS format: lines like "/*0000*/    INSTRUCTION ..."
            current_func = None
            func_counts = {}
            for line in sass_output.split("\n"):
                if "Function :" in line or ".text._Z21compute_oracle_kernel" in line:
                    # Extract mangled name
                    if "compute_oracle_kernel" in line:
                        parts = line.split()
                        for part in parts:
                            if "compute_oracle_kernel" in part:
                                current_func = part.strip()
                                break
                        if current_func not in func_counts:
                            func_counts[current_func] = {"MUFU": 0, "FFMA": 0, "FADD": 0, "FMUL": 0,
                                                          "LDG": 0, "STS": 0, "LDS": 0, "ATOM": 0, "RED": 0, "BAR": 0, "total": 0}
                elif current_func and line.strip().startswith("/*"):
                    # Instruction line
                    func_counts[current_func]["total"] += 1
                    upper = line.upper()
                    for inst in ["MUFU", "FFMA", "FADD", "FMUL", "LDG", "STS", "LDS", "ATOM", "RED", "BAR"]:
                        if inst in upper:
                            func_counts[current_func][inst] += 1

            # Map mangled names to variant names
            # The mangled name contains ILi3ELi2ELiVEE where V is the variant int
            for func, counts in func_counts.items():
                # Extract variant number from mangled name
                import re
                m = re.search(r"ILi3ELi2ELi(\d+)EE", func)
                if m:
                    vid = int(m.group(1))
                    vname = VARIANT_NAMES.get(vid, f"VAR_{vid}")
                    sass_info[vname] = counts
                    print(f"  {vname:20s}: total={counts['total']:4d} MUFU={counts['MUFU']:3d} FFMA={counts['FFMA']:3d} "
                          f"FADD={counts['FADD']:3d} FMUL={counts['FMUL']:3d} LDG={counts['LDG']:3d} "
                          f"STS={counts['STS']:3d} LDS={counts['LDS']:3d} ATOM={counts['ATOM']:3d} "
                          f"RED={counts['RED']:3d} BAR={counts['BAR']:3d}")

            # Verify: NO_EXP should have fewer MUFU than BASELINE
            if "BASELINE" in sass_info and "NO_EXP" in sass_info:
                base_mufu = sass_info["BASELINE"]["MUFU"]
                noexp_mufu = sass_info["NO_EXP"]["MUFU"]
                print(f"\n  MUFU check: BASELINE={base_mufu} NO_EXP={noexp_mufu} → {'PASS' if noexp_mufu < base_mufu else 'FAIL'}")
            if "BASELINE" in sass_info and "NO_VJP" in sass_info:
                base_arith = sass_info["BASELINE"]["FFMA"] + sass_info["BASELINE"]["FADD"] + sass_info["BASELINE"]["FMUL"]
                novjp_arith = sass_info["NO_VJP"]["FFMA"] + sass_info["NO_VJP"]["FADD"] + sass_info["NO_VJP"]["FMUL"]
                print(f"  Arithmetic check: BASELINE={base_arith} NO_VJP={novjp_arith} → {'PASS' if novjp_arith < base_arith else 'FAIL'}")

            # Save SASS output
            with open(os.path.join(args.out_dir, "sass_dump.txt"), "w") as f:
                f.write(sass_output)
        else:
            print("  cuobjdump not found or .so not found. Skipping SASS analysis.")
    except Exception as e:
        print(f"  SASS analysis failed: {e}")

    # Write SASS CSV
    with open(os.path.join(args.out_dir, "sass_instruction_counts.csv"), "w") as f:
        f.write("variant,total,MUFU,FFMA,FADD,FMUL,LDG,STS,LDS,ATOM,RED,BAR\n")
        for vname in [VARIANT_NAMES[i] for i in range(9)]:
            if vname in sass_info:
                c = sass_info[vname]
                f.write(f"{vname},{c['total']},{c['MUFU']},{c['FFMA']},{c['FADD']},{c['FMUL']},"
                        f"{c['LDG']},{c['STS']},{c['LDS']},{c['ATOM']},{c['RED']},{c['BAR']}\n")

    # ===== STEP 7: Interleaved randomized-block timing =====
    print(f"\n===== STEP 7: Interleaved timing ({args.n_reps} reps × {args.n_measure} measurements) =====")
    rng = random.Random(args.seed)
    timing_variants = [0, 1, 2, 3, 4, 5, 6]  # Exclude canaries from timing
    timing_data = {v: [] for v in timing_variants}
    # Store blocks for paired analysis
    all_blocks = []

    for rep in range(args.n_reps):
        for meas in range(args.n_measure):
            perm = timing_variants.copy()
            rng.shuffle(perm)
            block = {}
            for vid in perm:
                g = make_grads()
                ctr = make_counters()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                ext.run_variant(vid, means2d_flat, conics_flat, colors_flat, opacities_flat,
                                tile_offsets_flat, flatten_ids_flat, render_alphas_flat, last_ids_flat,
                                v_render_colors_flat, v_render_alphas_flat, width, height, tile_size,
                                *g, ctr)
                end.record()
                torch.cuda.synchronize(device)
                t = start.elapsed_time(end)
                block[vid] = t
                timing_data[vid].append(t)
            all_blocks.append(block)
        print(f"  rep {rep+1}/{args.n_reps} done")

    # Compute timing statistics
    timing_stats = {}
    paired_rows = []
    for vid in timing_variants:
        vname = VARIANT_NAMES[vid]
        times = np.array(timing_data[vid])
        median = float(np.median(times))
        mean = float(np.mean(times))
        std = float(np.std(times))
        timing_stats[vname] = {"median_ms": median, "mean_ms": mean, "std_ms": std,
                                "n": len(times), "times": [float(t) for t in times]}

    # Paired deltas vs BASELINE
    baseline_times = np.array(timing_data[0])
    for vid in timing_variants[1:]:
        vname = VARIANT_NAMES[vid]
        var_times = np.array(timing_data[vid])
        deltas = baseline_times - var_times  # positive = variant is faster
        paired_median = float(np.median(deltas))
        p50 = float(np.percentile(deltas, 50))
        p90 = float(np.percentile(deltas, 90))
        # Bootstrap CI
        n_boot = 1000
        boot_medians = []
        for _ in range(n_boot):
            idx = rng.sample(range(len(deltas)), len(deltas))
            boot_medians.append(np.median(deltas[idx]))
        ci_low = float(np.percentile(boot_medians, 2.5))
        ci_high = float(np.percentile(boot_medians, 97.5))
        timing_stats[vname]["paired_median_delta"] = paired_median
        timing_stats[vname]["paired_p50"] = p50
        timing_stats[vname]["paired_p90"] = p90
        timing_stats[vname]["ci_95_low"] = ci_low
        timing_stats[vname]["ci_95_high"] = ci_high
        paired_rows.append(f"{vname},{paired_median:.6f},{p50:.6f},{p90:.6f},{ci_low:.6f},{ci_high:.6f}")

    # Print timing
    T_base = timing_stats["BASELINE"]["median_ms"]
    print(f"\n  {'Variant':20s} {'median':>8s} {'mean':>8s} {'std':>8s} {'Δmed':>8s} {'p90':>8s} {'CI95%':>20s}")
    for vid in timing_variants:
        vname = VARIANT_NAMES[vid]
        s = timing_stats[vname]
        if vname == "BASELINE":
            print(f"  {vname:20s} {s['median_ms']:8.3f} {s['mean_ms']:8.3f} {s['std_ms']:8.4f}")
        else:
            ci = f"[{s['ci_95_low']:.4f}, {s['ci_95_high']:.4f}]"
            print(f"  {vname:20s} {s['median_ms']:8.3f} {s['mean_ms']:8.3f} {s['std_ms']:8.4f} {s['paired_median_delta']:8.4f} {s['paired_p90']:8.4f} {ci:>20s}")

    # Write paired timing CSV
    with open(os.path.join(args.out_dir, "paired_timing.csv"), "w") as f:
        f.write("variant,paired_median_delta,p50,p90,ci_95_low,ci_95_high\n")
        for row in paired_rows:
            f.write(row + "\n")

    # ===== STEP 8: T_F+B reference =====
    print("\n===== STEP 8: T_F+B reference =====")
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

    for _ in range(args.n_warmup): fb_fn()
    torch.cuda.synchronize(device)
    fb_times = []
    for _ in range(args.n_measure):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); fb_fn(); end.record(); torch.cuda.synchronize(device)
        fb_times.append(start.elapsed_time(end))
    T_fb = float(np.median(fb_times))
    print(f"  T_F+B: median={T_fb:.3f}ms")

    # ===== STEP 9: Classification =====
    print("\n===== STEP 9: Classification =====")

    # Check oracle validity
    harness_valid = all_sigs_ok and broken_vis_fail and zero_vjp_geom_fail
    counters_work = n_total > 0 and sum_check
    sass_valid = True
    if "BASELINE" in sass_info and "NO_EXP" in sass_info:
        sass_valid = sass_info["NO_EXP"]["MUFU"] < sass_info["BASELINE"]["MUFU"]
    if "BASELINE" in sass_info and "NO_VJP" in sass_info:
        base_arith = sass_info["BASELINE"]["FFMA"] + sass_info["BASELINE"]["FADD"] + sass_info["BASELINE"]["FMUL"]
        novjp_arith = sass_info["NO_VJP"]["FFMA"] + sass_info["NO_VJP"]["FADD"] + sass_info["NO_VJP"]["FMUL"]
        sass_valid = sass_valid and (novjp_arith < base_arith)

    if not harness_valid or not counters_work or not sass_valid:
        classification = "ORACLE_INVALID"
    else:
        # Check NO_EXP and NO_VJP timing
        no_exp_delta = timing_stats["NO_EXP"]["paired_median_delta"]
        no_vjp_delta = timing_stats["NO_VJP"]["paired_median_delta"]
        T_base_med = timing_stats["BASELINE"]["median_ms"]
        no_exp_pct = no_exp_delta / T_base_med * 100 if T_base_med > 0 else 0
        no_vjp_pct = no_vjp_delta / T_base_med * 100 if T_base_med > 0 else 0

        print(f"  NO_EXP: delta={no_exp_delta:.4f}ms ({no_exp_pct:.1f}% of baseline)")
        print(f"  NO_VJP: delta={no_vjp_delta:.4f}ms ({no_vjp_pct:.1f}% of baseline)")

        if abs(no_exp_pct) < 2 and abs(no_vjp_pct) < 2:
            classification = "ARITHMETIC_NOT_BOTTLENECK"
        else:
            classification = "ARITHMETIC_MATERIAL"

    print(f"\n  Classification: {classification}")

    # ===== Write all outputs =====
    # variant_dispatch.json
    with open(os.path.join(args.out_dir, "variant_dispatch.json"), "w") as f:
        json.dump(dispatch_info, f, indent=2)

    # buffer_identity.json
    with open(os.path.join(args.out_dir, "buffer_identity.json"), "w") as f:
        json.dump(buffer_info, f, indent=2)

    # oracle_timings.json
    oracle_timings = {
        "scene": args.scene, "camera_idx": args.cam_idx,
        "width": width, "height": height,
        "N_visible": int(N_visible), "n_isects": int(n_isects),
        "n_warmup": args.n_warmup, "n_measure": args.n_measure, "n_reps": args.n_reps,
        "seed": args.seed, "T_fb_reference_ms": T_fb,
        "timing": timing_stats, "correctness": correctness,
        "sigma_gate_counts": sigma_counts,
        "sass_info": sass_info,
        "classification": classification,
        "harness_valid": harness_valid,
        "counters_work": counters_work,
        "sass_valid": sass_valid,
    }
    with open(os.path.join(args.out_dir, "oracle_timings.json"), "w") as f:
        json.dump(oracle_timings, f, indent=2)

    # provenance.json
    provenance = {
        "run_id": run_id, "timestamp": timestamp,
        "scene": args.scene, "camera_idx": args.cam_idx,
        "width": width, "height": height,
        "gpu": args.gpu, "gpu_name": gpu_uuid,
        "cuda_source_sha256": cuda_source_sha256,
        "binary_sha256": binary_sha256,
        "fixture_hash": fixture_hash,
        "N_visible": int(N_visible), "n_isects": int(n_isects),
        "n_warmup": args.n_warmup, "n_measure": args.n_measure, "n_reps": args.n_reps,
        "seed": args.seed,
        "B2_base_commit": "77ab983ffe43420b2131669cb35776b883ca4c3c",
        "B2_patch_sha256": "74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c",
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\n===== DONE =====")
    print(f"  Classification: {classification}")
    print(f"  Artifacts in: {args.out_dir}")


if __name__ == "__main__":
    main()
