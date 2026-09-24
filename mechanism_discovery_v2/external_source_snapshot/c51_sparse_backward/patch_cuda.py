#!/usr/bin/env python3
"""Phase C51 Stage 4A — CUDA Kernel Patch

Patches the gsplat 1.5.3 source on the remote A100 to add importance_mask
support to the backward rasterization kernel.

Design B: Skip gradient computation for masked Gaussians while keeping
T/buffer reconstruction exact.

Three modes:
  - B1/B3: importance_mask=0 → skip ALL gradient compute + atomicAdd
  - B2: importance_mask=0 → compute v_means2d only (for densification),
         skip v_colors/v_conics/v_opacities compute + atomicAdd
  - Baseline: importance_mask=None → original behavior (all gradients computed)

Files patched:
  1. RasterizeToPixels3DGSBwd.cu — kernel + launch function
  2. Rasterization.h — declaration
  3. Rasterization.cpp — C++ binding
  4. _wrapper.py — Python autograd
  5. rendering.py — high-level API
"""
import sys
import os
import shutil
from pathlib import Path

GSPLAT_ROOT = Path("/tmp/gsplat_baseline/gsplat-1.5.3")
CSRC = GSPLAT_ROOT / "gsplat" / "cuda" / "csrc"
CUDA_PKG = GSPLAT_ROOT / "gsplat" / "cuda"
RENDERING = GSPLAT_ROOT / "gsplat" / "rendering.py"

def backup_file(path):
    bak = path.with_suffix(path.suffix + ".orig_stage4a")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name} → {bak.name}")
    else:
        print(f"  Backup exists: {bak.name}")

def patch_bwd_kernel():
    """Patch RasterizeToPixels3DGSBwd.cu — the core CUDA kernel."""
    fpath = CSRC / "RasterizeToPixels3DGSBwd.cu"
    backup_file(fpath)
    
    with open(fpath, 'r') as f:
        src = f.read()
    
    # 1. Add importance_mask and compute_densify_grad to kernel template params
    old_sig = """template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]"""
    
    new_sig = """template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    // sparse backward: importance mask
    const uint8_t *__restrict__ importance_mask, // [N] or [nnz], 1=compute grad, 0=skip
    const bool compute_densify_grad,             // B2: if true, compute v_means2d for masked"""
    
    assert old_sig in src, "Could not find kernel signature"
    src = src.replace(old_sig, new_sig)
    
    # 2. Add mask_batch to shared memory layout
    old_shmem = """    float *rgbs_batch =
        (float *)&conic_batch[block_size]; // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel"""
    
    new_shmem = """    float *rgbs_batch =
        (float *)&conic_batch[block_size]; // [block_size * CDIM]
    bool *mask_batch =
        (bool *)&rgbs_batch[block_size * CDIM]; // [block_size] — importance mask

    // this is the T AFTER the last gaussian in this pixel"""
    
    assert old_shmem in src, "Could not find shared memory layout"
    src = src.replace(old_shmem, new_shmem)
    
    # 3. Load mask during batch loading
    old_load = """        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }"""
    
    new_load = """        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
            // Load importance mask: 1=compute gradient, 0=skip
            if (importance_mask != nullptr) {
                int32_t mask_idx = packed ? g : (g - (int32_t)(image_id * N));
                mask_batch[tr] = importance_mask[mask_idx] != 0;
            } else {
                mask_batch[tr] = true; // no mask = compute all
            }
        } else {
            mask_batch[tr] = true; // default for out-of-range threads
        }"""
    
    assert old_load in src, "Could not find batch loading code"
    src = src.replace(old_load, new_load)
    
    # 4. Modify the gradient computation block
    # Replace the entire if(valid) block + warpSum + atomicAdd section
    old_grad = """            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }"""
    
    new_grad = """            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            // Determine gradient computation mode for this Gaussian.
            // mask_batch[t] is uniform across all threads in the warp (same Gaussian).
            bool g_masked = !mask_batch[t];  // true if this Gaussian is skipped
            bool do_full_grad = valid && !g_masked;
            bool do_densify_only = valid && g_masked && compute_densify_grad;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian — MUST ALWAYS DO
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                // Buffer update — MUST ALWAYS DO (for T/buffer correctness)
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }

                if (do_full_grad) {
                    // === Full gradient computation (selected Gaussians) ===
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] = fac * v_render_c[k];
                    }
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                                   v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }
                    if (opac * vis <= 0.999f) {
                        const float v_sigma = -opac * vis * v_alpha;
                        v_conic_local = {
                            0.5f * v_sigma * delta.x * delta.x,
                            v_sigma * delta.x * delta.y,
                            0.5f * v_sigma * delta.y * delta.y
                        };
                        v_xy_local = {
                            v_sigma * (conic.x * delta.x + conic.y * delta.y),
                            v_sigma * (conic.y * delta.x + conic.z * delta.y)
                        };
                        if (v_means2d_abs != nullptr) {
                            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                        }
                        v_opacity_local = vis * v_alpha;
                    }
                } else if (do_densify_only) {
                    // === B2: compute only v_xy_local for densification ===
                    // Need v_alpha and v_sigma to get v_xy_local.
                    // Skip v_rgb, v_conic, v_opacity computation.
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                                   v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }
                    if (opac * vis <= 0.999f) {
                        const float v_sigma = -opac * vis * v_alpha;
                        v_xy_local = {
                            v_sigma * (conic.x * delta.x + conic.y * delta.y),
                            v_sigma * (conic.y * delta.x + conic.z * delta.y)
                        };
                        if (v_means2d_abs != nullptr) {
                            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                        }
                        // v_rgb_local, v_conic_local, v_opacity_local stay 0
                    }
                }
                // else: B1/B3 mode, all gradient vars stay 0
            }
            // Warp reduction — uniform branch (mask_batch[t] same for all threads in warp)
            if (!g_masked) {
                // Full warpSum for selected Gaussians
                warpSum<CDIM>(v_rgb_local, warp);
                warpSum(v_conic_local, warp);
                warpSum(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum(v_xy_abs_local, warp);
                }
                warpSum(v_opacity_local, warp);
            } else if (compute_densify_grad) {
                // B2: only warpSum for v_xy (densification)
                warpSum(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum(v_xy_abs_local, warp);
                }
            }
            // else: B1/B3 — no warpSum needed (all zeros)
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                if (!g_masked) {
                    // Full atomicAdd for selected Gaussians
                    float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                    }
                    float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                    gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                    gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                    gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);
                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                    if (v_means2d_abs != nullptr) {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }
                    gpuAtomicAdd(v_opacities + g, v_opacity_local);
                } else if (compute_densify_grad) {
                    // B2: only v_means2d atomicAdd for densification
                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                    if (v_means2d_abs != nullptr) {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }
                }
                // else: B1/B3 — no atomicAdd
            }"""
    
    assert old_grad in src, "Could not find gradient computation block"
    src = src.replace(old_grad, new_grad)
    
    # 5. Update shared memory size calculation in launch function
    old_shmem_size = """    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);"""
    
    new_shmem_size = """    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM +
         sizeof(bool));"""
    
    assert old_shmem_size in src, "Could not find shared memory size"
    src = src.replace(old_shmem_size, new_shmem_size)
    
    # 6. Update launch function signature
    old_launch_sig = """template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
) {"""
    
    new_launch_sig = """template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad                 // B2 mode flag
) {"""
    
    assert old_launch_sig in src, "Could not find launch function signature"
    src = src.replace(old_launch_sig, new_launch_sig)
    
    # 7. Update kernel launch call in launch function
    old_launch_call = """    rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );"""
    
    new_launch_call = """    rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            importance_mask.has_value()
                ? reinterpret_cast<const uint8_t *>(
                      importance_mask.value().data_ptr<uint8_t>()
                  )
                : nullptr,
            compute_densify_grad,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );"""
    
    assert old_launch_call in src, "Could not find kernel launch call"
    src = src.replace(old_launch_call, new_launch_call)
    
    # 8. Update __INS__ macro
    old_ins = """#define __INS__(CDIM)                                                          \\
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \\
        const at::Tensor means2d,                                              \\
        const at::Tensor conics,                                               \\
        const at::Tensor colors,                                               \\
        const at::Tensor opacities,                                            \\
        const at::optional<at::Tensor> backgrounds,                            \\
        const at::optional<at::Tensor> masks,                                  \\
        uint32_t image_width,                                                  \\
        uint32_t image_height,                                                 \\
        uint32_t tile_size,                                                    \\
        const at::Tensor tile_offsets,                                         \\
        const at::Tensor flatten_ids,                                          \\
        const at::Tensor render_alphas,                                        \\
        const at::Tensor last_ids,                                             \\
        const at::Tensor v_render_colors,                                      \\
        const at::Tensor v_render_alphas,                                      \\
        at::optional<at::Tensor> v_means2d_abs,                                \\
        at::Tensor v_means2d,                                                  \\
        at::Tensor v_conics,                                                   \\
        at::Tensor v_colors,                                                   \\
        at::Tensor v_opacities                                                 \\
    );"""
    
    new_ins = """#define __INS__(CDIM)                                                          \\
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \\
        const at::Tensor means2d,                                              \\
        const at::Tensor conics,                                               \\
        const at::Tensor colors,                                               \\
        const at::Tensor opacities,                                            \\
        const at::optional<at::Tensor> backgrounds,                            \\
        const at::optional<at::Tensor> masks,                                  \\
        uint32_t image_width,                                                  \\
        uint32_t image_height,                                                 \\
        uint32_t tile_size,                                                    \\
        const at::Tensor tile_offsets,                                         \\
        const at::Tensor flatten_ids,                                          \\
        const at::Tensor render_alphas,                                        \\
        const at::Tensor last_ids,                                             \\
        const at::Tensor v_render_colors,                                      \\
        const at::Tensor v_render_alphas,                                      \\
        at::optional<at::Tensor> v_means2d_abs,                                \\
        at::Tensor v_means2d,                                                  \\
        at::Tensor v_conics,                                                   \\
        at::Tensor v_colors,                                                   \\
        at::Tensor v_opacities,                                                \\
        const at::optional<at::Tensor> importance_mask,                        \\
        const bool compute_densify_grad                                        \\
    );"""
    
    assert old_ins in src, "Could not find __INS__ macro"
    src = src.replace(old_ins, new_ins)
    
    # Add <cstdint> include for uint8_t
    if "#include <cstdint>" not in src:
        src = src.replace("#include <ATen/Dispatch.h>", "#include <ATen/Dispatch.h>\n#include <cstdint>")
    
    with open(fpath, 'w') as f:
        f.write(src)
    print(f"  Patched {fpath.name}")

def patch_rasterization_h():
    """Patch Rasterization.h — update declaration."""
    fpath = CSRC / "Rasterization.h"
    backup_file(fpath)
    
    with open(fpath, 'r') as f:
        src = f.read()
    
    old_decl = """template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets,    // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,     // [n_isects]
    // forward outputs
    const at::Tensor render_alphas,   // [..., image_height, image_width, 1]
    const at::Tensor last_ids,        // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
);"""
    
    new_decl = """template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets,    // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,     // [n_isects]
    // forward outputs
    const at::Tensor render_alphas,   // [..., image_height, image_width, 1]
    const at::Tensor last_ids,        // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad                 // B2 mode flag
);"""
    
    assert old_decl in src, "Could not find declaration in Rasterization.h"
    src = src.replace(old_decl, new_decl)
    
    with open(fpath, 'w') as f:
        f.write(src)
    print(f"  Patched {fpath.name}")

def patch_rasterization_cpp():
    """Patch Rasterization.cpp — update C++ binding."""
    fpath = CSRC / "Rasterization.cpp"
    backup_file(fpath)
    
    with open(fpath, 'r') as f:
        src = f.read()
    
    # 1. Update function signature
    old_sig = """std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, channels]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // options
    bool absgrad
) {"""
    
    new_sig = """std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, channels]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // options
    bool absgrad,
    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad                 // B2 mode flag
) {"""
    
    assert old_sig in src, "Could not find bwd function signature in Rasterization.cpp"
    src = src.replace(old_sig, new_sig)
    
    # 2. Update CHECK_INPUT for importance_mask
    old_check = """    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    uint32_t channels = colors.size(-1);

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);"""
    
    new_check = """    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    if (importance_mask.has_value()) {
        CHECK_INPUT(importance_mask.value());
    }

    uint32_t channels = colors.size(-1);

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);"""
    
    assert old_check in src, "Could not find CHECK_INPUT section"
    src = src.replace(old_check, new_check)
    
    # 3. Update __LAUNCH_KERNEL__ macro to pass importance_mask
    old_macro = """#define __LAUNCH_KERNEL__(N)                                                   \\
    case N:                                                                    \\
        launch_rasterize_to_pixels_3dgs_bwd_kernel<N>(                         \\
            means2d,                                                           \\
            conics,                                                            \\
            colors,                                                            \\
            opacities,                                                         \\
            backgrounds,                                                       \\
            masks,                                                             \\
            image_width,                                                       \\
            image_height,                                                      \\
            tile_size,                                                         \\
            tile_offsets,                                                      \\
            flatten_ids,                                                       \\
            render_alphas,                                                     \\
            last_ids,                                                          \\
            v_render_colors,                                                   \\
            v_render_alphas,                                                   \\
            absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt, \\
            v_means2d,                                                         \\
            v_conics,                                                          \\
            v_colors,                                                          \\
            v_opacities                                                        \\
        );                                                                     \\
        break;"""
    
    new_macro = """#define __LAUNCH_KERNEL__(N)                                                   \\
    case N:                                                                    \\
        launch_rasterize_to_pixels_3dgs_bwd_kernel<N>(                         \\
            means2d,                                                           \\
            conics,                                                            \\
            colors,                                                            \\
            opacities,                                                         \\
            backgrounds,                                                       \\
            masks,                                                             \\
            image_width,                                                       \\
            image_height,                                                      \\
            tile_size,                                                         \\
            tile_offsets,                                                      \\
            flatten_ids,                                                       \\
            render_alphas,                                                     \\
            last_ids,                                                          \\
            v_render_colors,                                                   \\
            v_render_alphas,                                                   \\
            absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt, \\
            v_means2d,                                                         \\
            v_conics,                                                          \\
            v_colors,                                                          \\
            v_opacities,                                                       \\
            importance_mask,                                                   \\
            compute_densify_grad                                               \\
        );                                                                     \\
        break;"""
    
    assert old_macro in src, "Could not find __LAUNCH_KERNEL__ macro in bwd"
    src = src.replace(old_macro, new_macro)
    
    with open(fpath, 'w') as f:
        f.write(src)
    print(f"  Patched {fpath.name}")

def patch_wrapper():
    """Patch _wrapper.py — update Python autograd."""
    fpath = CUDA_PKG / "_wrapper.py"
    backup_file(fpath)
    
    with open(fpath, 'r') as f:
        src = f.read()
    
    # 1. Update rasterize_to_pixels() signature
    old_sig = """def rasterize_to_pixels(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:"""
    
    new_sig = """def rasterize_to_pixels(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    importance_mask: Optional[Tensor] = None,  # [N] or [nnz], uint8
    compute_densify_grad: bool = False,  # B2 mode
) -> Tuple[Tensor, Tensor]:"""
    
    assert old_sig in src, "Could not find rasterize_to_pixels signature"
    src = src.replace(old_sig, new_sig)
    
    # 2. Update the _RasterizeToPixels.apply() call to pass mask
    old_apply = """    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
    )"""
    
    new_apply = """    if importance_mask is not None:
        importance_mask = importance_mask.contiguous()
    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        importance_mask,
        compute_densify_grad,
    )"""
    
    assert old_apply in src, "Could not find _RasterizeToPixels.apply call"
    src = src.replace(old_apply, new_apply)
    
    # 3. Update _RasterizeToPixels.forward() to accept and save mask
    old_fwd_sig = """    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [..., N, 2] or [nnz, 2]
        conics: Tensor,  # [..., N, 3] or [nnz, 3]
        colors: Tensor,  # [..., N, channels] or [nnz, channels]
        opacities: Tensor,  # [..., N] or [nnz]
        backgrounds: Tensor,  # [..., channels], Optional
        masks: Tensor,  # [..., tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [..., tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor]:"""
    
    new_fwd_sig = """    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [..., N, 2] or [nnz, 2]
        conics: Tensor,  # [..., N, 3] or [nnz, 3]
        colors: Tensor,  # [..., N, channels] or [nnz, channels]
        opacities: Tensor,  # [..., N] or [nnz]
        backgrounds: Tensor,  # [..., channels], Optional
        masks: Tensor,  # [..., tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [..., tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
        importance_mask: Tensor,  # [N] or [nnz], uint8, Optional
        compute_densify_grad: bool,  # B2 mode
    ) -> Tuple[Tensor, Tensor]:"""
    
    assert old_fwd_sig in src, "Could not find _RasterizeToPixels.forward signature"
    src = src.replace(old_fwd_sig, new_fwd_sig)
    
    # 4. Update save_for_backward and ctx attributes in forward
    old_save = """        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad"""
    
    new_save = """        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.importance_mask = importance_mask
        ctx.compute_densify_grad = compute_densify_grad"""
    
    assert old_save in src, "Could not find save_for_backward in forward"
    src = src.replace(old_save, new_save)
    
    # 5. Update backward to pass mask to CUDA
    old_bwd_call = """        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
        )"""
    
    new_bwd_call = """        importance_mask = ctx.importance_mask
        compute_densify_grad = ctx.compute_densify_grad
        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
            importance_mask,
            compute_densify_grad,
        )"""
    
    assert old_bwd_call in src, "Could not find backward CUDA call"
    src = src.replace(old_bwd_call, new_bwd_call)
    
    # 6. Update backward return tuple (add None for importance_mask and compute_densify_grad)
    old_return = """        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )"""
    
    new_return = """        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # importance_mask
            None,  # compute_densify_grad
        )"""
    
    assert old_return in src, "Could not find backward return tuple"
    src = src.replace(old_return, new_return)
    
    with open(fpath, 'w') as f:
        f.write(src)
    print(f"  Patched {fpath.name}")

def patch_rendering():
    """Patch rendering.py — add importance_mask to rasterization()."""
    fpath = RENDERING
    backup_file(fpath)
    
    with open(fpath, 'r') as f:
        src = f.read()
    
    # 1. Add importance_mask and compute_densify_grad to rasterization() signature
    old_sig = """    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor, Dict]:
    \"\"\"Rasterize a set of 3D Gaussians (N) to a batch of image planes (C)."""
    
    new_sig = """    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    importance_mask: Optional[Tensor] = None,  # [N], uint8, for sparse backward
    compute_densify_grad: bool = False,  # B2 mode: compute v_means2d for masked Gaussians
) -> Tuple[Tensor, Tensor, Dict]:
    \"\"\"Rasterize a set of 3D Gaussians (N) to a batch of image planes (C)."""
    
    assert old_sig in src, "Could not find rasterization signature in rendering.py"
    src = src.replace(old_sig, new_sig)
    
    # 2. Pass to the non-packed rasterize_to_pixels call (there are two: packed and non-packed)
    # Non-packed call (the one we use, packed=False):
    old_rtp_call = """            render_colors, render_alphas = rasterize_to_pixels(
                means2d,
                conics,
                colors,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                packed=packed,
                absgrad=absgrad,
            )"""
    
    new_rtp_call = """            render_colors, render_alphas = rasterize_to_pixels(
                means2d,
                conics,
                colors,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                packed=packed,
                absgrad=absgrad,
                importance_mask=importance_mask,
                compute_densify_grad=compute_densify_grad,
            )"""
    
    # This pattern appears twice (packed chunk loop and non-packed). Replace both.
    assert old_rtp_call in src, "Could not find rasterize_to_pixels call in rendering.py"
    src = src.replace(old_rtp_call, new_rtp_call)
    
    # Also patch the packed chunk loop version
    old_rtp_packed = """                render_colors_, render_alphas_ = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    packed=packed,
                    absgrad=absgrad,
                )"""
    
    new_rtp_packed = """                render_colors_, render_alphas_ = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    packed=packed,
                    absgrad=absgrad,
                    importance_mask=importance_mask,
                    compute_densify_grad=compute_densify_grad,
                )"""
    
    if old_rtp_packed in src:
        src = src.replace(old_rtp_packed, new_rtp_packed)
        print("  Also patched packed chunk loop variant")
    
    with open(fpath, 'w') as f:
        f.write(src)
    print(f"  Patched {fpath.name}")

def patch_ops_h():
    """Patch Ops.h — update the declaration that ext.cpp uses."""
    fpath = GSPLAT_ROOT / "gsplat" / "cuda" / "include" / "Ops.h"
    backup_file(fpath)
    
    with open(fpath, 'r') as f:
        src = f.read()
    
    old_decl = """    // options
    bool absgrad
);

// Rasterize 3D Gaussian, but only return the indices of gaussians and pixels."""
    
    new_decl = """    // options
    bool absgrad,
    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad                 // B2 mode flag
);

// Rasterize 3D Gaussian, but only return the indices of gaussians and pixels."""
    
    assert old_decl in src, "Could not find bwd declaration in Ops.h"
    src = src.replace(old_decl, new_decl)
    
    with open(fpath, 'w') as f:
        f.write(src)
    print(f"  Patched {fpath.name}")


def main():
    print("=" * 60)
    print("Phase C51 Stage 4A — CUDA Kernel Patch")
    print("=" * 60)
    
    print("\n1. Patching RasterizeToPixels3DGSBwd.cu (CUDA kernel)...")
    patch_bwd_kernel()
    
    print("\n2. Patching Rasterization.h (header)...")
    patch_rasterization_h()
    
    print("\n3. Patching Rasterization.cpp (C++ binding)...")
    patch_rasterization_cpp()
    
    print("\n4. Patching _wrapper.py (Python autograd)...")
    patch_wrapper()
    
    print("\n5. Patching rendering.py (high-level API)...")
    patch_rendering()
    
    print("\n6. Patching Ops.h (ext.cpp declaration)...")
    patch_ops_h()
    
    print("\n" + "=" * 60)
    print("All patches applied successfully!")
    print("Now rebuild gsplat with: cd /tmp/gsplat_baseline/gsplat-1.5.3 && pip install -e . --no-build-isolation")
    print("=" * 60)

if __name__ == "__main__":
    main()
